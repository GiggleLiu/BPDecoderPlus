#!/usr/bin/env python3
"""
Tropical TN threshold analysis for rotated surface codes.

This module performs MAP decoding using tropical tensor networks
and generates threshold plots across different code distances and error rates.

Usage:
    uv run python scripts/analyze_tropical_threshold.py
"""
import gc
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from bpdecoderplus.dem import (
    build_parity_check_matrix,
    dem_to_uai_for_decoding,
    load_dem,
)
from bpdecoderplus.syndrome import load_syndrome_database
from tropical_in_new.src import mpe_tropical
from tropical_in_new.src.utils import read_model_from_string, Factor, UAIModel

# Configuration
# Circuit-level depolarizing noise threshold for rotated surface code is ~0.7%.
# We scan around this threshold to observe the crossing behavior.
#
# NOTE: Exact tropical tensor network contraction has high memory requirements.
# d=3 works well, but d=5 requires >16GB RAM due to tensor network treewidth.
# On memory-constrained systems, only d=3 is enabled by default.
DISTANCES = [3]  # d=5 requires >16GB RAM for exact tropical contraction
ERROR_RATES = [0.001, 0.003, 0.005, 0.007, 0.010, 0.015]
# Sample sizes - adjust based on available memory and time
SAMPLE_SIZE_D3 = 500  # More samples for d=3 (fast)
SAMPLE_SIZE_D5 = 100  # Fewer samples for d=5 (if enabled)
SAMPLE_SIZE = 500  # Default for backward compatibility

# Memory limit check for d=5 (set to True to attempt d=5 on high-memory systems)
ENABLE_D5 = False


def compute_observable_prediction(solution: np.ndarray, obs_flip: np.ndarray) -> int:
    """
    Compute observable prediction using soft XOR probability chain.

    When hyperedges are merged, obs_flip stores conditional probabilities
    P(obs flip | hyperedge fires). This function correctly computes
    P(odd number of observable flips) by chaining XOR probabilities.

    Args:
        solution: Binary error pattern from decoder
        obs_flip: Observable flip probabilities (0.0 to 1.0)

    Returns:
        Predicted observable value (0 or 1)
    """
    p_flip = 0.0
    for i in range(len(solution)):
        if solution[i] == 1:
            # XOR probability: P(odd flips so far) XOR P(this flips)
            # P(A XOR B) = P(A)(1-P(B)) + P(B)(1-P(A))
            p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
    return int(p_flip > 0.5)


class PrecomputedModel:
    """
    Memory-efficient model that precomputes static structure.

    Instead of rebuilding the entire UAI string for each syndrome,
    we precompute the static parts (priors, constraint scopes) and
    only update constraint factor values based on the syndrome.

    Large parity constraints (>3 variables) are decomposed into chains
    of smaller XOR factors using auxiliary variables. This reduces
    treewidth and prevents memory explosion during contraction.
    """

    # Maximum constraint size before decomposition
    # Use 2 for aggressive decomposition (smallest possible treewidth contribution)
    MAX_DIRECT_VARS = 2

    def __init__(self, H: np.ndarray, priors: np.ndarray):
        """
        Initialize precomputed model structure.

        Args:
            H: Parity check matrix, shape (n_detectors, n_errors)
            priors: Prior error probabilities, shape (n_errors,)
        """
        self.n_detectors, self.n_errors = H.shape
        self.priors = priors

        # Precompute constraint scopes (which errors each detector connects to)
        self.constraint_scopes = []
        for d in range(self.n_detectors):
            error_indices = np.where(H[d, :] == 1)[0]
            self.constraint_scopes.append(tuple(error_indices))

        # Count auxiliary variables needed for decomposition
        self.n_aux = 0
        self.decomposition_info = []  # List of (needs_decomp, n_aux_for_this)
        for scope in self.constraint_scopes:
            n_vars = len(scope)
            if n_vars > self.MAX_DIRECT_VARS:
                # Need (n_vars - 2) auxiliary variables for chain decomposition
                n_aux_needed = n_vars - 2
                self.decomposition_info.append((True, n_aux_needed, self.n_aux))
                self.n_aux += n_aux_needed
            else:
                self.decomposition_info.append((False, 0, 0))

        self.total_vars = self.n_errors + self.n_aux

        # Precompute prior factors (these never change) - use float32 to save memory
        self.prior_factors = []
        for i in range(self.n_errors):
            p = float(priors[i])
            # vars are 1-indexed in tropical library
            factor = Factor(
                vars=(i + 1,),
                values=torch.tensor([1.0 - p, p], dtype=torch.float32)
            )
            self.prior_factors.append(factor)

        # Prior factors for auxiliary variables (uniform, p=0.5)
        for i in range(self.n_aux):
            factor = Factor(
                vars=(self.n_errors + i + 1,),
                values=torch.tensor([0.5, 0.5], dtype=torch.float32)
            )
            self.prior_factors.append(factor)

        # Precompute 3-variable XOR lookup table (used for chain decomposition)
        # XOR(a, b, c) = 0 means even parity among a, b, c
        self.xor3_even = torch.zeros((2, 2, 2), dtype=torch.float32)
        self.xor3_odd = torch.zeros((2, 2, 2), dtype=torch.float32)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    parity = (a + b + c) % 2
                    if parity == 0:
                        self.xor3_even[a, b, c] = 1.0
                        self.xor3_odd[a, b, c] = 1e-30
                    else:
                        self.xor3_even[a, b, c] = 1e-30
                        self.xor3_odd[a, b, c] = 1.0

        # Precompute small constraint lookup tables (for direct constraints)
        self.small_parity_tables = {}
        for n in range(1, self.MAX_DIRECT_VARS + 1):
            n_entries = 2 ** n
            even_table = np.zeros(n_entries, dtype=np.float32)
            odd_table = np.zeros(n_entries, dtype=np.float32)
            for i in range(n_entries):
                parity = bin(i).count("1") % 2
                if parity == 0:
                    even_table[i] = 1.0
                    odd_table[i] = 1e-30
                else:
                    even_table[i] = 1e-30
                    odd_table[i] = 1.0
            self.small_parity_tables[n] = (
                torch.from_numpy(even_table.reshape((2,) * n)),
                torch.from_numpy(odd_table.reshape((2,) * n))
            )

    def build_model_for_syndrome(self, syndrome: np.ndarray) -> UAIModel:
        """
        Build UAI model for a specific syndrome, reusing precomputed structure.

        Large parity constraints are decomposed into chains:
        XOR(x1, x2, ..., xn) = s becomes:
          aux1 = XOR(x1, x2)  (implicit, enforced by factor)
          aux2 = XOR(aux1, x3)
          ...
          XOR(aux_{n-2}, x_{n-1}, x_n) = s

        Args:
            syndrome: Binary syndrome, shape (n_detectors,)

        Returns:
            UAIModel ready for tropical inference
        """
        factors = list(self.prior_factors)  # Start with prior factors

        # Build constraint factors based on syndrome
        for d, scope in enumerate(self.constraint_scopes):
            n_vars = len(scope)
            if n_vars == 0:
                # Empty scope - trivial factor
                factor = Factor(vars=(), values=torch.tensor(1.0, dtype=torch.float32))
                factors.append(factor)
                continue

            syndrome_bit = int(syndrome[d])
            needs_decomp, n_aux_needed, aux_start = self.decomposition_info[d]

            if not needs_decomp:
                # Small constraint - use direct lookup table
                even_table, odd_table = self.small_parity_tables[n_vars]
                values = even_table if syndrome_bit == 0 else odd_table
                vars_1indexed = tuple(e + 1 for e in scope)
                factor = Factor(vars=vars_1indexed, values=values)
                factors.append(factor)
            else:
                # Large constraint - decompose into chain of XOR3 factors
                # aux_i are 1-indexed as (n_errors + aux_start + i + 1)
                vars_list = [e + 1 for e in scope]  # Convert to 1-indexed

                # First factor: XOR(x1, x2, aux1) = 0 (defines aux1 = x1 XOR x2)
                aux1_var = self.n_errors + aux_start + 1
                factor = Factor(
                    vars=(vars_list[0], vars_list[1], aux1_var),
                    values=self.xor3_even  # aux1 = XOR(x1, x2), so parity is 0
                )
                factors.append(factor)

                # Middle factors: XOR(aux_i, x_{i+2}, aux_{i+1}) = 0
                for i in range(n_aux_needed - 1):
                    aux_in = self.n_errors + aux_start + i + 1
                    aux_out = self.n_errors + aux_start + i + 2
                    x_var = vars_list[i + 2]
                    factor = Factor(
                        vars=(aux_in, x_var, aux_out),
                        values=self.xor3_even
                    )
                    factors.append(factor)

                # Final factor: XOR(aux_{n-2}, x_{n-1}, x_n) = syndrome_bit
                aux_last = self.n_errors + aux_start + n_aux_needed
                x_second_last = vars_list[-2]
                x_last = vars_list[-1]
                values = self.xor3_even if syndrome_bit == 0 else self.xor3_odd
                factor = Factor(
                    vars=(aux_last, x_second_last, x_last),
                    values=values
                )
                factors.append(factor)

        return UAIModel(
            nvars=self.total_vars,
            cards=[2] * self.total_vars,
            factors=factors
        )


def _build_decoding_uai_from_matrix(
    H: np.ndarray,
    priors: np.ndarray,
    syndrome: np.ndarray,
) -> str:
    """
    Build UAI model string for MAP decoding from parity check matrix.

    Creates a factor graph where:
    - Variables = error bits (columns of H)
    - Prior factors = error probabilities
    - Constraint factors = syndrome parity checks

    Args:
        H: Parity check matrix, shape (n_detectors, n_errors)
        priors: Prior error probabilities, shape (n_errors,)
        syndrome: Binary syndrome, shape (n_detectors,)

    Returns:
        UAI format string.
    """
    n_detectors, n_errors = H.shape

    lines = []

    # UAI header
    lines.append("MARKOV")
    lines.append(str(n_errors))
    lines.append(" ".join(["2"] * n_errors))

    # Count factors: n_errors prior factors + n_detectors constraint factors
    n_factors = n_errors + n_detectors
    lines.append(str(n_factors))

    # Factor scopes
    # Prior factors (each covers one error variable)
    for i in range(n_errors):
        lines.append(f"1 {i}")

    # Constraint factors (each covers errors connected to a detector)
    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        n_vars = len(error_indices)
        if n_vars > 0:
            scope_str = " ".join(str(e) for e in error_indices)
            lines.append(f"{n_vars} {scope_str}")
        else:
            lines.append("0")

    lines.append("")

    # Factor values
    # Prior factors
    for i in range(n_errors):
        p = priors[i]
        lines.append("2")
        lines.append(str(1.0 - p))
        lines.append(str(p))
        lines.append("")

    # Constraint factors
    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        n_vars = len(error_indices)
        if n_vars > 0:
            syndrome_bit = int(syndrome[d])
            n_entries = 2**n_vars
            lines.append(str(n_entries))
            for i in range(n_entries):
                parity = bin(i).count("1") % 2
                if parity == syndrome_bit:
                    lines.append("1.0")
                else:
                    lines.append("1e-30")
            lines.append("")
        else:
            lines.append("1")
            lines.append("1.0")
            lines.append("")

    return "\n".join(lines)


def run_tropical_decoder(
    H: np.ndarray,
    syndrome: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    precomputed_model: PrecomputedModel | None = None,
) -> tuple[np.ndarray, int]:
    """
    Run tropical TN MAP decoder on a single syndrome.

    Constructs a UAI model from the parity check matrix and syndrome,
    then uses tropical tensor network contraction to find the MPE assignment.

    Args:
        H: Parity check matrix, shape (n_detectors, n_errors)
        syndrome: Binary syndrome, shape (n_detectors,)
        priors: Prior error probabilities, shape (n_errors,)
        obs_flip: Observable flip probabilities, shape (n_errors,)
        precomputed_model: Optional precomputed model for memory efficiency

    Returns:
        Tuple of (solution, predicted_observable) where:
        - solution: Binary error pattern, shape (n_errors,)
        - predicted_observable: Predicted observable value (0 or 1)
    """
    n_errors = H.shape[1]

    # Use precomputed model if available (memory efficient)
    if precomputed_model is not None:
        model = precomputed_model.build_model_for_syndrome(syndrome)
    else:
        # Fallback to string-based construction
        uai_str = _build_decoding_uai_from_matrix(H, priors, syndrome)
        model = read_model_from_string(uai_str)

    # Run tropical MPE inference
    assignment, score, info = mpe_tropical(model)

    # Convert 1-indexed assignment to 0-indexed error vector
    # Only extract the original error variables, not auxiliary ones
    # UAI format uses 0-indexed variables, but tropical_in_new uses 1-indexed internally
    solution = np.zeros(n_errors, dtype=np.int32)
    for i in range(n_errors):
        # Variables are 1-indexed in the assignment dict
        solution[i] = assignment.get(i + 1, 0)

    # Compute observable prediction using soft XOR
    predicted_obs = compute_observable_prediction(solution, obs_flip)

    return solution, predicted_obs


def load_dataset(distance: int, error_rate: float):
    """
    Load dataset for given distance and error rate.

    Args:
        distance: Code distance
        error_rate: Physical error rate

    Returns:
        Tuple of (H, syndromes, observables, priors, obs_flip) or None if not found
    """
    rounds = distance
    p_str = f"{error_rate:.4f}"[2:]
    base_name = f"sc_d{distance}_r{rounds}_p{p_str}_z"

    dem_path = Path(f"datasets/{base_name}.dem")
    npz_path = Path(f"datasets/{base_name}.npz")

    if not dem_path.exists() or not npz_path.exists():
        return None

    dem = load_dem(str(dem_path))
    syndromes, observables, _ = load_syndrome_database(str(npz_path))
    H, priors, obs_flip = build_parity_check_matrix(dem)

    return H, syndromes, observables, priors, obs_flip


def run_tropical_decoder_batch(
    H: np.ndarray,
    syndromes: np.ndarray,
    observables: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    verbose: bool = False,
) -> float:
    """
    Run tropical TN decoder on a batch of syndromes.

    Uses memory-efficient precomputed model structure and explicit
    garbage collection to avoid OOM on larger instances.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        priors: Prior error probabilities
        obs_flip: Observable flip probabilities
        verbose: Whether to print progress

    Returns:
        Logical error rate
    """
    total_errors = 0
    n_samples = len(syndromes)

    # Precompute model structure once (major memory optimization for d=5+)
    precomputed = PrecomputedModel(H, priors)

    # Determine GC frequency based on problem size
    # For larger problems (d=5), clean up more frequently
    gc_frequency = 10 if H.shape[1] > 200 else 50

    for i, syndrome in enumerate(syndromes):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processing sample {i + 1}/{n_samples}...")

        try:
            _, predicted_obs = run_tropical_decoder(
                H, syndrome, priors, obs_flip,
                precomputed_model=precomputed
            )
            if predicted_obs != observables[i]:
                total_errors += 1
        except MemoryError:
            # Memory exhausted - abort this batch
            print(f"\n    MemoryError at sample {i}: tensor network too large")
            print("    Consider reducing problem size or increasing available RAM")
            return float("nan")  # Signal failure
        except Exception as e:
            # If decoding fails, count as error
            print(f"    Warning: Decoding failed for sample {i}: {e}")
            total_errors += 1

        # Explicit garbage collection to prevent memory buildup
        # The tropical contraction creates many intermediate tensors
        if (i + 1) % gc_frequency == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_errors / n_samples


def collect_tropical_threshold_data(max_samples: int = SAMPLE_SIZE, distances: list[int] | None = None):
    """
    Collect logical error rates for tropical TN decoder across distances and error rates.

    Args:
        max_samples: Maximum samples per dataset (may be overridden per-distance)
        distances: List of distances to test (defaults to DISTANCES)

    Returns:
        Dict mapping distance -> {error_rate: ler}
    """
    if distances is None:
        distances = DISTANCES

    results = {}

    for d in distances:
        results[d] = {}
        print(f"\nDistance d={d}:")

        # Use distance-specific sample size if available
        if d == 3:
            effective_max = min(max_samples, SAMPLE_SIZE_D3)
        elif d == 5:
            effective_max = min(max_samples, SAMPLE_SIZE_D5)
        else:
            effective_max = min(max_samples, 50)  # Very few for d>=7

        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"  p={p}: Dataset not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip = data
            num_samples = min(effective_max, len(syndromes))

            print(f"  p={p}: Decoding {num_samples} samples...", end=" ", flush=True)

            ler = run_tropical_decoder_batch(
                H,
                syndromes[:num_samples],
                observables[:num_samples],
                priors,
                obs_flip,
                verbose=False,
            )

            if np.isnan(ler):
                print(f"FAILED (memory)")
                # Skip remaining error rates for this distance
                break
            else:
                results[d][p] = ler
                print(f"LER={ler:.4f}")

    return results


def plot_threshold_curve(results: dict, output_path: str = "outputs/tropical_threshold_plot.png"):
    """
    Plot logical error rate vs physical error rate for different distances.

    Args:
        results: Dict mapping distance -> {error_rate: ler}
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, d in enumerate(sorted(results.keys())):
        if not results[d]:
            continue
        error_rates = sorted(results[d].keys())
        lers = [results[d][p] for p in error_rates]
        plt.plot(
            error_rates,
            lers,
            f"{markers[i % len(markers)]}-",
            color=colors[i % len(colors)],
            label=f"d={d}",
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Physical Error Rate (p)", fontsize=12)
    plt.ylabel("Logical Error Rate", fontsize=12)
    plt.title("Tropical TN MAP Decoder Threshold", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.xscale("log")

    # Add threshold region annotation
    plt.axvline(x=0.007, color="gray", linestyle="--", alpha=0.5, label="p=0.7%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nThreshold plot saved to: {output_path}")


def main():
    """
    Generate threshold plots for tropical TN MAP decoder.

    This function collects threshold data for d=3 (and optionally d=5) and generates
    the threshold plot (tropical_threshold_plot.png).

    Environment variables:
        TROPICAL_ENABLE_D5: Set to "1" to enable d=5 (requires >16GB RAM)
    """
    import os

    # Check if d=5 should be enabled
    distances = list(DISTANCES)
    enable_d5 = ENABLE_D5 or os.environ.get("TROPICAL_ENABLE_D5", "").lower() in ("1", "true", "yes")
    if enable_d5 and 5 not in distances:
        distances.append(5)
        distances.sort()

    print("=" * 60)
    print("Tropical TN MAP Decoder Threshold Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Distances: {distances}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Max samples per dataset: {SAMPLE_SIZE}")
    if 5 not in distances:
        print("\nNote: d=5 disabled (requires >16GB RAM for exact tropical contraction)")
        print("      Set TROPICAL_ENABLE_D5=1 to enable on high-memory systems.")
    else:
        print("\nWarning: d=5 enabled - requires >16GB RAM for exact tropical contraction")

    print("\nCollecting threshold data...")
    results = collect_tropical_threshold_data(max_samples=SAMPLE_SIZE, distances=distances)

    # Check we have at least some data
    total_points = sum(len(v) for v in results.values())
    if total_points == 0:
        print("\nError: No threshold data collected - check that datasets exist")
        print("Run 'python scripts/generate_threshold_datasets.py' first if needed.")
        return

    print(f"\nCollected {total_points} data points")

    # Generate threshold plot
    plot_threshold_curve(results, "outputs/tropical_threshold_plot.png")

    # Print summary
    print("\n" + "=" * 60)
    print("Tropical TN Threshold Analysis Summary")
    print("=" * 60)
    print(f"{'Distance':<10} {'Error Rates Tested':<25} {'Min LER':<12} {'Max LER':<12}")
    print("-" * 60)
    for d in sorted(results.keys()):
        if results[d]:
            error_rates = sorted(results[d].keys())
            lers = [results[d][p] for p in error_rates]
            er_str = f"{len(error_rates)} points ({min(error_rates):.3f}-{max(error_rates):.3f})"
            print(f"d={d:<8} {er_str:<25} {min(lers):<12.4f} {max(lers):<12.4f}")


if __name__ == "__main__":
    main()
