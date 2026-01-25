#!/usr/bin/env python3
"""
Tropical TN threshold analysis for rotated surface codes.

This module performs MAP decoding using tropical tensor networks
and generates threshold plots across different code distances and error rates.

IMPORTANT: MAP vs Marginal Decoding
-----------------------------------
The tropical tensor network performs exact MAP (Maximum A Posteriori) inference,
finding the single most likely error pattern. However, for quantum error correction,
what matters is the most likely OBSERVABLE VALUE, not the most likely error pattern.

When multiple error patterns satisfy the syndrome, the MAP pattern may predict the
wrong observable even when the total probability mass favors the correct one. This
is why BP+OSD (which uses marginal probabilities) typically achieves lower logical
error rates than pure MAP decoding.

This script demonstrates tropical tensor network MAP decoding for educational purposes.
For production QEC applications, marginal-based decoders like BP+OSD are preferred.

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

from bpdecoderplus.dem import build_parity_check_matrix, load_dem
from bpdecoderplus.syndrome import load_syndrome_database
from tropical_in_new.src import mpe_tropical
from tropical_in_new.src.utils import read_model_from_string

# Configuration
# Circuit-level depolarizing noise threshold for rotated surface code is ~0.7%.
# We scan around this threshold to observe the crossing behavior.
#
# NOTE: Exact tropical tensor network contraction has high memory requirements.
# d=3 works well, but d=5 requires >16GB RAM due to tensor network treewidth.
DISTANCES = [3]  # d=5 requires >16GB RAM for exact tropical contraction
ERROR_RATES = [0.001, 0.003, 0.005, 0.007, 0.010, 0.015]
SAMPLE_SIZE = 500


def compute_observable_prediction(solution: np.ndarray, obs_flip: np.ndarray) -> int:
    """
    Compute observable prediction using mod-2 arithmetic.

    For MAP decoding (tropical tensor network), the solution is a deterministic
    binary error pattern. The observable prediction is the parity (XOR) of
    observable flips for all errors in the solution.

    Args:
        solution: Binary error pattern from decoder
        obs_flip: Observable flip indicators (thresholded at 0.5 for soft values)

    Returns:
        Predicted observable value (0 or 1)
    """
    # Threshold obs_flip at 0.5 for soft values from hyperedge merging
    obs_flip_binary = (obs_flip > 0.5).astype(int)
    # Compute mod-2 dot product (XOR of all flipped observables)
    return int(np.dot(solution, obs_flip_binary) % 2)


def build_decoding_uai_from_matrix(
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
) -> tuple[np.ndarray, int]:
    """
    Run tropical TN MAP decoder on a single syndrome.

    Constructs a UAI model from the parity check matrix and syndrome,
    then uses tropical tensor network contraction to find the MPE assignment.

    Args:
        H: Parity check matrix, shape (n_detectors, n_errors)
        syndrome: Binary syndrome, shape (n_detectors,)
        priors: Prior error probabilities, shape (n_errors,)
        obs_flip: Observable flip indicators, shape (n_errors,)

    Returns:
        Tuple of (solution, predicted_observable) where:
        - solution: Binary error pattern, shape (n_errors,)
        - predicted_observable: Predicted observable value (0 or 1)
    """
    n_errors = H.shape[1]

    # Build UAI model string
    uai_str = build_decoding_uai_from_matrix(H, priors, syndrome)
    model = read_model_from_string(uai_str)

    # Run tropical MPE inference
    assignment, score, info = mpe_tropical(model)

    # Convert 1-indexed assignment to 0-indexed error vector
    # UAI format uses 0-indexed variables, but tropical_in_new uses 1-indexed internally
    # Variables not in assignment default to 0 (most likely value for small priors)
    solution = np.zeros(n_errors, dtype=np.int32)
    for i in range(n_errors):
        solution[i] = assignment.get(i + 1, 0)

    # Compute observable prediction using mod-2 arithmetic
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

    # GC frequency based on problem size
    gc_frequency = 10 if H.shape[1] > 200 else 50

    for i, syndrome in enumerate(syndromes):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processing sample {i + 1}/{n_samples}...")

        try:
            _, predicted_obs = run_tropical_decoder(H, syndrome, priors, obs_flip)
            if predicted_obs != observables[i]:
                total_errors += 1
        except MemoryError:
            print(f"\n    MemoryError at sample {i}: tensor network too large")
            print("    Consider reducing problem size or increasing available RAM")
            return float("nan")
        except Exception as e:
            print(f"    Warning: Decoding failed for sample {i}: {e}")
            total_errors += 1

        # Explicit garbage collection to prevent memory buildup
        if (i + 1) % gc_frequency == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_errors / n_samples


def collect_tropical_threshold_data(max_samples: int = SAMPLE_SIZE):
    """
    Collect logical error rates for tropical TN decoder across distances and error rates.

    Args:
        max_samples: Maximum samples per dataset

    Returns:
        Dict mapping distance -> {error_rate: ler}
    """
    results = {}

    for d in DISTANCES:
        results[d] = {}
        print(f"\nDistance d={d}:")

        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"  p={p}: Dataset not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip = data
            num_samples = min(max_samples, len(syndromes))

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
                print("FAILED (memory)")
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
    """
    print("=" * 60)
    print("Tropical TN MAP Decoder Threshold Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Max samples per dataset: {SAMPLE_SIZE}")

    print("\nCollecting threshold data...")
    results = collect_tropical_threshold_data(max_samples=SAMPLE_SIZE)

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
