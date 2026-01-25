#!/usr/bin/env python3
"""
Tropical TN threshold analysis for rotated surface codes.

This module performs MAP decoding using tropical tensor networks
and generates threshold plots across different code distances and error rates.

Usage:
    uv run python scripts/analyze_tropical_threshold.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from bpdecoderplus.dem import (
    build_parity_check_matrix,
    dem_to_uai_for_decoding,
    load_dem,
)
from bpdecoderplus.syndrome import load_syndrome_database
from tropical_in_new.src import mpe_tropical
from tropical_in_new.src.utils import read_model_from_string

# Configuration
# Circuit-level depolarizing noise threshold for rotated surface code is ~0.7%.
# We scan around this threshold to observe the crossing behavior.
# Using smaller distances for tropical TN as it can be slow for large problems
DISTANCES = [3, 5]  # Tropical TN is efficient for small distances
ERROR_RATES = [0.001, 0.003, 0.005, 0.007, 0.010, 0.015]
# Sample sizes - d=5 is slower so use fewer samples
SAMPLE_SIZE_D3 = 500  # More samples for d=3 (fast)
SAMPLE_SIZE_D5 = 100  # Fewer samples for d=5 (slower)
SAMPLE_SIZE = 500  # Default for backward compatibility


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
        obs_flip: Observable flip probabilities, shape (n_errors,)

    Returns:
        Tuple of (solution, predicted_observable) where:
        - solution: Binary error pattern, shape (n_errors,)
        - predicted_observable: Predicted observable value (0 or 1)
    """
    n_errors = H.shape[1]

    # Build UAI model string from H, priors, and syndrome
    uai_str = _build_decoding_uai_from_matrix(H, priors, syndrome)

    # Parse the UAI model
    model = read_model_from_string(uai_str)

    # Run tropical MPE inference
    assignment, score, info = mpe_tropical(model)

    # Convert 1-indexed assignment to 0-indexed error vector
    # UAI format uses 0-indexed variables, but tropical_in_new uses 1-indexed internally
    solution = np.zeros(n_errors, dtype=np.int32)
    for i in range(n_errors):
        # Variables are 1-indexed in the assignment dict
        solution[i] = assignment.get(i + 1, 0)

    # Compute observable prediction using soft XOR
    predicted_obs = compute_observable_prediction(solution, obs_flip)

    return solution, predicted_obs


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

    for i, syndrome in enumerate(syndromes):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processing sample {i + 1}/{n_samples}...")

        try:
            _, predicted_obs = run_tropical_decoder(H, syndrome, priors, obs_flip)
            if predicted_obs != observables[i]:
                total_errors += 1
        except Exception as e:
            # If decoding fails, count as error
            print(f"    Warning: Decoding failed for sample {i}: {e}")
            total_errors += 1

    return total_errors / n_samples


def collect_tropical_threshold_data(max_samples: int = SAMPLE_SIZE):
    """
    Collect logical error rates for tropical TN decoder across distances and error rates.

    Args:
        max_samples: Maximum samples per dataset (may be overridden per-distance)

    Returns:
        Dict mapping distance -> {error_rate: ler}
    """
    results = {}

    for d in DISTANCES:
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

    This function collects threshold data for d=3,5 and generates
    the threshold plot (tropical_threshold_plot.png).
    """
    print("=" * 60)
    print("Tropical TN MAP Decoder Threshold Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Max samples per dataset: {SAMPLE_SIZE}")
    print("\nNote: Tropical TN is efficient for small distances (d=3,5)")
    print("      but may be slow for larger distances (d>=7).")

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
