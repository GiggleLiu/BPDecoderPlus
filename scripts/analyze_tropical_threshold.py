#!/usr/bin/env python3
"""
Tropical TN threshold analysis for rotated surface codes.

This module performs MAP decoding using tropical tensor networks
and generates threshold plots across different code distances and error rates.

IMPORTANT: This decoder uses the DEM functions from bpdecoderplus.dem
-----------------------------------------------------------------------------
The tropical TN decoder uses `build_parity_check_matrix` with `merge_hyperedges=True`
for efficient computation. Key implementation details:
1. merge_hyperedges=True creates a smaller matrix (faster contraction)
2. obs_flip becomes a conditional probability, thresholded at 0.5 for prediction
3. The connected components fix ensures all factors are included in contraction

The tropical tensor network performs exact MAP inference on the factor graph.
Results should be similar to MWPM, though not identical due to different
graph structures and degeneracy handling.

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

from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from tropical_in_new.src import mpe_tropical
from tropical_in_new.src.utils import read_model_from_string

# Optional: pymatching for comparison
try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

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
        obs_flip: Observable flip indicators (may be soft values from hyperedge merging)

    Returns:
        Predicted observable value (0 or 1)
    """
    # Threshold obs_flip at 0.5 to convert soft probabilities to binary
    # This handles both binary obs_flip (merge_hyperedges=False) and
    # soft obs_flip (merge_hyperedges=True) correctly
    obs_flip_binary = (obs_flip > 0.5).astype(int)
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
        obs_flip: Binary observable flip indicators, shape (n_errors,)

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

    Uses build_parity_check_matrix from bpdecoderplus.dem with merge_hyperedges=False
    to ensure binary obs_flip values for correct observable prediction.

    Args:
        distance: Code distance
        error_rate: Physical error rate

    Returns:
        Tuple of (H, syndromes, observables, priors, obs_flip, dem) or None if not found
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

    # Build parity check matrix with merge_hyperedges=True for faster computation
    # When merge_hyperedges=True, obs_flip becomes a conditional probability
    # which we threshold at 0.5 in compute_observable_prediction
    H, priors, obs_flip = build_parity_check_matrix(
        dem, 
        split_by_separator=True,
        merge_hyperedges=True,  # Faster computation with smaller matrix
    )

    return H, syndromes, observables, priors, obs_flip, dem


def run_tropical_decoder_batch(
    H: np.ndarray,
    syndromes: np.ndarray,
    observables: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    dem=None,
    verbose: bool = False,
    compare_mwpm: bool = True,
) -> tuple[float, float, int]:
    """
    Run tropical TN decoder on a batch of syndromes.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        priors: Prior error probabilities
        obs_flip: Binary observable flip indicators
        dem: Detector error model for MWPM comparison (optional)
        verbose: Whether to print progress
        compare_mwpm: Whether to compare with MWPM

    Returns:
        Tuple of (tropical_ler, mwpm_ler, num_differs) where:
        - tropical_ler: Tropical TN logical error rate
        - mwpm_ler: MWPM logical error rate (0 if pymatching not available)
        - num_differs: Number of samples where predictions differ
    """
    tropical_errors = 0
    mwpm_errors = 0
    differs = 0
    n_samples = len(syndromes)

    # Get MWPM predictions if pymatching is available
    mwpm_preds = None
    if HAS_PYMATCHING and dem is not None and compare_mwpm:
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_preds = matcher.decode_batch(syndromes)
        if mwpm_preds.ndim > 1:
            mwpm_preds = mwpm_preds.flatten()
        mwpm_errors = np.sum(mwpm_preds != observables)

    # GC frequency based on problem size
    gc_frequency = 10 if H.shape[1] > 200 else 50

    for i, syndrome in enumerate(syndromes):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processing sample {i + 1}/{n_samples}...")

        try:
            _, predicted_obs = run_tropical_decoder(H, syndrome, priors, obs_flip)
            if predicted_obs != observables[i]:
                tropical_errors += 1
            
            # Compare with MWPM
            if mwpm_preds is not None and predicted_obs != mwpm_preds[i]:
                differs += 1
                
        except MemoryError:
            print(f"\n    MemoryError at sample {i}: tensor network too large")
            print("    Consider reducing problem size or increasing available RAM")
            return float("nan"), mwpm_errors / n_samples if mwpm_preds is not None else 0.0, differs
        except Exception as e:
            print(f"    Warning: Decoding failed for sample {i}: {e}")
            tropical_errors += 1

        # Explicit garbage collection to prevent memory buildup
        if (i + 1) % gc_frequency == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tropical_ler = tropical_errors / n_samples
    mwpm_ler = mwpm_errors / n_samples if mwpm_preds is not None else 0.0
    return tropical_ler, mwpm_ler, differs


def collect_tropical_threshold_data(max_samples: int = SAMPLE_SIZE):
    """
    Collect logical error rates for tropical TN decoder across distances and error rates.

    Also compares with MWPM to verify consistency.

    Args:
        max_samples: Maximum samples per dataset

    Returns:
        Tuple of (tropical_results, mwpm_results) where each is
        Dict mapping distance -> {error_rate: ler}
    """
    tropical_results = {}
    mwpm_results = {}

    for d in DISTANCES:
        tropical_results[d] = {}
        mwpm_results[d] = {}
        print(f"\nDistance d={d}:")

        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"  p={p}: Dataset not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip, dem = data
            num_samples = min(max_samples, len(syndromes))

            print(f"  p={p}: Decoding {num_samples} samples (H shape: {H.shape})...", end=" ", flush=True)

            tropical_ler, mwpm_ler, differs = run_tropical_decoder_batch(
                H,
                syndromes[:num_samples],
                observables[:num_samples],
                priors,
                obs_flip,
                dem=dem,
                verbose=False,
            )

            if np.isnan(tropical_ler):
                print("FAILED (memory)")
                break
            else:
                tropical_results[d][p] = tropical_ler
                mwpm_results[d][p] = mwpm_ler
                mwpm_info = f", MWPM LER={mwpm_ler:.4f}, differs={differs}" if HAS_PYMATCHING else ""
                print(f"Tropical LER={tropical_ler:.4f}{mwpm_info}")

    return tropical_results, mwpm_results


def plot_threshold_curve(
    tropical_results: dict,
    mwpm_results: dict,
    output_path: str = "outputs/tropical_threshold_plot.png"
):
    """
    Plot logical error rate vs physical error rate for both Tropical TN and MWPM.

    Args:
        tropical_results: Dict mapping distance -> {error_rate: ler} for Tropical TN
        mwpm_results: Dict mapping distance -> {error_rate: ler} for MWPM
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, d in enumerate(sorted(tropical_results.keys())):
        if not tropical_results[d]:
            continue
        error_rates = sorted(tropical_results[d].keys())
        
        # Tropical TN (solid line)
        tropical_lers = [tropical_results[d][p] for p in error_rates]
        plt.plot(
            error_rates,
            tropical_lers,
            f"{markers[i % len(markers)]}-",
            color=colors[i % len(colors)],
            label=f"d={d} Tropical",
            linewidth=2,
            markersize=8,
        )
        
        # MWPM (dashed line) if available
        if d in mwpm_results and mwpm_results[d]:
            mwpm_lers = [mwpm_results[d][p] for p in error_rates]
            if any(l > 0 for l in mwpm_lers):  # Only plot if we have MWPM data
                plt.plot(
                    error_rates,
                    mwpm_lers,
                    f"{markers[i % len(markers)]}--",
                    color=colors[i % len(colors)],
                    label=f"d={d} MWPM",
                    linewidth=2,
                    markersize=6,
                    alpha=0.7,
                )

    plt.xlabel("Physical Error Rate (p)", fontsize=12)
    plt.ylabel("Logical Error Rate", fontsize=12)
    plt.title("Tropical TN vs MWPM Decoder Comparison", fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.xscale("log")

    # Add threshold region annotation
    plt.axvline(x=0.007, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nThreshold plot saved to: {output_path}")


def main():
    """
    Generate threshold plots for tropical TN MAP decoder with MWPM comparison.
    """
    print("=" * 60)
    print("Tropical TN MAP Decoder Threshold Analysis")
    print("(Using bpdecoderplus.dem for parity check matrix)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Max samples per dataset: {SAMPLE_SIZE}")
    print(f"  pymatching available: {HAS_PYMATCHING}")

    print("\nCollecting threshold data...")
    tropical_results, mwpm_results = collect_tropical_threshold_data(max_samples=SAMPLE_SIZE)

    # Check we have at least some data
    total_points = sum(len(v) for v in tropical_results.values())
    if total_points == 0:
        print("\nError: No threshold data collected - check that datasets exist")
        print("Run 'python scripts/generate_threshold_datasets.py' first if needed.")
        return

    print(f"\nCollected {total_points} data points")

    # Generate threshold plot
    plot_threshold_curve(tropical_results, mwpm_results, "outputs/tropical_threshold_plot.png")

    # Print summary
    print("\n" + "=" * 60)
    print("Tropical TN vs MWPM Comparison Summary")
    print("=" * 60)
    if HAS_PYMATCHING:
        print(f"{'Distance':<10} {'p':<10} {'Tropical LER':<15} {'MWPM LER':<15}")
        print("-" * 60)
        for d in sorted(tropical_results.keys()):
            if tropical_results[d]:
                for p in sorted(tropical_results[d].keys()):
                    tropical_ler = tropical_results[d][p]
                    mwpm_ler = mwpm_results.get(d, {}).get(p, float('nan'))
                    status = "✓" if abs(tropical_ler - mwpm_ler) < 0.01 else "≠"
                    print(f"d={d:<8} {p:<10.4f} {tropical_ler:<15.4f} {mwpm_ler:<15.4f} {status}")
    else:
        print(f"{'Distance':<10} {'p':<10} {'Tropical LER':<15}")
        print("-" * 45)
        for d in sorted(tropical_results.keys()):
            if tropical_results[d]:
                for p in sorted(tropical_results[d].keys()):
                    tropical_ler = tropical_results[d][p]
                    print(f"d={d:<8} {p:<10.4f} {tropical_ler:<15.4f}")


if __name__ == "__main__":
    main()
