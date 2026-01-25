#!/usr/bin/env python3
"""
Threshold analysis for BP+OSD decoder.

This module generates threshold plots across different code distances
and error rates, including comparison with the ldpc library.

Usage:
    uv run python scripts/analyze_threshold.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch

from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()


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


def compute_observable_predictions_batch(solutions: np.ndarray, obs_flip: np.ndarray) -> np.ndarray:
    """
    Compute observable predictions for a batch of solutions using soft XOR.

    Vectorized version of soft XOR probability computation.

    Args:
        solutions: Batch of binary error patterns, shape (batch, n_errors)
        obs_flip: Observable flip probabilities (0.0 to 1.0)

    Returns:
        Predicted observable values, shape (batch,)
    """
    batch_size = solutions.shape[0]
    predictions = np.zeros(batch_size, dtype=int)
    for b in range(batch_size):
        p_flip = 0.0
        # Only iterate over active hyperedges (where solution[b,i] == 1)
        for i in np.where(solutions[b] == 1)[0]:
            p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
        predictions[b] = int(p_flip > 0.5)
    return predictions


# Check if ldpc is available
try:
    from ldpc import BpOsdDecoder
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False

# Matrix construction mode:
# - "merged": split_by_separator=True, merge_hyperedges=True (default, smaller matrix)
# - "split": split_by_separator=True, merge_hyperedges=False (binary obs_flip)
# - "raw": split_by_separator=False, merge_hyperedges=False (direct from DEM)
MATRIX_MODE = "merged"

# Configuration
# Circuit-level depolarizing noise threshold for rotated surface code is ~0.7%.
# We scan around this threshold to observe the crossing behavior.
DISTANCES = [3, 5, 7, 9]
ERROR_RATES = [0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.015]  # Scanning around ~0.7% threshold
ITER = 60  # Increased for complex circuit-level factor graphs
SAMPLE_SIZE = 5000

def run_bpdecoderplus_gpu_batch(H, syndromes, observables, obs_flip, priors,
                                 osd_order=10, max_iter=ITER, chunk_size=50000):
    """
    Run BPDecoderPlus with GPU batch processing for faster threshold analysis.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        obs_flip: Observable flip probabilities per hyperedge (soft values 0.0-1.0
            when using hyperedge merging, or binary 0/1 without merging)
        priors: Per-qubit error probabilities
        osd_order: OSD search depth
        max_iter: Maximum BP iterations
        chunk_size: Process in chunks to avoid GPU OOM

    Returns:
        Logical error rate
    """
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    bp_decoder = BatchBPDecoder(H, priors, device=device)
    osd_decoder = BatchOSDDecoder(H, device=device)

    total_errors = 0
    n_samples = len(syndromes)

    # Process in chunks to avoid GPU OOM
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_syndromes = syndromes[start:end]
        chunk_observables = observables[start:end]

        batch_syndromes = torch.from_numpy(chunk_syndromes).float().to(device)
        marginals = bp_decoder.decode(batch_syndromes, max_iter=max_iter, damping=0.2)

        # Use batch solve
        marginals_np = marginals.cpu().numpy()
        solutions = osd_decoder.solve_batch(chunk_syndromes, marginals_np, osd_order=osd_order)

        # Compute predictions using soft XOR (handles fractional obs_flip from hyperedge merging)
        predictions = compute_observable_predictions_batch(solutions, obs_flip)

        total_errors += np.sum(predictions != chunk_observables)

        # Free GPU memory
        del batch_syndromes, marginals
        torch.cuda.empty_cache()

    return total_errors / n_samples


def run_ldpc_decoder(H, syndromes, observables, obs_flip, error_rate=0.01,
                     osd_order=10, max_iter=ITER):
    """
    Run ldpc library BP+OSD decoder.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        obs_flip: Observable flip indicators per error
        error_rate: Physical error rate for BP
        osd_order: OSD search depth
        max_iter: Maximum BP iterations

    Returns:
        Logical error rate
    """
    if not LDPC_AVAILABLE:
        raise ImportError("ldpc library not installed")

    ldpc_decoder = BpOsdDecoder(
        H.astype(np.uint8),
        error_rate=error_rate,
        max_iter=max_iter,
        bp_method='product_sum',
        osd_method='osd_e',
        osd_order=osd_order
    )

    errors = 0
    for i, syndrome in enumerate(syndromes):
        result = ldpc_decoder.decode(syndrome.astype(np.uint8))
        predicted_obs = compute_observable_prediction(result, obs_flip)
        if predicted_obs != observables[i]:
            errors += 1

    return errors / len(syndromes)


def load_dataset(distance: int, error_rate: float, matrix_mode: str = MATRIX_MODE):
    """
    Load dataset for given distance and error rate.

    Args:
        distance: Code distance
        error_rate: Physical error rate
        matrix_mode: Matrix construction mode ("merged", "split", or "raw")

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
    
    if matrix_mode == "merged":
        H, priors, obs_flip = build_parity_check_matrix(dem, split_by_separator=True, merge_hyperedges=True)
    elif matrix_mode == "split":
        H, priors, obs_flip = build_parity_check_matrix(dem, split_by_separator=True, merge_hyperedges=False)
    elif matrix_mode == "raw":
        H, priors, obs_flip = build_parity_check_matrix(dem, split_by_separator=False, merge_hyperedges=False)
    else:
        raise ValueError(f"Unknown matrix_mode: {matrix_mode}")

    return H, syndromes, observables, priors, obs_flip


def collect_threshold_data(osd_order: int = 10, max_samples: int = SAMPLE_SIZE):
    """
    Collect logical error rates for threshold analysis using GPU batch processing.

    Args:
        osd_order: OSD search depth
        max_samples: Maximum samples per dataset
        use_gpu: Whether to use GPU batch processing (faster)

    Returns:
        Dict mapping distance -> {error_rate: ler}
    """
    results = {}

    for d in DISTANCES:
        results[d] = {}
        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"  Dataset d={d}, p={p} not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip = data
            num_samples = min(max_samples, len(syndromes))

            ler = run_bpdecoderplus_gpu_batch(
                H, syndromes[:num_samples],
                observables[:num_samples], obs_flip, priors,
                osd_order=osd_order
            )

            results[d][p] = ler
            print(f"  d={d}, p={p}: LER={ler:.4f} ({num_samples} samples)")

    return results


def collect_threshold_data_ldpc(osd_order: int = 10, max_samples: int = SAMPLE_SIZE):
    """
    Collect logical error rates for threshold analysis using ldpc library.

    Args:
        osd_order: OSD search depth
        max_samples: Maximum samples per dataset

    Returns:
        Dict mapping distance -> {error_rate: ler}
    """
    if not LDPC_AVAILABLE:
        raise ImportError("ldpc library not installed")

    results = {}

    for d in DISTANCES:
        results[d] = {}
        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"  [ldpc] Dataset d={d}, p={p} not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip = data
            num_samples = min(max_samples, len(syndromes))

            ler = run_ldpc_decoder(
                H, syndromes[:num_samples],
                observables[:num_samples], obs_flip,
                error_rate=p, osd_order=osd_order
            )
            results[d][p] = ler
            print(f"  [ldpc] d={d}, p={p}: LER={ler:.4f} ({num_samples} samples)")

    return results


def plot_threshold_comparison(bp_results: dict, ldpc_results: dict,
                               output_path: str = "outputs/threshold_comparison.png"):
    """
    Plot threshold comparison between BPDecoderPlus and ldpc library.

    Args:
        bp_results: Dict mapping distance -> {error_rate: ler} for BPDecoderPlus
        ldpc_results: Dict mapping distance -> {error_rate: ler} for ldpc
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot BPDecoderPlus results
    ax1 = axes[0]
    for i, d in enumerate(sorted(bp_results.keys())):
        if not bp_results[d]:
            continue
        error_rates = sorted(bp_results[d].keys())
        lers = [bp_results[d][p] for p in error_rates]
        ax1.plot(error_rates, lers, f'{markers[i % len(markers)]}-',
                 color=colors[i % len(colors)], label=f'd={d}', linewidth=2, markersize=8)

    ax1.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate', fontsize=12)
    ax1.set_title('BPDecoderPlus', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.axvline(x=0.007, color='gray', linestyle='--', alpha=0.5, label='p=0.7%')

    # Plot ldpc results
    ax2 = axes[1]
    for i, d in enumerate(sorted(ldpc_results.keys())):
        if not ldpc_results[d]:
            continue
        error_rates = sorted(ldpc_results[d].keys())
        lers = [ldpc_results[d][p] for p in error_rates]
        ax2.plot(error_rates, lers, f'{markers[i % len(markers)]}-',
                 color=colors[i % len(colors)], label=f'd={d}', linewidth=2, markersize=8)

    ax2.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate', fontsize=12)
    ax2.set_title('ldpc Library', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.axvline(x=0.007, color='gray', linestyle='--', alpha=0.5, label='p=0.7%')

    plt.suptitle('BP+OSD Decoder Threshold Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Threshold comparison plot saved to: {output_path}")


def plot_threshold_overlay(bp_results: dict, ldpc_results: dict,
                            output_path: str = "outputs/threshold_overlay.png"):
    """
    Plot threshold curves with both implementations overlaid on the same graph.

    Args:
        bp_results: Dict mapping distance -> {error_rate: ler} for BPDecoderPlus
        ldpc_results: Dict mapping distance -> {error_rate: ler} for ldpc
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, d in enumerate(sorted(bp_results.keys())):
        color = colors[i % len(colors)]

        # BPDecoderPlus (solid line)
        if bp_results[d]:
            error_rates = sorted(bp_results[d].keys())
            lers = [bp_results[d][p] for p in error_rates]
            plt.plot(error_rates, lers, 'o-', color=color,
                     label=f'd={d} (BPDecoderPlus)', linewidth=2, markersize=8)

        # ldpc (dashed line)
        if ldpc_results[d]:
            error_rates = sorted(ldpc_results[d].keys())
            lers = [ldpc_results[d][p] for p in error_rates]
            plt.plot(error_rates, lers, 's--', color=color,
                     label=f'd={d} (ldpc)', linewidth=2, markersize=8, alpha=0.7)

    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('BP+OSD Decoder Threshold: BPDecoderPlus vs ldpc', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(x=0.007, color='gray', linestyle='--', alpha=0.5, label='p=0.7%')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Threshold overlay plot saved to: {output_path}")


def plot_threshold_curve(results: dict, output_path: str = "outputs/threshold_plot.png"):
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

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, d in enumerate(sorted(results.keys())):
        if not results[d]:
            continue
        error_rates = sorted(results[d].keys())
        lers = [results[d][p] for p in error_rates]
        plt.plot(error_rates, lers, f'{markers[i % len(markers)]}-',
                 color=colors[i % len(colors)], label=f'd={d}', linewidth=2, markersize=8)

    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('BP+OSD Decoder Threshold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')

    # Add threshold region annotation
    plt.axvline(x=0.007, color='gray', linestyle='--', alpha=0.5, label='p=0.7%')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Threshold plot saved to: {output_path}")


def main():
    """
    Generate threshold plots for d=3,5,7 using GPU batch processing.

    This function collects BPDecoderPlus data and generates the main threshold plot.
    If ldpc library is available, it also collects ldpc data and generates
    comparison plots (threshold_comparison.png, threshold_overlay.png, threshold_plot_ldpc.png).
    """
    print(f"\nMatrix construction mode: {MATRIX_MODE}")
    print("Collecting threshold data (GPU batch mode)...")

    # Collect BPDecoderPlus results
    print("\n[BPDecoderPlus]")
    bp_results = collect_threshold_data(osd_order=10, max_samples=SAMPLE_SIZE)

    # Check we have at least some data
    bp_points = sum(len(v) for v in bp_results.values())
    if bp_points == 0:
        print("Error: No threshold data collected - generate datasets first")
        return

    print(f"\nCollected {bp_points} BPDecoderPlus data points")

    # Generate BPDecoderPlus threshold plot
    plot_threshold_curve(bp_results, "outputs/threshold_plot.png")
    if not Path("outputs/threshold_plot.png").exists():
        print("Warning: Threshold plot was not created")

    # Print BPDecoderPlus summary
    print("\nBPDecoderPlus Threshold Analysis Summary:")
    print(f"{'Distance':<10} {'Error Rates Tested':<30} {'Min LER':<15} {'Max LER':<15}")
    print("-" * 70)
    for d in sorted(bp_results.keys()):
        if bp_results[d]:
            error_rates = sorted(bp_results[d].keys())
            lers = [bp_results[d][p] for p in error_rates]
            print(f"d={d:<8} {len(error_rates)} points ({min(error_rates):.3f}-{max(error_rates):.3f})"
                  f"  {min(lers):.4f}         {max(lers):.4f}")

    # If ldpc is available, also collect ldpc data and generate comparison plots
    if LDPC_AVAILABLE:
        print("\n[ldpc]")
        ldpc_results = collect_threshold_data_ldpc(osd_order=10, max_samples=SAMPLE_SIZE)
        ldpc_points = sum(len(v) for v in ldpc_results.values())

        if ldpc_points > 0:
            print(f"\nCollected {ldpc_points} ldpc data points")

            # Generate ldpc-only threshold plot
            plot_threshold_curve(ldpc_results, "outputs/threshold_plot_ldpc.png")
            if not Path("outputs/threshold_plot_ldpc.png").exists():
                print("Warning: ldpc threshold plot was not created")

            # Generate comparison plots
            plot_threshold_comparison(bp_results, ldpc_results, "outputs/threshold_comparison.png")
            plot_threshold_overlay(bp_results, ldpc_results, "outputs/threshold_overlay.png")
            if not Path("outputs/threshold_comparison.png").exists():
                print("Warning: Comparison plot was not created")
            if not Path("outputs/threshold_overlay.png").exists():
                print("Warning: Overlay plot was not created")

            # Print ldpc summary
            print("\nldpc Threshold Analysis Summary:")
            print(f"{'Distance':<10} {'Error Rates Tested':<30} {'Min LER':<15} {'Max LER':<15}")
            print("-" * 70)
            for d in sorted(ldpc_results.keys()):
                if ldpc_results[d]:
                    error_rates = sorted(ldpc_results[d].keys())
                    lers = [ldpc_results[d][p] for p in error_rates]
                    print(f"d={d:<8} {len(error_rates)} points ({min(error_rates):.3f}-{max(error_rates):.3f})"
                          f"  {min(lers):.4f}         {max(lers):.4f}")

            # Print comparison summary
            print("\nThreshold Comparison Summary:")
            print(f"{'Distance':<10} {'Error Rate':<12} {'BPDecoderPlus':<15} {'ldpc':<15} {'Diff':<10}")
            print("-" * 62)
            for d in sorted(bp_results.keys()):
                for p in sorted(bp_results[d].keys()):
                    bp_ler = bp_results[d].get(p)
                    ldpc_ler = ldpc_results[d].get(p)
                    if bp_ler is not None and ldpc_ler is not None:
                        diff = bp_ler - ldpc_ler
                        print(f"d={d:<8} p={p:<10.4f} {bp_ler:<15.4f} {ldpc_ler:<15.4f} {diff:+.4f}")
        else:
            print("\nNo ldpc data collected, skipping comparison plots")
    else:
        print("\nldpc library not available, skipping comparison plots")


if __name__ == "__main__":
    main()
