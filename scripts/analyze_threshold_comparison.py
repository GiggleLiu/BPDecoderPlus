#!/usr/bin/env python3
"""
Threshold comparison between Tropical TN (with syndrome projection) and BP+OSD.

This script compares the thresholds of:
1. BP+OSD decoder (reference)
2. Tropical TN with syndrome projection (improved approximate method)

Usage:
    uv run python scripts/analyze_threshold_comparison.py
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from bpdecoderplus.dem import load_dem, build_parity_check_matrix, build_decoding_uai
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder
from tropical_in_new.src import mpe_tropical_approximate
from tropical_in_new.src.utils import read_model_from_string

# Configuration
DISTANCES = [3, 5]
ERROR_RATES = [0.001, 0.003, 0.005, 0.007, 0.009, 0.012]
SAMPLE_SIZE = 200  # Per error rate
CHI = 32

CUDA_AVAILABLE = torch.cuda.is_available()


def load_dataset(distance: int, error_rate: float):
    """Load dataset for given distance and error rate."""
    rounds = distance
    p_str = f"{error_rate:.4f}"[2:]
    base_name = f"sc_d{distance}_r{rounds}_p{p_str}_z"

    dem_path = Path(f"datasets/{base_name}.dem")
    npz_path = Path(f"datasets/{base_name}.npz")

    if not dem_path.exists() or not npz_path.exists():
        return None

    dem = load_dem(str(dem_path))
    syndromes, observables, _ = load_syndrome_database(str(npz_path))
    H, priors, obs_flip = build_parity_check_matrix(
        dem, split_by_separator=True, merge_hyperedges=True
    )

    return H, syndromes, observables, priors, obs_flip, dem


def compute_observable_prediction(solution: np.ndarray, obs_flip: np.ndarray) -> int:
    """Compute observable prediction using mod-2 arithmetic."""
    obs_flip_binary = (obs_flip > 0.5).astype(int)
    return int(np.dot(solution, obs_flip_binary) % 2)


def run_bposd_batch(H, syndromes, observables, priors, obs_flip):
    """Run BP+OSD decoder and return LER."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    bp_decoder = BatchBPDecoder(H, priors, device=device)
    osd_decoder = BatchOSDDecoder(H, device=device)

    batch_syndromes = torch.from_numpy(syndromes).float().to(device)
    marginals = bp_decoder.decode(batch_syndromes, max_iter=60, damping=0.2)
    solutions = osd_decoder.solve_batch(syndromes, marginals.cpu().numpy(), osd_order=10)

    errors = 0
    for b in range(len(syndromes)):
        p_flip = 0.0
        for i in np.where(solutions[b] == 1)[0]:
            p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
        if int(p_flip > 0.5) != observables[b]:
            errors += 1

    return errors / len(syndromes)


def run_tropical_projection_batch(H, syndromes, observables, priors, obs_flip, chi=CHI):
    """Run Tropical TN with syndrome projection and return LER."""
    n_errors = H.shape[1]
    errors = 0

    for i, syndrome in enumerate(syndromes):
        uai_str = build_decoding_uai(H, priors, syndrome)
        model = read_model_from_string(uai_str)

        try:
            assignment, score, info = mpe_tropical_approximate(
                model,
                method="sweep",
                chi=chi,
                refine=False,
                syndrome_projection=True,
                H=H,
                syndrome=syndrome,
                priors=priors,
            )
        except Exception as e:
            errors += 1
            continue

        solution = np.zeros(n_errors, dtype=np.int32)
        for j in range(n_errors):
            solution[j] = assignment.get(j + 1, 0)

        pred = compute_observable_prediction(solution, obs_flip)
        if pred != observables[i]:
            errors += 1

        if (i + 1) % 20 == 0:
            gc.collect()

    return errors / len(syndromes)


def main():
    print("=" * 70)
    print("Threshold Comparison: BP+OSD vs Tropical TN + Syndrome Projection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Samples: {SAMPLE_SIZE}")
    print(f"  Chi: {CHI}")
    print(f"  CUDA: {CUDA_AVAILABLE}")

    results = {"bposd": {}, "tropical_proj": {}}

    for d in DISTANCES:
        results["bposd"][d] = {}
        results["tropical_proj"][d] = {}
        print(f"\n{'='*70}")
        print(f"Distance d={d}")
        print("=" * 70)

        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"\n  p={p}: Dataset not found")
                continue

            H, syndromes, observables, priors, obs_flip, dem = data
            n_samples = min(SAMPLE_SIZE, len(syndromes))
            print(f"\n  p={p} (H: {H.shape}, {n_samples} samples)")

            # BP+OSD
            print("    BP+OSD...", end=" ", flush=True)
            start = time.perf_counter()
            bposd_ler = run_bposd_batch(
                H, syndromes[:n_samples], observables[:n_samples], priors, obs_flip
            )
            bposd_time = (time.perf_counter() - start) * 1000 / n_samples
            print(f"LER={bposd_ler:.4f} ({bposd_time:.1f}ms/sample)")
            results["bposd"][d][p] = bposd_ler

            # Tropical with projection
            print("    Tropical+Proj...", end=" ", flush=True)
            start = time.perf_counter()
            tropical_ler = run_tropical_projection_batch(
                H, syndromes[:n_samples], observables[:n_samples], priors, obs_flip
            )
            tropical_time = (time.perf_counter() - start) * 1000 / n_samples
            print(f"LER={tropical_ler:.4f} ({tropical_time:.1f}ms/sample)")
            results["tropical_proj"][d][p] = tropical_ler

            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 70)
    print("THRESHOLD SUMMARY")
    print("=" * 70)

    for d in sorted(results["bposd"].keys()):
        print(f"\nDistance d={d}:")
        print(f"{'p':<10} {'BP+OSD':<12} {'Tropical+Proj':<15} {'Ratio':<10}")
        print("-" * 50)
        for p in sorted(results["bposd"][d].keys()):
            bposd = results["bposd"][d].get(p, float("nan"))
            tropical = results["tropical_proj"][d].get(p, float("nan"))
            ratio = tropical / bposd if bposd > 0 else float("inf")
            print(f"{p:<10.4f} {bposd:<12.4f} {tropical:<15.4f} {ratio:<10.2f}x")

    # Check for threshold behavior (LER decreases with increasing d at fixed p)
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    for method, name in [("bposd", "BP+OSD"), ("tropical_proj", "Tropical+Proj")]:
        print(f"\n{name}:")
        for p in sorted(ERROR_RATES):
            lers = []
            ds = []
            for d in sorted(results[method].keys()):
                if p in results[method][d]:
                    lers.append(results[method][d][p])
                    ds.append(d)
            if len(lers) >= 2:
                # Check if LER decreases with d (below threshold)
                decreasing = all(lers[i] >= lers[i + 1] for i in range(len(lers) - 1))
                trend = "below threshold (LER decreases with d)" if decreasing else "above threshold (LER increases with d)"
                print(f"  p={p:.4f}: {trend}")
                for i, d in enumerate(ds):
                    print(f"    d={d}: LER={lers[i]:.4f}")

    # Save results
    output_path = Path("outputs/threshold_comparison.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), results=results)
    print(f"\nResults saved to: {output_path}")

    # Generate plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        markers = ["o", "s", "^", "D"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, (method, title) in enumerate([("bposd", "BP+OSD"), ("tropical_proj", "Tropical TN + Projection")]):
            ax = axes[idx]
            for i, d in enumerate(sorted(results[method].keys())):
                if not results[method][d]:
                    continue
                ps = sorted(results[method][d].keys())
                lers = [results[method][d][p] for p in ps]
                ax.plot(ps, lers, f"{markers[i]}-", color=colors[i], label=f"d={d}", linewidth=2, markersize=8)

            ax.set_xlabel("Physical Error Rate (p)", fontsize=12)
            ax.set_ylabel("Logical Error Rate", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.axvline(x=0.007, color="gray", linestyle="--", alpha=0.5)

        plt.suptitle("Threshold Comparison: BP+OSD vs Tropical TN + Projection", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig("outputs/threshold_comparison_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: outputs/threshold_comparison_plot.png")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


if __name__ == "__main__":
    results = main()
