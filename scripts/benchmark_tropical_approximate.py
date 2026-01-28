#!/usr/bin/env python3
"""
Benchmark approximate tropical TN decoder for d=5 surface codes.

This script tests whether the approximate MPS/Sweep contraction methods
can handle d=5 cases that were previously infeasible with exact methods,
and compares decoding performance with BP+OSD.

Usage:
    uv run python scripts/benchmark_tropical_approximate.py
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
from tropical_in_new.src import mpe_tropical
from tropical_in_new.src.utils import read_model_from_string

# Configuration
DISTANCES = [3, 5]
ERROR_RATES = [0.003, 0.005, 0.007, 0.010]
SAMPLE_SIZE = 100  # Smaller for benchmarking
CHI_VALUES = [16, 32, 64]  # Bond dimensions to test
METHODS = ["mps", "sweep"]

# Check CUDA
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


def run_tropical_approximate(
    H: np.ndarray,
    syndrome: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    method: str = "mps",
    chi: int = 32,
) -> tuple[np.ndarray, int, float]:
    """
    Run approximate tropical TN decoder.

    Returns:
        Tuple of (solution, predicted_observable, decode_time)
    """
    n_errors = H.shape[1]
    uai_str = build_decoding_uai(H, priors, syndrome)
    model = read_model_from_string(uai_str)

    start_time = time.perf_counter()
    try:
        assignment, score, info = mpe_tropical(model, method=method, chi=chi)
    except Exception as e:
        print(f"    Error with {method} chi={chi}: {e}")
        return np.zeros(n_errors, dtype=np.int32), 0, 0.0
    decode_time = time.perf_counter() - start_time

    solution = np.zeros(n_errors, dtype=np.int32)
    for i in range(n_errors):
        solution[i] = assignment.get(i + 1, 0)

    predicted_obs = compute_observable_prediction(solution, obs_flip)
    return solution, predicted_obs, decode_time


def run_bposd(
    H: np.ndarray,
    syndromes: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    osd_order: int = 10,
    max_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run BP+OSD decoder batch.

    Returns:
        Tuple of (solutions, predictions, total_time)
    """
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    bp_decoder = BatchBPDecoder(H, priors, device=device)
    osd_decoder = BatchOSDDecoder(H, device=device)

    start_time = time.perf_counter()
    batch_syndromes = torch.from_numpy(syndromes).float().to(device)
    marginals = bp_decoder.decode(batch_syndromes, max_iter=max_iter, damping=0.2)
    marginals_np = marginals.cpu().numpy()
    solutions = osd_decoder.solve_batch(syndromes, marginals_np, osd_order=osd_order)
    total_time = time.perf_counter() - start_time

    # Compute predictions
    predictions = np.zeros(len(syndromes), dtype=int)
    for b in range(len(syndromes)):
        p_flip = 0.0
        for i in np.where(solutions[b] == 1)[0]:
            p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
        predictions[b] = int(p_flip > 0.5)

    return solutions, predictions, total_time


def benchmark_single_config(
    H, syndromes, observables, priors, obs_flip, method, chi, max_samples
):
    """Benchmark a single configuration."""
    errors = 0
    total_time = 0.0
    successful = 0

    for i in range(min(max_samples, len(syndromes))):
        try:
            _, pred, decode_time = run_tropical_approximate(
                H, syndromes[i], priors, obs_flip, method=method, chi=chi
            )
            if pred != observables[i]:
                errors += 1
            total_time += decode_time
            successful += 1
        except MemoryError:
            print(f"    MemoryError at sample {i}")
            break
        except Exception as e:
            print(f"    Error at sample {i}: {e}")
            errors += 1
            successful += 1

        if (i + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if successful == 0:
        return float("nan"), 0.0
    return errors / successful, total_time / successful


def main():
    print("=" * 70)
    print("Approximate Tropical TN Decoder Benchmark")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Methods: {METHODS}")
    print(f"  Chi values: {CHI_VALUES}")
    print(f"  Samples per config: {SAMPLE_SIZE}")
    print(f"  CUDA available: {CUDA_AVAILABLE}")

    results = {}

    for d in DISTANCES:
        results[d] = {}
        print(f"\n{'='*70}")
        print(f"Distance d={d}")
        print("=" * 70)

        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"\n  p={p}: Dataset not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip, dem = data
            num_samples = min(SAMPLE_SIZE, len(syndromes))

            print(f"\n  p={p} (H shape: {H.shape}, {num_samples} samples)")
            results[d][p] = {}

            # BP+OSD baseline
            print("    Running BP+OSD...", end=" ", flush=True)
            _, bposd_preds, bposd_time = run_bposd(
                H, syndromes[:num_samples], priors, obs_flip
            )
            bposd_errors = np.sum(bposd_preds != observables[:num_samples])
            bposd_ler = bposd_errors / num_samples
            bposd_time_per_sample = bposd_time / num_samples
            print(f"LER={bposd_ler:.4f}, time={bposd_time_per_sample*1000:.2f}ms/sample")
            results[d][p]["bposd"] = {
                "ler": bposd_ler,
                "time_ms": bposd_time_per_sample * 1000,
            }

            # Tropical approximate methods
            for method in METHODS:
                for chi in CHI_VALUES:
                    print(f"    Running {method} chi={chi}...", end=" ", flush=True)
                    ler, avg_time = benchmark_single_config(
                        H,
                        syndromes[:num_samples],
                        observables[:num_samples],
                        priors,
                        obs_flip,
                        method,
                        chi,
                        num_samples,
                    )

                    if np.isnan(ler):
                        print("FAILED")
                    else:
                        print(f"LER={ler:.4f}, time={avg_time*1000:.2f}ms/sample")
                        results[d][p][f"{method}_chi{chi}"] = {
                            "ler": ler,
                            "time_ms": avg_time * 1000,
                        }

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nLogical Error Rate Comparison:")
    header = f"{'d':<4} {'p':<8} {'BP+OSD':<10}"
    for method in METHODS:
        for chi in CHI_VALUES:
            header += f" {method[:3]}χ{chi:<5}"
    print(header)
    print("-" * len(header))

    for d in sorted(results.keys()):
        for p in sorted(results[d].keys()):
            row = f"{d:<4} {p:<8.4f}"
            if "bposd" in results[d][p]:
                row += f" {results[d][p]['bposd']['ler']:<10.4f}"
            else:
                row += f" {'N/A':<10}"
            for method in METHODS:
                for chi in CHI_VALUES:
                    key = f"{method}_chi{chi}"
                    if key in results[d][p]:
                        row += f" {results[d][p][key]['ler']:<8.4f}"
                    else:
                        row += f" {'N/A':<8}"
            print(row)

    print("\nDecoding Time (ms/sample):")
    header = f"{'d':<4} {'p':<8} {'BP+OSD':<10}"
    for method in METHODS:
        for chi in CHI_VALUES:
            header += f" {method[:3]}χ{chi:<5}"
    print(header)
    print("-" * len(header))

    for d in sorted(results.keys()):
        for p in sorted(results[d].keys()):
            row = f"{d:<4} {p:<8.4f}"
            if "bposd" in results[d][p]:
                row += f" {results[d][p]['bposd']['time_ms']:<10.2f}"
            else:
                row += f" {'N/A':<10}"
            for method in METHODS:
                for chi in CHI_VALUES:
                    key = f"{method}_chi{chi}"
                    if key in results[d][p]:
                        row += f" {results[d][p][key]['time_ms']:<8.1f}"
                    else:
                        row += f" {'N/A':<8}"
            print(row)

    # Save results
    output_path = Path("outputs/tropical_approximate_benchmark.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), results=results)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
