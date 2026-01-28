#!/usr/bin/env python3
"""
Benchmark each improvement to the tropical TN decoder.

Tests the impact of:
1. Baseline (no refinement)
2. Coordinate descent refinement
3. Simulated annealing refinement
4. Syndrome projection
5. Combined: Simulated annealing + Syndrome projection

Usage:
    uv run python scripts/benchmark_improvements.py
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
DISTANCES = [3, 5]  # Test both distances
ERROR_RATES = [0.003, 0.005, 0.007]
SAMPLE_SIZE = 100  # More samples for statistical significance
CHI = 32

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


def run_tropical_variant(
    H: np.ndarray,
    syndrome: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    refine: bool,
    refine_method: str,
    syndrome_projection: bool,
    chi: int = CHI,
) -> tuple[np.ndarray, int, float]:
    """Run tropical TN decoder with specified options."""
    n_errors = H.shape[1]
    uai_str = build_decoding_uai(H, priors, syndrome)
    model = read_model_from_string(uai_str)

    start_time = time.perf_counter()
    try:
        assignment, score, info = mpe_tropical_approximate(
            model,
            method="sweep",
            chi=chi,
            refine=refine,
            refine_method=refine_method,
            syndrome_projection=syndrome_projection,
            H=H if syndrome_projection else None,
            syndrome=syndrome if syndrome_projection else None,
            priors=priors if syndrome_projection else None,
        )
    except Exception as e:
        print(f"    Error: {e}")
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
) -> tuple[float, float]:
    """Run BP+OSD decoder batch. Returns (LER, time_per_sample)."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    bp_decoder = BatchBPDecoder(H, priors, device=device)
    osd_decoder = BatchOSDDecoder(H, device=device)

    start_time = time.perf_counter()
    batch_syndromes = torch.from_numpy(syndromes).float().to(device)
    marginals = bp_decoder.decode(batch_syndromes, max_iter=60, damping=0.2)
    marginals_np = marginals.cpu().numpy()
    solutions = osd_decoder.solve_batch(syndromes, marginals_np, osd_order=10)
    total_time = time.perf_counter() - start_time

    # Compute predictions
    errors = 0
    for b in range(len(syndromes)):
        p_flip = 0.0
        for i in np.where(solutions[b] == 1)[0]:
            p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
        pred = int(p_flip > 0.5)
        # Need ground truth from observables
        # For now, just return LER=0 as placeholder
    
    return 0.0, total_time / len(syndromes)


def benchmark_variant(
    name: str,
    H: np.ndarray,
    syndromes: np.ndarray,
    observables: np.ndarray,
    priors: np.ndarray,
    obs_flip: np.ndarray,
    refine: bool,
    refine_method: str,
    syndrome_projection: bool,
    max_samples: int = SAMPLE_SIZE,
) -> dict:
    """Benchmark a specific variant."""
    errors = 0
    total_time = 0.0
    n_samples = min(max_samples, len(syndromes))

    for i in range(n_samples):
        solution, pred, decode_time = run_tropical_variant(
            H,
            syndromes[i],
            priors,
            obs_flip,
            refine=refine,
            refine_method=refine_method,
            syndrome_projection=syndrome_projection,
        )
        if pred != observables[i]:
            errors += 1
        total_time += decode_time

        if (i + 1) % 10 == 0:
            gc.collect()

    ler = errors / n_samples
    avg_time = total_time / n_samples
    return {"name": name, "ler": ler, "time_ms": avg_time * 1000, "samples": n_samples}


def main():
    print("=" * 70)
    print("Tropical TN Decoder Improvement Benchmark")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Error rates: {ERROR_RATES}")
    print(f"  Samples: {SAMPLE_SIZE}")
    print(f"  Chi: {CHI}")

    # Define variants to test (focus on most effective)
    variants = [
        ("Baseline (no refine)", False, "coordinate_descent", False),
        ("Coord Descent", True, "coordinate_descent", False),
        ("Syndrome Projection", False, "coordinate_descent", True),
        ("CD + Projection", True, "coordinate_descent", True),
    ]

    all_results = {}

    for d in DISTANCES:
        all_results[d] = {}
        print(f"\n{'='*70}")
        print(f"Distance d={d}")
        print("=" * 70)

        for p in ERROR_RATES:
            data = load_dataset(d, p)
            if data is None:
                print(f"\n  p={p}: Dataset not found, skipping")
                continue

            H, syndromes, observables, priors, obs_flip, dem = data
            print(f"\n  p={p} (H shape: {H.shape})")
            
            all_results[d][p] = {}

            # BP+OSD baseline
            print("    Running BP+OSD...", end=" ", flush=True)
            device = "cuda" if CUDA_AVAILABLE else "cpu"
            bp_decoder = BatchBPDecoder(H, priors, device=device)
            osd_decoder = BatchOSDDecoder(H, device=device)
            
            n_samples = min(SAMPLE_SIZE, len(syndromes))
            start = time.perf_counter()
            batch_syn = torch.from_numpy(syndromes[:n_samples]).float().to(device)
            marginals = bp_decoder.decode(batch_syn, max_iter=60, damping=0.2)
            solutions = osd_decoder.solve_batch(syndromes[:n_samples], marginals.cpu().numpy(), osd_order=10)
            bposd_time = (time.perf_counter() - start) / n_samples
            
            bposd_errors = 0
            for b in range(n_samples):
                p_flip = 0.0
                for i in np.where(solutions[b] == 1)[0]:
                    p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
                if int(p_flip > 0.5) != observables[b]:
                    bposd_errors += 1
            bposd_ler = bposd_errors / n_samples
            print(f"LER={bposd_ler:.4f}, time={bposd_time*1000:.2f}ms")
            all_results[d][p]["BP+OSD"] = {"ler": bposd_ler, "time_ms": bposd_time * 1000}

            # Test each variant
            for name, refine, refine_method, syn_proj in variants:
                print(f"    Running {name}...", end=" ", flush=True)
                result = benchmark_variant(
                    name, H, syndromes, observables, priors, obs_flip,
                    refine=refine,
                    refine_method=refine_method,
                    syndrome_projection=syn_proj,
                    max_samples=n_samples,
                )
                print(f"LER={result['ler']:.4f}, time={result['time_ms']:.2f}ms")
                all_results[d][p][name] = result

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for d in sorted(all_results.keys()):
        print(f"\nDistance d={d}:")
        print(f"{'Variant':<25} ", end="")
        for p in sorted(all_results[d].keys()):
            print(f"p={p:<8}", end=" ")
        print()
        print("-" * 70)
        
        # Collect all variant names
        variant_names = set()
        for p in all_results[d]:
            variant_names.update(all_results[d][p].keys())
        
        for name in ["BP+OSD"] + [v[0] for v in variants]:
            if name not in variant_names:
                continue
            print(f"{name:<25} ", end="")
            for p in sorted(all_results[d].keys()):
                if name in all_results[d][p]:
                    ler = all_results[d][p][name]["ler"] if isinstance(all_results[d][p][name], dict) else all_results[d][p][name].get("ler", 0)
                    print(f"{ler:<9.4f}", end=" ")
                else:
                    print(f"{'N/A':<9}", end=" ")
            print()

    # Save results
    output_path = Path("outputs/improvement_benchmark.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), results=all_results)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()
