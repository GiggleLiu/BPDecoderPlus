#!/usr/bin/env python3
"""
Manifest threshold behavior for surface code - comparing BP+OSD vs MWPM.

This script demonstrates:
1. Correct threshold behavior (d↑ → LER↓ below threshold)
2. Comparison between BP+OSD and MWPM decoders
3. Both decoders working correctly on surface code
4. Not cirucit level noise, but noise before round and after measure

Usage:
    uv run python scripts/manifest_phenomeno_threshold.py
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import torch
import stim
import pymatching
import matplotlib.pyplot as plt
from typing import Callable

from bpdecoderplus.dem import build_parity_check_matrix
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder

# Configuration
DISTANCES = [3, 5, 7]
ERROR_RATES = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
NUM_SHOTS = 10000
ROUNDS = 5  # Fixed rounds
MAX_ITER = 20
OSD_ORDER = 7

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_circuit(distance: int, error_rate: float) -> stim.Circuit:
    """Generate surface code circuit with phenomenological noise."""
    return stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        distance=distance,
        rounds=ROUNDS,
        before_round_data_depolarization=error_rate,
        before_measure_flip_probability=error_rate,
    )


def decode_mwpm(circuit: stim.Circuit, num_shots: int) -> float:
    """Decode using MWPM (pymatching) and return LER."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(num_shots, append_observables=True)
    syndromes = samples[:, :-1]
    actual_obs = samples[:, -1]
    
    predicted_obs = matcher.decode_batch(syndromes)
    errors = np.sum(predicted_obs.flatten() != actual_obs)
    
    return errors / num_shots


def decode_bp_osd(circuit: stim.Circuit, num_shots: int) -> float:
    """Decode using BP+OSD and return LER."""
    dem = circuit.detector_error_model(decompose_errors=True)
    H, priors, obs_flip = build_parity_check_matrix(dem)
    
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(num_shots, append_observables=True)
    syndromes = samples[:, :-1].astype(np.float32)
    actual_obs = samples[:, -1].astype(np.int32)
    
    # BP decode
    bp_decoder = BatchBPDecoder(H, priors, device=DEVICE)
    osd_decoder = BatchOSDDecoder(H, device=DEVICE)
    
    batch_syndromes = torch.from_numpy(syndromes).float().to(DEVICE)
    marginals = bp_decoder.decode(batch_syndromes, max_iter=MAX_ITER, damping=0.25)
    
    # OSD post-process
    marginals_np = marginals.cpu().numpy()
    solutions = osd_decoder.solve_batch(syndromes.astype(np.uint8), marginals_np, osd_order=OSD_ORDER)
    
    # Compute predictions
    obs_flip_binary = (obs_flip > 0.5).astype(int)
    predicted_obs = (solutions @ obs_flip_binary) % 2
    
    errors = np.sum(predicted_obs != actual_obs)
    
    # Cleanup
    del batch_syndromes, marginals
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    return errors / num_shots


def collect_data(decoder_fn: Callable, name: str) -> dict:
    """Collect threshold data for a decoder."""
    print(f"\n[{name}]")
    results = {d: {} for d in DISTANCES}
    
    total = len(DISTANCES) * len(ERROR_RATES)
    count = 0
    
    for d in DISTANCES:
        for p in ERROR_RATES:
            count += 1
            circuit = generate_circuit(d, p)
            ler = decoder_fn(circuit, NUM_SHOTS)
            results[d][p] = ler
            print(f"  [{count}/{total}] d={d}, p={p:.3f}: LER={ler:.5f}")
    
    return results


def plot_comparison(mwpm_results: dict, bp_osd_results: dict, output_path: str):
    """Plot comparison between MWPM and BP+OSD."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for ax, results, title in [(axes[0], mwpm_results, 'MWPM (pymatching)'),
                                (axes[1], bp_osd_results, 'BP+OSD (BPDecoderPlus)')]:
        for i, d in enumerate(DISTANCES):
            ps = sorted(results[d].keys())
            lers = [results[d][p] for p in ps]
            valid = [(p, l) for p, l in zip(ps, lers) if l > 0]
            if valid:
                ax.plot([v[0]*100 for v in valid], [v[1] for v in valid],
                       f'{markers[i]}-', color=colors[i], label=f'd={d}', 
                       linewidth=2, markersize=8)
        
        ax.set_xlabel('Physical Error Rate (%)', fontsize=12)
        ax.set_ylabel('Logical Error Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)
    
    plt.suptitle(f'Surface Code Threshold Comparison\n(phenomenological noise, r={ROUNDS}, {NUM_SHOTS} shots)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overlay(mwpm_results: dict, bp_osd_results: dict, output_path: str):
    """Plot both decoders overlaid."""
    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, d in enumerate(DISTANCES):
        # MWPM (solid)
        ps = sorted(mwpm_results[d].keys())
        lers = [mwpm_results[d][p] for p in ps]
        valid = [(p, l) for p, l in zip(ps, lers) if l > 0]
        if valid:
            plt.plot([v[0]*100 for v in valid], [v[1] for v in valid],
                    'o-', color=colors[i], label=f'd={d} MWPM', linewidth=2, markersize=8)
        
        # BP+OSD (dashed)
        lers = [bp_osd_results[d][p] for p in ps]
        valid = [(p, l) for p, l in zip(ps, lers) if l > 0]
        if valid:
            plt.plot([v[0]*100 for v in valid], [v[1] for v in valid],
                    's--', color=colors[i], label=f'd={d} BP+OSD', linewidth=2, markersize=6, alpha=0.7)
    
    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title(f'Decoder Comparison: MWPM vs BP+OSD\n(phenomenological noise, r={ROUNDS})', fontsize=14)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.ylim(1e-5, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary(mwpm_results: dict, bp_osd_results: dict):
    """Print summary table."""
    print(f"\n{'='*70}")
    print("Threshold Analysis Summary")
    print(f"{'='*70}")
    print(f"Configuration: r={ROUNDS}, {NUM_SHOTS} shots")
    print(f"Noise model: phenomenological (threshold ~2.9%)")
    
    print(f"\n{'p(%)':<8}", end='')
    for d in DISTANCES:
        print(f"d={d} MWPM   d={d} BP+OSD ", end='')
    print()
    print('-' * 70)
    
    for p in ERROR_RATES:
        print(f"{p*100:<8.1f}", end='')
        for d in DISTANCES:
            mwpm_ler = mwpm_results[d].get(p, 0)
            bp_ler = bp_osd_results[d].get(p, 0)
            print(f"{mwpm_ler:<10.5f}{bp_ler:<10.5f}", end='')
        print()
    
    # Verify threshold behavior
    print(f"\nThreshold Behavior Verification:")
    for name, results in [("MWPM", mwpm_results), ("BP+OSD", bp_osd_results)]:
        all_below = True
        for p in ERROR_RATES:
            lers = [results[d].get(p, 0) for d in DISTANCES]
            if not all(lers[i] >= lers[i+1] for i in range(len(lers)-1)):
                all_below = False
                break
        status = "✓ Correct" if all_below else "✗ Anomaly"
        print(f"  {name}: {status} (d↑ → LER↓ for all tested p)")


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Surface Code Threshold Manifest")
    print(f"Device: {DEVICE}")
    print(f"Distances: {DISTANCES}")
    print(f"Error rates: {[f'{p:.3f}' for p in ERROR_RATES]}")
    print(f"Rounds: {ROUNDS}, Shots: {NUM_SHOTS}")
    
    # Collect MWPM data (fast)
    mwpm_results = collect_data(decode_mwpm, "MWPM")
    
    # Collect BP+OSD data (slower)
    bp_osd_results = collect_data(decode_bp_osd, "BP+OSD")
    
    # Generate plots
    plot_comparison(mwpm_results, bp_osd_results, str(output_dir / "threshold_comparison_decoders.png"))
    print(f"\nComparison plot saved: outputs/threshold_comparison_decoders.png")
    
    plot_overlay(mwpm_results, bp_osd_results, str(output_dir / "threshold_overlay_decoders.png"))
    print(f"Overlay plot saved: outputs/threshold_overlay_decoders.png")
    
    # Summary
    print_summary(mwpm_results, bp_osd_results)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
