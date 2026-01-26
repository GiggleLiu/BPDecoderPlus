#!/usr/bin/env python3
"""
Quick test to verify the Tropical TN fix matches MWPM.

Uses bpdecoderplus.dem functions for parity check matrix construction.

Usage:
    uv run python scripts/test_tropical_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import stim

from bpdecoderplus.dem import build_parity_check_matrix

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False


def build_uai(H, priors, syndrome):
    """Build UAI model."""
    n_detectors, n_errors = H.shape
    lines = []
    lines.append("MARKOV")
    lines.append(str(n_errors))
    lines.append(" ".join(["2"] * n_errors))

    n_factors = n_errors + n_detectors
    lines.append(str(n_factors))

    for i in range(n_errors):
        lines.append(f"1 {i}")

    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        if len(error_indices) > 0:
            lines.append(f"{len(error_indices)} " + " ".join(str(e) for e in error_indices))
        else:
            lines.append("0")

    lines.append("")

    for i in range(n_errors):
        p = priors[i]
        lines.append("2")
        lines.append(str(1.0 - p))
        lines.append(str(p))
        lines.append("")

    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        if len(error_indices) > 0:
            syndrome_bit = int(syndrome[d])
            n_entries = 2**len(error_indices)
            lines.append(str(n_entries))
            for i in range(n_entries):
                parity = bin(i).count("1") % 2
                if parity == syndrome_bit:
                    lines.append("1.0")
                else:
                    lines.append("1e-30")
            lines.append("")
        else:
            # Empty detector: probability depends on whether the syndrome is consistent.
            # If syndrome[d] == 0, the constraint is satisfied (probability 1.0).
            # If syndrome[d] != 0, the constraint is unsatisfiable (near-zero probability).
            syndrome_bit = int(syndrome[d])
            lines.append("1")
            if syndrome_bit == 0:
                lines.append("1.0")
            else:
                lines.append("1e-30")
            lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Testing Tropical TN Fix - Quick Verification")
    print("(Using bpdecoderplus.dem for parity check matrix)")
    print("=" * 60)
    
    from tropical_in_new.src import mpe_tropical
    from tropical_in_new.src.utils import read_model_from_string
    
    # Generate test circuit
    distance = 3
    error_rate = 0.01
    
    circuit = stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=error_rate,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    
    # Build parity check matrix using bpdecoderplus.dem
    # Use merge_hyperedges=True for faster computation (smaller matrix)
    # obs_flip will be thresholded at 0.5 for observable prediction
    H, priors, obs_flip = build_parity_check_matrix(
        dem,
        split_by_separator=True,
        merge_hyperedges=True,  # Faster with smaller matrix
    )
    
    print(f"\nTest setup:")
    print(f"  DEM: {dem.num_detectors} detectors, {dem.num_observables} observables")
    print(f"  Matrix H: {H.shape}")
    print(f"  obs_flip: {np.sum(obs_flip)} errors flip observable (out of {len(obs_flip)})")
    print(f"  obs_flip unique values: {np.unique(obs_flip)}")
    
    # Sample
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(100, append_observables=True)
    syndromes = samples[:, :-1].astype(np.uint8)
    observables = samples[:, -1].astype(np.int32)
    
    # MWPM decode (if available)
    mwpm_preds = None
    if HAS_PYMATCHING:
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_preds = matcher.decode_batch(syndromes)
        if mwpm_preds.ndim > 1:
            mwpm_preds = mwpm_preds.flatten()
        print(f"  MWPM available: Yes")
    else:
        print(f"  MWPM available: No (pymatching not installed)")
    
    print(f"\nDecoding {len(syndromes)} samples...")
    
    tropical_correct = 0
    mwpm_correct = 0
    agrees = 0
    
    for i in range(len(syndromes)):
        syndrome = syndromes[i]
        actual = observables[i]
        
        # Tropical TN
        uai_str = build_uai(H, priors, syndrome)
        model = read_model_from_string(uai_str)
        assignment, score, info = mpe_tropical(model)
        
        solution = np.zeros(H.shape[1], dtype=np.int32)
        for j in range(H.shape[1]):
            solution[j] = assignment.get(j + 1, 0)
        
        # Threshold obs_flip at 0.5 for soft values from hyperedge merging
        obs_flip_binary = (obs_flip > 0.5).astype(int)
        tropical_pred = int(np.dot(solution, obs_flip_binary) % 2)
        
        if tropical_pred == actual:
            tropical_correct += 1
        
        if mwpm_preds is not None:
            mwpm_pred = int(mwpm_preds[i])
            if mwpm_pred == actual:
                mwpm_correct += 1
            if tropical_pred == mwpm_pred:
                agrees += 1
            elif i < 10:  # Only print first 10 disagreements
                print(f"  Sample {i}: Tropical={tropical_pred}, MWPM={mwpm_pred}, Actual={actual}")
    
    print(f"\nResults ({len(syndromes)} samples):")
    print(f"  Tropical correct: {tropical_correct}/{len(syndromes)} ({100*tropical_correct/len(syndromes):.1f}%)")
    
    if mwpm_preds is not None:
        print(f"  MWPM correct: {mwpm_correct}/{len(syndromes)} ({100*mwpm_correct/len(syndromes):.1f}%)")
        print(f"  Tropical agrees with MWPM: {agrees}/{len(syndromes)} ({100*agrees/len(syndromes):.1f}%)")
        
        agreement_rate = 100*agrees/len(syndromes)
        if agreement_rate >= 95:
            print(f"\n✓ SUCCESS: Tropical TN matches MWPM on {agreement_rate:.1f}% of samples!")
            if agrees < len(syndromes):
                print("  (Disagreements may be due to degeneracy - multiple optimal solutions)")
        else:
            print(f"\n✗ WARNING: Tropical TN differs from MWPM on {len(syndromes)-agrees} samples ({100-agreement_rate:.1f}%)")
            print("  This suggests a bug in the decoder")
    else:
        if tropical_correct >= len(syndromes) * 0.95:
            print(f"\n✓ SUCCESS: Tropical TN achieves {100*tropical_correct/len(syndromes):.1f}% accuracy")
        else:
            print(f"\n✗ WARNING: Tropical TN accuracy is low")


if __name__ == "__main__":
    main()
