#!/usr/bin/env python3
"""
Quick test to verify the Tropical TN fix matches MWPM.

Usage:
    uv run python scripts/test_tropical_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np
import stim
import pymatching


def build_parity_check_matrix_from_matching(matcher):
    """Build parity check matrix from pymatching's Matching graph."""
    n_detectors = matcher.num_detectors
    edges = matcher.edges()
    n_edges = len(edges)

    H = np.zeros((n_detectors, n_edges), dtype=np.uint8)
    priors = np.zeros(n_edges, dtype=np.float64)
    obs_flip = np.zeros(n_edges, dtype=np.uint8)

    for j, (node1, node2, data) in enumerate(edges):
        weight = data.get('weight', 1.0)
        error_prob = data.get('error_probability', -1.0)
        fault_ids = data.get('fault_ids', set())

        if error_prob < 0 and weight >= 0:
            error_prob = 1.0 / (1.0 + math.exp(weight))
        elif error_prob < 0:
            error_prob = 0.01

        priors[j] = np.clip(error_prob, 1e-10, 1 - 1e-10)

        if node1 is not None and 0 <= node1 < n_detectors:
            H[node1, j] = 1
        if node2 is not None and 0 <= node2 < n_detectors:
            H[node2, j] = 1

        if fault_ids:
            obs_flip[j] = 1

    return H, priors, obs_flip


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
            lines.append("1")
            lines.append("1.0")
            lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Testing Tropical TN Fix - Quick Verification")
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
    
    # Create matcher
    matcher = pymatching.Matching.from_detector_error_model(dem)
    H, priors, obs_flip = build_parity_check_matrix_from_matching(matcher)
    
    print(f"\nTest setup:")
    print(f"  DEM: {dem.num_detectors} detectors, {dem.num_observables} observables")
    print(f"  Matcher: {matcher.num_edges} edges, {matcher.num_fault_ids} fault_ids")
    print(f"  Matrix H: {H.shape}")
    print(f"  obs_flip: {np.sum(obs_flip)} edges flip observable (out of {len(obs_flip)})")
    
    # Sample
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(100, append_observables=True)
    syndromes = samples[:, :-1].astype(np.uint8)
    observables = samples[:, -1].astype(np.int32)
    
    # MWPM decode
    mwpm_preds = matcher.decode_batch(syndromes)
    if mwpm_preds.ndim > 1:
        mwpm_preds = mwpm_preds.flatten()
    
    print(f"\nDecoding {len(syndromes)} samples...")
    
    tropical_correct = 0
    mwpm_correct = 0
    agrees = 0
    
    for i in range(len(syndromes)):
        syndrome = syndromes[i]
        actual = observables[i]
        mwpm_pred = int(mwpm_preds[i])
        
        # Tropical TN
        uai_str = build_uai(H, priors, syndrome)
        model = read_model_from_string(uai_str)
        assignment, score, info = mpe_tropical(model)
        
        solution = np.zeros(H.shape[1], dtype=np.int32)
        for j in range(H.shape[1]):
            solution[j] = assignment.get(j + 1, 0)
        
        tropical_pred = int(np.dot(solution, obs_flip.astype(int)) % 2)
        
        if tropical_pred == actual:
            tropical_correct += 1
        if mwpm_pred == actual:
            mwpm_correct += 1
        if tropical_pred == mwpm_pred:
            agrees += 1
        else:
            print(f"  Sample {i}: Tropical={tropical_pred}, MWPM={mwpm_pred}, Actual={actual}")
    
    print(f"\nResults ({len(syndromes)} samples):")
    print(f"  Tropical correct: {tropical_correct}/{len(syndromes)} ({100*tropical_correct/len(syndromes):.1f}%)")
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


if __name__ == "__main__":
    main()
