#!/usr/bin/env python3
"""Run the Complete Decoding Example from getting_started.md"""

import numpy as np
from bpdecoderplus.pytorch_bp import (
    read_model_file, BeliefPropagation,
    belief_propagate, compute_marginals, apply_evidence
)
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.osd import OSDDecoder
# Load model and data
model = read_model_file('datasets/sc_d3_r3_p0010_z.uai')
dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')

print(f"Loaded {len(syndromes)} syndromes")
print(f"Syndrome shape: {syndromes.shape}")
print(f"Observable shape: {observables.shape}")

# Build BP decoder
bp = BeliefPropagation(model)
H, priors, obs_flip = build_parity_check_matrix(dem)

print(f"H matrix shape: {H.shape}")
print(f"Number of error mechanisms: {len(obs_flip)}")

#OSD feature
osd = OSDDecoder(H)


# Evaluate on test set
print(f"\nEvaluating on 1000 test samples...")
predictions = []
for idx, syndrome in enumerate(syndromes[:1000]):
    if idx % 100 == 0:
        print(f"  Processed {idx}/1000 samples...")

    evidence = {i+1: int(syndrome[i]) for i in range(len(syndrome))}
    bp_ev = apply_evidence(bp, evidence)
    state, _ = belief_propagate(bp_ev, max_iter=20, tol=1e-6, damping=0.2)
    # Get the soft information from the BP state
    marginals = compute_marginals(state, bp_ev)
    # Get error variable indices (skip detector variables)
    num_detectors = len(syndromes[0])
    error_var_start = num_detectors + 1

    estimated_errors = np.zeros(len(obs_flip), dtype=int)
    for i, var_idx in enumerate(range(error_var_start, error_var_start + len(obs_flip))):
        if var_idx in marginals:
            p0, p1 = marginals[var_idx][0].item(), marginals[var_idx][1].item()
            estimated_errors[i] = 1 if p1 > p0 else 0
    # estimated_errors = osd.solve(syndrome, marginals, error_var_start,osd_order=10)
    predictions.append(np.dot(estimated_errors, obs_flip) % 2)

logical_error_rate = np.mean(np.array(predictions) != observables[:1000])
print(f"\nLogical error rate: {logical_error_rate:.4f}")
print(f"Decoder accuracy: {(1 - logical_error_rate):.4f}")
