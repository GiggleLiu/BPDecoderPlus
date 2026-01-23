#!/usr/bin/env python3
"""
BP + OSD Decoder Demonstration

This script demonstrates the corrected BP+OSD decoder with soft-weighted cost function.
See docs/bp_osd_fix.md for detailed explanation of the fix.

Compares:
- BP-only: Hard decision on BP marginals
- BP+OSD-0: Gaussian elimination only (no search)  
- BP+OSD-15: Exhaustive search over 15 least reliable free variables

Expected results on d=3 surface code (p=0.001):
- BP-only: ~11% logical error rate
- BP+OSD-15: ~7% logical error rate (37% improvement)
"""

import numpy as np
import torch
import time
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.osd import OSDDecoder

# Configuration
OSD_ORDER = 15  # Search 2^15 = 32768 candidates (good balance of speed/accuracy)
BP_MAX_ITER = 50  # BP iterations (increased for better convergence at low error rates)
BATCH_SIZE = 50
NUM_SAMPLES = 1000  # 1000 samples for reasonable runtime and statistical significance

# Load data
print("Loading data...")
dem = load_dem('datasets/sc_d3_r3_p0001_z.dem')
syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0001_z.npz')
H, priors, obs_flip = build_parity_check_matrix(dem)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Parity check matrix H: {H.shape[0]} checks x {H.shape[1]} error mechanisms")
print(f"Test samples: {len(syndromes)}")

# Initialize decoders
bp_decoder = BatchBPDecoder(H, priors, device=device)
osd_decoder = OSDDecoder(H)

# Decode
predictions_bp = []
predictions_osd = []

print(f"\nRunning BP (iter={BP_MAX_ITER}) + OSD-{OSD_ORDER} on {NUM_SAMPLES} samples...")
print(f"OSD search space: 2^{OSD_ORDER} = {2**OSD_ORDER:,} candidates per syndrome\n")

start_time = time.time()

for start_idx in range(0, NUM_SAMPLES, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, NUM_SAMPLES)
    batch_syndromes = torch.from_numpy(syndromes[start_idx:end_idx]).to(dtype=torch.float32)

    # Run batch BP
    marginals_batch = bp_decoder.decode(batch_syndromes, max_iter=BP_MAX_ITER, damping=0.2)

    # Decode each sample
    for i, marginals in enumerate(marginals_batch):
        syndrome = syndromes[start_idx + i]
        error_probs = marginals.cpu().numpy()
        
        # BP-only: threshold at 0.5
        estimated_bp = (marginals > 0.5).cpu().numpy().astype(int)
        predictions_bp.append(np.dot(estimated_bp, obs_flip) % 2)
        
        # BP+OSD: use soft-weighted cost function
        estimated_osd = osd_decoder.solve(syndrome, error_probs, osd_order=OSD_ORDER)
        predictions_osd.append(np.dot(estimated_osd, obs_flip) % 2)

    elapsed = time.time() - start_time
    print(f"  {end_idx}/{NUM_SAMPLES} ({elapsed:.0f}s)")

# Results
ler_bp = np.mean(np.array(predictions_bp) != observables[:NUM_SAMPLES])
ler_osd = np.mean(np.array(predictions_osd) != observables[:NUM_SAMPLES])

print("\n" + "="*60)
print(f"Results (n={NUM_SAMPLES}, BP iter={BP_MAX_ITER}, OSD order={OSD_ORDER})")
print("="*60)
print(f"BP-only Logical Error Rate:    {ler_bp:.4f} ({int(ler_bp*NUM_SAMPLES)} errors)")
print(f"BP+OSD-{OSD_ORDER} Logical Error Rate: {ler_osd:.4f} ({int(ler_osd*NUM_SAMPLES)} errors)")
print("="*60)

if ler_bp > 0:
    improvement = (ler_bp - ler_osd) / ler_bp * 100
    print(f"\nImprovement: {improvement:.1f}% reduction in logical errors")

print(f"Total time: {time.time() - start_time:.0f}s")

# Validate syndrome consistency
print("\nValidating OSD produces valid codewords...")
valid = 0
for i in range(min(100, NUM_SAMPLES)):
    syndrome = syndromes[i]
    # Use uniform probs for validation (just checking syndrome constraint)
    estimated = osd_decoder.solve(syndrome, np.ones(H.shape[1]) * 0.01, osd_order=0)
    if np.array_equal((H @ estimated) % 2, syndrome):
        valid += 1
print(f"Valid syndromes: {valid}/100 (100% expected)")
