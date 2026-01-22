import numpy as np
import torch
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.osd import OSDDecoder

dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
H, priors, obs_flip = build_parity_check_matrix(dem)

bp_decoder = BatchBPDecoder(H, priors, device='cpu')
osd_decoder = OSDDecoder(H)

# Test 500 samples
num_samples = 500
print(f"Testing on {num_samples} samples from d=3, r=3, p=0.010 surface code")
print(f"Dataset: datasets/sc_d3_r3_p0010_z.npz")
print(f"Parity check matrix: {H.shape[0]} checks x {H.shape[1]} qubits")
print()

batch_syndromes = torch.from_numpy(syndromes[:num_samples]).float()
marginals = bp_decoder.decode(batch_syndromes, max_iter=20, damping=0.2)

# Generate predictions
pred_bp = []
pred_osd10 = []
pred_osd15 = []

for i in range(num_samples):
    probs = marginals[i].cpu().numpy()
    pred_bp.append(np.dot((probs > 0.5).astype(int), obs_flip) % 2)
    pred_osd10.append(np.dot(osd_decoder.solve(syndromes[i], probs, osd_order=10), obs_flip) % 2)
    pred_osd15.append(np.dot(osd_decoder.solve(syndromes[i], probs, osd_order=15), obs_flip) % 2)

# Calculate baseline (do-nothing error rate)
baseline_ler = np.mean(observables[:num_samples])

# Calculate decoder logical error rates
ler_bp = np.mean(np.array(pred_bp) != observables[:num_samples])
ler_osd10 = np.mean(np.array(pred_osd10) != observables[:num_samples])
ler_osd15 = np.mean(np.array(pred_osd15) != observables[:num_samples])

# Print results with baseline comparison
print(f"{'='*60}")
print(f"Results (n={num_samples}, d=3, r=3, p=0.010)")
print(f"{'='*60}")
print(f"Baseline (no correction):  {baseline_ler:.2%} ({int(baseline_ler*num_samples)}/{num_samples} errors)")
print(f"BP-only (iter=20):         {ler_bp:.2%} ({int(ler_bp*num_samples)}/{num_samples} errors)")
print(f"BP+OSD-10:                 {ler_osd10:.2%} ({int(ler_osd10*num_samples)}/{num_samples} errors)")
print(f"BP+OSD-15:                 {ler_osd15:.2%} ({int(ler_osd15*num_samples)}/{num_samples} errors)")
print(f"{'='*60}")

# Check if decoders are effective
print("\nDecoder Effectiveness:")
if ler_bp > baseline_ler:
    print(f"\033[91m⚠ BP-only is making things WORSE! ({ler_bp:.2%} > {baseline_ler:.2%})\033[0m")
else:
    print(f"✓ BP-only is effective ({ler_bp:.2%} ≤ {baseline_ler:.2%})")

if ler_osd10 > baseline_ler:
    print(f"\033[91m⚠ BP+OSD-10 is making things WORSE! ({ler_osd10:.2%} > {baseline_ler:.2%})\033[0m")
else:
    print(f"✓ BP+OSD-10 is effective ({ler_osd10:.2%} ≤ {baseline_ler:.2%})")

if ler_osd15 > baseline_ler:
    print(f"\033[91m⚠ BP+OSD-15 is making things WORSE! ({ler_osd15:.2%} > {baseline_ler:.2%})\033[0m")
else:
    print(f"✓ BP+OSD-15 is effective ({ler_osd15:.2%} ≤ {baseline_ler:.2%})")
