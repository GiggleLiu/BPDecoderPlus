#!/usr/bin/env python3
"""
Generate syndrome dataset and validate it.
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.syndrome import sample_syndromes, save_syndrome_database
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix

# Generate circuit
print("Generating surface code circuit (d=3, r=3, p=0.01)...")
circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

# Extract DEM
print("Extracting detector error model...")
dem = extract_dem(circuit)
print(f"  - Detectors: {dem.num_detectors}")
print(f"  - Observables: {dem.num_observables}")

# Build parity check matrix
print("\nBuilding parity check matrix...")
H, priors, obs_flip = build_parity_check_matrix(dem)
print(f"  - H shape: {H.shape}")
print(f"  - Number of error mechanisms: {len(priors)}")
print(f"  - Errors that flip observable: {obs_flip.sum()}")

# Sample syndromes
print("\nSampling syndromes (100 shots)...")
syndromes, observables = sample_syndromes(circuit, num_shots=100)
print(f"  - Syndromes shape: {syndromes.shape}")
print(f"  - Observables shape: {observables.shape}")
print(f"  - Syndrome dtype: {syndromes.dtype}")

# Statistics
print("\nDataset statistics:")
print(f"  - Detection event rate: {syndromes.mean():.4f}")
print(f"  - Observable flip rate: {observables.mean():.4f}")
print(f"  - Non-trivial syndromes: {(syndromes.sum(axis=1) > 0).sum()}/{len(syndromes)}")

# Validate consistency
print("\nValidation checks:")
# Check 1: Syndrome dimensions match DEM
assert syndromes.shape[1] == dem.num_detectors, "Syndrome dimension mismatch!"
print("  ✓ Syndrome dimensions match DEM")

# Check 2: Values are binary
assert np.all((syndromes == 0) | (syndromes == 1)), "Syndromes not binary!"
assert np.all((observables == 0) | (observables == 1)), "Observables not binary!"
print("  ✓ All values are binary")

# Check 3: Detection events are sparse (for low error rate)
detection_rate = syndromes.mean()
assert 0 < detection_rate < 0.5, f"Detection rate {detection_rate} seems wrong!"
print(f"  ✓ Detection rate ({detection_rate:.4f}) is reasonable for p=0.01")

# Save dataset
output_dir = Path("datasets/noisy_circuits")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "sc_d3_r3_p0010_z_demo.npz"
metadata = {
    "distance": 3,
    "rounds": 3,
    "p": 0.01,
    "task": "z",
    "num_shots": 100,
    "num_detectors": dem.num_detectors,
}
save_syndrome_database(syndromes, observables, output_path, metadata)
print(f"\n✓ Dataset saved to {output_path}")

# Show example syndrome
print("\nExample syndrome (first shot):")
print(f"  Detectors fired: {np.where(syndromes[0])[0].tolist()}")
print(f"  Observable flip: {bool(observables[0])}")

print("\n✓ All validation checks passed!")
