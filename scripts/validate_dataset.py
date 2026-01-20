"""
Syndrome Dataset Validation and Demonstration

This script demonstrates the syndrome dataset format and validates its properties.
Run with: uv run python validate_dataset.py
"""

from pathlib import Path
import numpy as np
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.syndrome import sample_syndromes, save_syndrome_database, load_syndrome_database
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix


def generate_and_validate():
    """Generate syndrome dataset and validate it."""

    print("=" * 70)
    print("SYNDROME DATASET GENERATION AND VALIDATION")
    print("=" * 70)

    # Parameters
    distance = 3
    rounds = 3
    p = 0.01
    task = "z"
    num_shots = 100

    print(f"\nParameters:")
    print(f"  Distance (d): {distance}")
    print(f"  Rounds (r): {rounds}")
    print(f"  Error rate (p): {p}")
    print(f"  Task: {task}-memory")
    print(f"  Shots: {num_shots}")

    # Step 1: Generate circuit
    print(f"\n{'='*70}")
    print("STEP 1: Generate Circuit")
    print("="*70)
    circuit = generate_circuit(distance=distance, rounds=rounds, p=p, task=task)
    print(f"✓ Generated surface code circuit")

    # Step 2: Extract DEM
    print(f"\n{'='*70}")
    print("STEP 2: Extract Detector Error Model")
    print("="*70)
    dem = extract_dem(circuit)
    print(f"  Detectors: {dem.num_detectors}")
    print(f"  Observables: {dem.num_observables}")

    # Build parity check matrix
    H, priors, obs_flip = build_parity_check_matrix(dem)
    print(f"\nParity Check Matrix:")
    print(f"  H shape: {H.shape}")
    print(f"  Error mechanisms: {len(priors)}")
    print(f"  Errors flipping observable: {obs_flip.sum()}")

    # Step 3: Sample syndromes
    print(f"\n{'='*70}")
    print("STEP 3: Sample Syndromes")
    print("="*70)
    syndromes, observables = sample_syndromes(circuit, num_shots=num_shots)
    print(f"  Syndromes shape: {syndromes.shape}")
    print(f"  Observables shape: {observables.shape}")
    print(f"  Syndromes dtype: {syndromes.dtype}")
    print(f"  Observables dtype: {observables.dtype}")

    # Step 4: Statistics
    print(f"\n{'='*70}")
    print("STEP 4: Dataset Statistics")
    print("="*70)

    detection_rate = float(syndromes.mean())
    obs_flip_rate = float(observables.mean())
    non_trivial = int((syndromes.sum(axis=1) > 0).sum())

    print(f"\nDetection Events:")
    print(f"  Detection rate: {detection_rate:.4f}")
    print(f"  Mean detections per shot: {syndromes.sum(axis=1).mean():.2f}")
    print(f"  Min detections: {syndromes.sum(axis=1).min()}")
    print(f"  Max detections: {syndromes.sum(axis=1).max()}")
    print(f"  Non-trivial syndromes: {non_trivial}/{num_shots} ({100*non_trivial/num_shots:.1f}%)")

    print(f"\nObservable Flips:")
    print(f"  Logical error rate: {obs_flip_rate:.4f} ({100*obs_flip_rate:.2f}%)")
    print(f"  Successful shots: {num_shots - observables.sum()}/{num_shots}")

    # Step 5: Validation
    print(f"\n{'='*70}")
    print("STEP 5: Validation Checks")
    print("="*70)

    checks_passed = 0
    total_checks = 5

    # Check 1: Dimensions
    try:
        assert syndromes.shape[1] == dem.num_detectors
        print("  ✓ Check 1: Syndrome dimensions match DEM")
        checks_passed += 1
    except AssertionError:
        print("  ✗ Check 1: Syndrome dimensions mismatch!")

    # Check 2: Binary values
    try:
        assert np.all((syndromes == 0) | (syndromes == 1))
        assert np.all((observables == 0) | (observables == 1))
        print("  ✓ Check 2: All values are binary (0 or 1)")
        checks_passed += 1
    except AssertionError:
        print("  ✗ Check 2: Non-binary values found!")

    # Check 3: Detection rate
    try:
        assert 0.01 < detection_rate < 0.1
        print(f"  ✓ Check 3: Detection rate ({detection_rate:.4f}) is reasonable")
        checks_passed += 1
    except AssertionError:
        print(f"  ✗ Check 3: Detection rate ({detection_rate:.4f}) seems wrong!")

    # Check 4: Observable flip rate
    try:
        assert 0 <= obs_flip_rate < 0.05
        print(f"  ✓ Check 4: Observable flip rate ({obs_flip_rate:.4f}) is reasonable")
        checks_passed += 1
    except AssertionError:
        print(f"  ✗ Check 4: Observable flip rate ({obs_flip_rate:.4f}) seems wrong!")

    # Check 5: Non-trivial syndromes
    try:
        assert non_trivial > 0.8 * num_shots
        print(f"  ✓ Check 5: Most syndromes are non-trivial ({100*non_trivial/num_shots:.1f}%)")
        checks_passed += 1
    except AssertionError:
        print(f"  ✗ Check 5: Too few non-trivial syndromes!")

    print(f"\n  Result: {checks_passed}/{total_checks} checks passed")

    # Step 6: Example data
    print(f"\n{'='*70}")
    print("STEP 6: Example Data")
    print("="*70)

    for i in range(min(3, num_shots)):
        detectors_fired = np.where(syndromes[i])[0]
        print(f"\nShot #{i}:")
        print(f"  Detectors fired: {detectors_fired.tolist()}")
        print(f"  Number of detections: {len(detectors_fired)}")
        print(f"  Observable flip: {bool(observables[i])}")

    # Step 7: Save dataset
    print(f"\n{'='*70}")
    print("STEP 7: Save Dataset")
    print("="*70)

    output_dir = Path("datasets/noisy_circuits")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"sc_d{distance}_r{rounds}_p{int(p*10000):04d}_{task}_demo.npz"

    metadata = {
        "distance": distance,
        "rounds": rounds,
        "p": p,
        "task": task,
        "num_shots": num_shots,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
    }

    save_syndrome_database(syndromes, observables, output_path, metadata)
    print(f"  ✓ Saved to: {output_path}")

    # Step 8: Load and verify
    print(f"\n{'='*70}")
    print("STEP 8: Load and Verify")
    print("="*70)

    loaded_syndromes, loaded_observables, loaded_metadata = load_syndrome_database(output_path)

    print(f"  Loaded syndromes shape: {loaded_syndromes.shape}")
    print(f"  Loaded observables shape: {loaded_observables.shape}")
    print(f"  Loaded metadata: {loaded_metadata}")

    # Verify round-trip
    assert np.array_equal(syndromes, loaded_syndromes)
    assert np.array_equal(observables, loaded_observables)
    print(f"  ✓ Round-trip verification passed")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"✓ Dataset generated successfully")
    print(f"✓ All validation checks passed ({checks_passed}/{total_checks})")
    print(f"✓ Data saved and verified")
    print(f"\nDataset location: {output_path}")
    print(f"Dataset size: {output_path.stat().st_size / 1024:.2f} KB")

    return checks_passed == total_checks


if __name__ == "__main__":
    try:
        success = generate_and_validate()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
