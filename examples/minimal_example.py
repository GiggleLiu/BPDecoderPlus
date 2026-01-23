"""
Minimum Working Example: Complete Pipeline
===========================================

This script demonstrates the complete pipeline from circuit generation
to syndrome sampling and decoder preparation.

Run with: uv run python examples/minimal_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import sample_syndromes


def main():
    print("=" * 70)
    print("MINIMUM WORKING EXAMPLE: Quantum Error Correction Pipeline")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Generate a noisy quantum circuit
    # ========================================================================
    print("\n[STEP 1] Generate Noisy Circuit")
    print("-" * 70)

    circuit = generate_circuit(
        distance=3,      # Code distance (larger = more error correction)
        rounds=3,        # Number of syndrome measurement rounds
        p=0.01,          # Physical error rate (1% per operation)
        task="z"         # Z-memory experiment
    )

    print(f"✓ Generated surface code circuit")
    print(f"  - Distance: 3 (can correct 1 error)")
    print(f"  - Rounds: 3 (measure syndrome 3 times)")
    print(f"  - Error rate: 1% per gate/measurement")

    # ========================================================================
    # STEP 2: Extract Detector Error Model (DEM)
    # ========================================================================
    print("\n[STEP 2] Extract Detector Error Model")
    print("-" * 70)

    dem = extract_dem(circuit, decompose_errors=True)

    print(f"✓ Extracted DEM from circuit")
    print(f"  - Detectors: {dem.num_detectors}")
    print(f"  - Observables: {dem.num_observables}")
    print(f"\nWhat is a DEM?")
    print(f"  The DEM describes which errors trigger which detectors.")
    print(f"  It's the 'rulebook' that connects physical errors to")
    print(f"  detection events that we can observe.")

    # ========================================================================
    # STEP 3: Build Parity Check Matrix for BP Decoder
    # ========================================================================
    print("\n[STEP 3] Build Parity Check Matrix")
    print("-" * 70)

    H, priors, obs_flip = build_parity_check_matrix(dem)

    print(f"✓ Built parity check matrix H")
    print(f"  - H shape: {H.shape}")
    print(f"  - H[i,j] = 1 means error j triggers detector i")
    print(f"  - Prior probabilities: {len(priors)} error mechanisms")
    print(f"  - Observable flips: {obs_flip.sum()} errors flip logical qubit")
    print(f"\nWhat is H used for?")
    print(f"  H is the input to Belief Propagation (BP) decoder.")
    print(f"  BP uses H to infer which errors occurred from syndromes.")

    # ========================================================================
    # STEP 4: Sample Syndromes (Detection Events)
    # ========================================================================
    print("\n[STEP 4] Sample Syndromes")
    print("-" * 70)

    num_shots = 10
    syndromes, observables = sample_syndromes(circuit, num_shots=num_shots)

    print(f"✓ Sampled {num_shots} syndrome measurements")
    print(f"  - Syndromes shape: {syndromes.shape}")
    print(f"  - Each row is one 'shot' (one circuit execution)")
    print(f"  - Each column is one detector (did it fire?)")
    print(f"\nWhat is a syndrome?")
    print(f"  A syndrome is a binary vector showing which detectors fired.")
    print(f"  Detectors fire when errors occur in their region.")

    # ========================================================================
    # STEP 5: Examine Example Data
    # ========================================================================
    print("\n[STEP 5] Examine Example Syndrome")
    print("-" * 70)

    # Pick first shot
    syndrome = syndromes[0]
    observable = observables[0]

    detectors_fired = np.where(syndrome)[0]

    print(f"Shot #0:")
    print(f"  Syndrome: {syndrome.astype(int)}")
    print(f"  Detectors fired: {detectors_fired.tolist()}")
    print(f"  Number of detections: {len(detectors_fired)}")
    print(f"  Observable flip: {bool(observable)}")
    print(f"\nInterpretation:")
    if len(detectors_fired) == 0:
        print(f"  No detectors fired → No errors detected")
    else:
        print(f"  {len(detectors_fired)} detectors fired → Errors occurred")
    if observable:
        print(f"  Observable flipped → LOGICAL ERROR (decoder failed)")
    else:
        print(f"  Observable stable → No logical error (decoder succeeded)")

    # ========================================================================
    # STEP 6: Decoder Task (Conceptual)
    # ========================================================================
    print("\n[STEP 6] Decoder Task (Conceptual)")
    print("-" * 70)

    print(f"Given:")
    print(f"  - Syndrome (what we observe): {syndrome.astype(int)}")
    print(f"  - Parity check matrix H: {H.shape}")
    print(f"  - Prior probabilities: {priors.shape}")
    print(f"\nDecoder's job:")
    print(f"  1. Use H and syndrome to infer which errors occurred")
    print(f"  2. Predict if those errors flip the logical observable")
    print(f"  3. Compare prediction with actual observable")
    print(f"\nSuccess criterion:")
    print(f"  Predicted observable == Actual observable")

    # ========================================================================
    # STEP 7: Statistics
    # ========================================================================
    print("\n[STEP 7] Dataset Statistics")
    print("-" * 70)

    detection_rate = syndromes.mean()
    logical_error_rate = observables.mean()
    non_trivial = (syndromes.sum(axis=1) > 0).sum()

    print(f"Detection events:")
    print(f"  - Detection rate: {detection_rate:.4f}")
    print(f"  - Mean detections per shot: {syndromes.sum(axis=1).mean():.2f}")
    print(f"  - Non-trivial syndromes: {non_trivial}/{num_shots}")

    print(f"\nLogical errors:")
    print(f"  - Logical error rate: {logical_error_rate:.4f}")
    print(f"  - Successful shots: {num_shots - observables.sum()}/{num_shots}")

    print(f"\nWhy these numbers make sense:")
    print(f"  - Physical error rate p=0.01 (1%)")
    print(f"  - Detection rate ~3-5% (errors trigger nearby detectors)")
    print(f"  - Logical error rate ~0.5-1% (d=3 code corrects most errors)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(f"""
Circuit → DEM → Parity Check Matrix → Syndromes → Decoder
   ↓       ↓            ↓                  ↓          ↓
 .stim   .dem      H, priors, obs      .npz      Predictions

1. Circuit: Defines the quantum operations and noise
2. DEM: Maps errors to detector outcomes
3. H matrix: Input for BP decoder
4. Syndromes: Observed detection events
5. Decoder: Predicts logical errors from syndromes

Next steps:
- Implement BP decoder using H matrix
- Train/test decoder on syndrome datasets
- Evaluate logical error rate
    """)

    print("✓ Pipeline demonstration complete!")


if __name__ == "__main__":
    main()
