#!/usr/bin/env python3
"""
Generate circuit-level syndrome datasets for decoder benchmarking.

This script uses Stim to generate:
1. Detector Error Model (DEM) files
2. Detection event datasets (syndrome samples)
3. Observable flip labels (for training/evaluation)

Output format is compatible with tesseract-decoder and other Stim-based decoders.

Reference: https://github.com/quantumlib/tesseract-decoder
           https://github.com/quantumlib/Stim

Usage:
    python python/generate_circuit_data.py [--quick]
    python python/generate_circuit_data.py --distance 5 --rounds 5 --shots 100000
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path

try:
    import stim
except ImportError:
    raise ImportError("Stim is required. Install with: pip install stim")


def generate_surface_code_circuit(
    distance: int,
    rounds: int,
    noise_model: str = "depolarizing",
    p_error: float = 0.001,
    p_meas: float = None,
    p_reset: float = None,
) -> stim.Circuit:
    """
    Generate a noisy surface code memory circuit using Stim.

    Args:
        distance: Code distance (d)
        rounds: Number of syndrome extraction rounds
        noise_model: Type of noise model ("depolarizing", "phenomenological", "SI1000")
        p_error: Base error probability
        p_meas: Measurement error probability (defaults to p_error)
        p_reset: Reset error probability (defaults to p_error)

    Returns:
        Stim circuit with noise, detectors, and observables annotated
    """
    if p_meas is None:
        p_meas = p_error
    if p_reset is None:
        p_reset = p_error

    if noise_model == "depolarizing":
        # Standard depolarizing noise model
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p_error,
            after_reset_flip_probability=p_reset,
            before_measure_flip_probability=p_meas,
            before_round_data_depolarization=p_error,
        )
    elif noise_model == "phenomenological":
        # Phenomenological noise (only measurement errors)
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            before_measure_flip_probability=p_meas,
        )
    elif noise_model == "SI1000":
        # Google's SI1000 noise model (superconducting qubits)
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p_error,
            after_reset_flip_probability=p_reset,
            before_measure_flip_probability=p_meas,
            before_round_data_depolarization=p_error * 10,  # Higher idle errors
        )
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    return circuit


def extract_detector_error_model(
    circuit: stim.Circuit,
    decompose_errors: bool = True,
) -> stim.DetectorErrorModel:
    """
    Extract the detector error model from a circuit.

    Args:
        circuit: Stim circuit with noise
        decompose_errors: If True, decompose hyperedges into graphlike edges

    Returns:
        Detector error model (DEM)
    """
    return circuit.detector_error_model(decompose_errors=decompose_errors)


def sample_detection_events(
    circuit: stim.Circuit,
    num_shots: int,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample detection events and observable flips from a circuit.

    Args:
        circuit: Stim circuit
        num_shots: Number of shots to sample
        seed: Random seed for reproducibility

    Returns:
        Tuple of (detection_events, observable_flips)
        - detection_events: shape (num_shots, num_detectors), dtype=uint8
        - observable_flips: shape (num_shots, num_observables), dtype=uint8
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(
        num_shots,
        separate_observables=True
    )
    return detection_events.astype(np.uint8), observable_flips.astype(np.uint8)


def save_dem_file(dem: stim.DetectorErrorModel, filepath: str):
    """Save detector error model to file."""
    with open(filepath, 'w') as f:
        f.write(str(dem))


def save_detection_events_01(events: np.ndarray, filepath: str):
    """
    Save detection events in Stim's .01 format.

    The .01 format is a binary format where each line is a shot,
    and each character is '0' or '1' for each detector.
    """
    with open(filepath, 'w') as f:
        for shot in events:
            f.write(''.join(str(b) for b in shot) + '\n')


def save_detection_events_b8(events: np.ndarray, filepath: str):
    """
    Save detection events in Stim's .b8 format (packed bits).

    More space-efficient than .01 format.
    """
    packed = np.packbits(events, axis=1, bitorder='little')
    packed.tofile(filepath)


def save_observable_flips(obs: np.ndarray, filepath: str):
    """Save observable flips in .01 format."""
    with open(filepath, 'w') as f:
        for shot in obs:
            f.write(''.join(str(b) for b in shot) + '\n')


def save_metadata(metadata: dict, filepath: str):
    """Save dataset metadata to JSON."""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_dataset(
    output_dir: str,
    distance: int,
    rounds: int,
    p_error: float,
    num_shots: int,
    noise_model: str = "depolarizing",
    seed: int = None,
    save_b8: bool = True,
):
    """
    Generate a complete circuit-level dataset.

    Creates:
        - {prefix}.dem: Detector error model
        - {prefix}.stim: Stim circuit
        - {prefix}_events.01: Detection events (text format)
        - {prefix}_events.b8: Detection events (binary format, optional)
        - {prefix}_obs.01: Observable flips
        - {prefix}_metadata.json: Dataset metadata

    Args:
        output_dir: Output directory
        distance: Code distance
        rounds: Number of syndrome rounds
        p_error: Physical error rate
        num_shots: Number of samples
        noise_model: Noise model type
        seed: Random seed
        save_b8: Also save binary format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename prefix
    prefix = f"surface_d{distance}_r{rounds}_p{p_error:.4f}"

    print(f"  Generating dataset: {prefix}")
    print(f"    Distance: {distance}, Rounds: {rounds}, p={p_error}, shots={num_shots}")

    # Generate circuit
    circuit = generate_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        noise_model=noise_model,
        p_error=p_error,
    )

    # Extract DEM
    dem = extract_detector_error_model(circuit, decompose_errors=True)

    # Sample detection events
    detection_events, observable_flips = sample_detection_events(
        circuit, num_shots, seed=seed
    )

    # Get statistics
    num_detectors = detection_events.shape[1]
    num_observables = observable_flips.shape[1]
    logical_error_rate = np.mean(observable_flips)

    print(f"    Detectors: {num_detectors}, Observables: {num_observables}")
    print(f"    Logical error rate: {logical_error_rate:.4f}")

    # Save files
    circuit_path = os.path.join(output_dir, f"{prefix}.stim")
    dem_path = os.path.join(output_dir, f"{prefix}.dem")
    events_01_path = os.path.join(output_dir, f"{prefix}_events.01")
    obs_path = os.path.join(output_dir, f"{prefix}_obs.01")
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")

    # Save circuit
    with open(circuit_path, 'w') as f:
        f.write(str(circuit))

    # Save DEM
    save_dem_file(dem, dem_path)

    # Save detection events
    save_detection_events_01(detection_events, events_01_path)

    if save_b8:
        events_b8_path = os.path.join(output_dir, f"{prefix}_events.b8")
        save_detection_events_b8(detection_events, events_b8_path)

    # Save observable flips
    save_observable_flips(observable_flips, obs_path)

    # Save metadata
    metadata = {
        "code": "surface_code:rotated_memory_z",
        "distance": distance,
        "rounds": rounds,
        "p_error": p_error,
        "noise_model": noise_model,
        "num_shots": num_shots,
        "num_detectors": num_detectors,
        "num_observables": num_observables,
        "logical_error_rate": float(logical_error_rate),
        "seed": seed,
        "files": {
            "circuit": f"{prefix}.stim",
            "dem": f"{prefix}.dem",
            "events_01": f"{prefix}_events.01",
            "events_b8": f"{prefix}_events.b8" if save_b8 else None,
            "observables": f"{prefix}_obs.01",
        }
    }
    save_metadata(metadata, metadata_path)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate circuit-level syndrome datasets using Stim"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer samples")
    parser.add_argument("--distance", "-d", type=int, default=None,
                        help="Single code distance to generate")
    parser.add_argument("--rounds", "-r", type=int, default=None,
                        help="Number of syndrome rounds")
    parser.add_argument("--shots", "-n", type=int, default=None,
                        help="Number of shots to sample")
    parser.add_argument("--p-error", "-p", type=float, default=0.001,
                        help="Physical error rate")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="benchmark/circuit_data",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--noise-model", type=str, default="depolarizing",
                        choices=["depolarizing", "phenomenological", "SI1000"],
                        help="Noise model type")

    args = parser.parse_args()

    print("=" * 60)
    print("Circuit-Level Syndrome Dataset Generation")
    print("=" * 60)

    # Determine parameters
    if args.distance is not None:
        # Single dataset mode
        distances = [args.distance]
        rounds_list = [args.rounds if args.rounds else args.distance]
        error_rates = [args.p_error]
        num_shots = args.shots if args.shots else 10000
    elif args.quick:
        # Quick mode
        distances = [3, 5]
        rounds_list = None  # Will use distance as rounds
        error_rates = [0.001, 0.005, 0.01]
        num_shots = 10000
    else:
        # Full mode
        distances = [3, 5, 7, 9]
        rounds_list = None
        error_rates = [0.001, 0.002, 0.005, 0.01]
        num_shots = 100000

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Noise model: {args.noise_model}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    all_metadata = []

    for d in distances:
        rounds = rounds_list[distances.index(d)] if rounds_list else d

        for p in error_rates:
            print(f"\n>>> Distance {d}, Rounds {rounds}, p={p}")

            metadata = generate_dataset(
                output_dir=output_dir,
                distance=d,
                rounds=rounds,
                p_error=p,
                num_shots=num_shots,
                noise_model=args.noise_model,
                seed=args.seed,
            )
            all_metadata.append(metadata)

    # Save combined metadata
    combined_metadata = {
        "datasets": all_metadata,
        "noise_model": args.noise_model,
        "seed": args.seed,
    }
    combined_path = os.path.join(output_dir, "datasets_index.json")
    save_metadata(combined_metadata, combined_path)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Generated {len(all_metadata)} datasets")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Print usage example
    print("\nUsage with tesseract-decoder:")
    if all_metadata:
        example = all_metadata[0]
        prefix = f"surface_d{example['distance']}_r{example['rounds']}_p{example['p_error']:.4f}"
        print(f"  ./tesseract --dem {output_dir}/{prefix}.dem \\")
        print(f"              --in {output_dir}/{prefix}_events.01 \\")
        print(f"              --in-format 01")


if __name__ == "__main__":
    main()
