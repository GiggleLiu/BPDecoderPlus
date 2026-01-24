#!/usr/bin/env python3
"""
Generate syndrome datasets for threshold analysis.

This script generates datasets for multiple code distances and error rates
to enable threshold curve plotting for the BP+OSD decoder.

Usage:
    uv run python scripts/generate_threshold_datasets.py
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.syndrome import sample_syndromes, save_syndrome_database
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix

# Configuration for threshold analysis
# Circuit-level threshold is ~0.5-1% per noise source, so total ~0.1-0.25%.
# Test error rates spanning below and above threshold.
DISTANCES = [3, 5, 7, 9, 11]
ERROR_RATES = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.009, 0.01, 0.012, 0.015]
NUM_SHOTS = 20000


def generate_dataset(distance: int, error_rate: float, num_shots: int, output_dir: Path):
    """
    Generate a single dataset for given distance and error rate.

    Args:
        distance: Code distance (d)
        error_rate: Physical error rate (p)
        num_shots: Number of syndrome samples
        output_dir: Directory to save output files
    """
    rounds = distance  # Standard choice: r = d

    # Format error rate for filename (e.g., 0.0010 -> "0010", 0.01 -> "0100")
    p_str = f"{error_rate:.4f}"[2:]  # Remove "0." prefix
    print(p_str)
    base_name = f"sc_d{distance}_r{rounds}_p{p_str}_z"

    dem_path = output_dir / f"{base_name}.dem"
    npz_path = output_dir / f"{base_name}.npz"

    # Skip if already exists
    if dem_path.exists() and npz_path.exists():
        print(f"  Skipping {base_name} (already exists)")
        return

    print(f"  Generating {base_name}...")

    # Generate circuit
    circuit = generate_circuit(distance=distance, rounds=rounds, p=error_rate, task="z")

    # Extract DEM
    dem = extract_dem(circuit)

    # Build unseparated parity check matrix
    H, priors, obs_flip = build_parity_check_matrix(dem)

    # Sample syndromes
    syndromes, observables = sample_syndromes(circuit, num_shots=num_shots)

    # Save DEM file (use str() for proper stim format)
    with open(dem_path, 'w') as f:
        f.write(str(dem))

    # Save syndrome database
    metadata = {
        "distance": distance,
        "rounds": rounds,
        "p": error_rate,
        "task": "z",
        "num_shots": num_shots,
        "num_detectors": dem.num_detectors,
    }
    save_syndrome_database(syndromes, observables, npz_path, metadata)

    # Print statistics
    detection_rate = syndromes.mean()
    obs_flip_rate = observables.mean()
    print(f"    H shape: {H.shape}, detection rate: {detection_rate:.4f}, obs flip rate: {obs_flip_rate:.4f}")


def main():
    """Generate all threshold datasets."""
    output_dir = Path("datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_datasets = len(DISTANCES) * len(ERROR_RATES)
    print(f"Generating {total_datasets} datasets for threshold analysis")
    print(f"Distances: {DISTANCES}")
    print(f"Error rates: {ERROR_RATES}")
    print(f"Shots per dataset: {NUM_SHOTS}")
    print()

    for d in DISTANCES:
        print(f"Distance d={d}:")
        for p in ERROR_RATES:
            generate_dataset(d, p, NUM_SHOTS, output_dir)
        print()

    print("Dataset generation complete!")
    print(f"Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
