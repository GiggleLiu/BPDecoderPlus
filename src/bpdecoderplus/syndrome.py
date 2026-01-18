"""
Syndrome database generation module for noisy surface-code circuits.

This module provides functions to sample detection events (syndromes) from
circuits and save them in a structured format for decoder training/testing.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import stim


def sample_syndromes(
    circuit: stim.Circuit,
    num_shots: int,
    include_observables: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Sample detection events (syndromes) from a circuit.

    Args:
        circuit: Stim circuit to sample from.
        num_shots: Number of syndrome samples to generate.
        include_observables: Whether to include observable flip outcomes.

    Returns:
        Tuple of (syndromes, observables) where:
        - syndromes: Array of shape (num_shots, num_detectors)
        - observables: Array of shape (num_shots,) if include_observables else None
    """
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(num_shots, append_observables=include_observables)

    if include_observables:
        syndromes = samples[:, :-1]
        observables = samples[:, -1]
        return syndromes, observables
    else:
        return samples, None


def save_syndrome_database(
    syndromes: np.ndarray,
    observables: np.ndarray | None,
    output_path: pathlib.Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save syndrome database to disk in npz format.

    Args:
        syndromes: Array of detection events, shape (num_shots, num_detectors).
        observables: Array of observable flips, shape (num_shots,), or None.
        output_path: Path to save the database (.npz file).
        metadata: Optional metadata dictionary to save alongside the data.
    """
    save_dict = {"syndromes": syndromes}

    if observables is not None:
        save_dict["observables"] = observables

    if metadata is not None:
        # Save metadata as JSON string in the npz file
        save_dict["metadata"] = np.array([json.dumps(metadata)])

    np.savez_compressed(output_path, **save_dict)


def load_syndrome_database(
    input_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    """
    Load syndrome database from disk.

    Args:
        input_path: Path to the database file (.npz).

    Returns:
        Tuple of (syndromes, observables, metadata) where:
        - syndromes: Array of shape (num_shots, num_detectors)
        - observables: Array of shape (num_shots,) or None
        - metadata: Dictionary of metadata or None
    """
    data = np.load(input_path, allow_pickle=True)

    syndromes = data["syndromes"]
    observables = data.get("observables", None)

    metadata = None
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"][0]))

    return syndromes, observables, metadata


def generate_syndrome_database_from_circuit(
    circuit_path: pathlib.Path,
    num_shots: int,
    output_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Generate and save syndrome database from a circuit file.

    Args:
        circuit_path: Path to the circuit file (.stim).
        num_shots: Number of syndrome samples to generate.
        output_path: Optional output path. If None, uses circuit_path with .npz extension.

    Returns:
        Path to the saved database file.
    """
    # Load circuit
    circuit = stim.Circuit.from_file(str(circuit_path))

    # Generate output path if not provided
    if output_path is None:
        output_path = circuit_path.with_suffix(".npz")

    # Sample syndromes
    syndromes, observables = sample_syndromes(circuit, num_shots, include_observables=True)

    # Create metadata
    dem = circuit.detector_error_model()
    metadata = {
        "circuit_file": str(circuit_path.name),
        "num_shots": num_shots,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
    }

    # Save database
    save_syndrome_database(syndromes, observables, output_path, metadata)

    return output_path
