"""
Circuit generation module for noisy surface-code circuits.

This module provides functions to generate rotated surface-code memory
experiments with circuit-level depolarizing noise using Stim.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import stim


def parse_rounds(values: Iterable[int]) -> list[int]:
    """
    Parse and validate round counts.

    Args:
        values: Iterable of round counts.

    Returns:
        Sorted list of unique positive integers.

    Raises:
        ValueError: If no positive integers are provided.
    """
    unique = sorted({int(v) for v in values if int(v) > 0})
    if not unique:
        raise ValueError("At least one positive integer round count is required.")
    return unique


def prob_tag(p: float) -> str:
    """
    Convert probability to filename-safe tag.

    Args:
        p: Probability value (e.g., 0.01).

    Returns:
        String tag (e.g., "p0010" for p=0.01).
    """
    # Format as 3 decimal places without decimal point
    # e.g., 0.01 -> "p0010", 0.001 -> "p0001"
    return f"p{p:.3f}".replace(".", "")


def generate_circuit(
    distance: int,
    rounds: int,
    p: float,
    task: str = "z",
) -> stim.Circuit:
    """
    Build a rotated memory surface-code circuit with depolarizing noise.

    Args:
        distance: Surface code distance.
        rounds: Number of measurement rounds.
        p: Depolarizing error rate (must be in (0, 1)).
        task: Memory experiment orientation, either "z" or "x".

    Returns:
        Stim circuit with noise applied to:
        - Clifford gates (after_clifford_depolarization)
        - Data qubits between rounds (before_round_data_depolarization)
        - Measurements (before_measure_flip_probability)
        - Resets (after_reset_flip_probability)

    Raises:
        ValueError: If task is not 'x' or 'z'.
        ValueError: If p is not in (0, 1).
    """
    if task not in {"x", "z"}:
        raise ValueError("task must be 'x' or 'z'")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    name = f"surface_code:rotated_memory_{task}"
    return stim.Circuit.generated(
        name,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def run_smoke_test(circuit: stim.Circuit, shots: int = 4) -> None:
    """
    Quick structural check: compile detector sampler and draw samples.

    Args:
        circuit: Stim circuit to test.
        shots: Number of samples to draw.

    Raises:
        Exception: If the circuit is invalid or cannot be sampled.
    """
    sampler = circuit.compile_detector_sampler()
    sampler.sample(shots)


def write_circuit(circuit: stim.Circuit, path: pathlib.Path) -> None:
    """
    Write a Stim circuit to a file.

    Args:
        circuit: Stim circuit to write.
        path: Output file path.
    """
    path.write_text(str(circuit))


def generate_filename(
    distance: int,
    rounds: int,
    p: float,
    task: str,
) -> str:
    """
    Generate a standardized filename for a circuit.

    Args:
        distance: Surface code distance.
        rounds: Number of measurement rounds.
        p: Error probability.
        task: Memory experiment orientation.

    Returns:
        Filename string (e.g., "sc_d3_r5_p0010_z.stim").
    """
    return f"sc_d{distance}_r{rounds}_{prob_tag(p)}_{task}.stim"
