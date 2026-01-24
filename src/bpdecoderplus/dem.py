"""
Detector Error Model (DEM) extraction module for noisy circuits.

This module provides functions to extract and save Detector Error Models
from Stim circuits for use in decoder implementations.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import stim


def extract_dem(
    circuit: stim.Circuit,
    decompose_errors: bool = True,
) -> stim.DetectorErrorModel:
    """
    Extract Detector Error Model from a circuit.

    Args:
        circuit: Stim circuit to extract DEM from.
        decompose_errors: Whether to decompose errors into components.

    Returns:
        Detector Error Model describing error mechanisms.
    """
    return circuit.detector_error_model(decompose_errors=decompose_errors)


def save_dem(
    dem: stim.DetectorErrorModel,
    output_path: pathlib.Path,
) -> None:
    """
    Save Detector Error Model to file in stim format.

    Args:
        dem: Detector Error Model to save.
        output_path: Path to save the DEM (.dem file).
    """
    output_path.write_text(str(dem))


def load_dem(input_path: pathlib.Path) -> stim.DetectorErrorModel:
    """
    Load Detector Error Model from file.

    Args:
        input_path: Path to the DEM file (.dem).

    Returns:
        Loaded Detector Error Model.
    """
    return stim.DetectorErrorModel.from_file(str(input_path))


def _split_error_by_separator(targets: list) -> list[dict]:
    """
    Split error targets by ^ separator into independent components.

    In DEM format, error(p) D0 D1 ^ D2 means probability p error triggers
    {D0, D1} AND {D2} simultaneously. These should be treated as separate
    correlated fault locations that share the same probability.

    Args:
        targets: List of stim.DemTarget from an error instruction.

    Returns:
        List of component dicts, each with 'detectors' and 'observables' lists.
    """
    components = []
    current_detectors = []
    current_observables = []

    for t in targets:
        if t.is_separator():
            if current_detectors or current_observables:
                components.append({
                    "detectors": current_detectors,
                    "observables": current_observables,
                })
            current_detectors = []
            current_observables = []
        elif t.is_relative_detector_id():
            current_detectors.append(t.val)
        elif t.is_logical_observable_id():
            current_observables.append(t.val)

    # Don't forget the last component
    if current_detectors or current_observables:
        components.append({
            "detectors": current_detectors,
            "observables": current_observables,
        })

    return components


def dem_to_dict(dem: stim.DetectorErrorModel) -> dict[str, Any]:
    """
    Convert DEM to dictionary with structured information.

    Args:
        dem: Detector Error Model to convert.

    Returns:
        Dictionary with DEM statistics and error information.
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()

            for comp in targets:
                errors.append({
                    "probability": float(prob),
                    "detectors": comp["detectors"],
                    "observables": comp["observables"],
                })

    return {
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "num_errors": len(errors),
        "errors": errors,
    }


def save_dem_json(
    dem: stim.DetectorErrorModel,
    output_path: pathlib.Path,
) -> None:
    """
    Save DEM as JSON for easier analysis.

    Args:
        dem: Detector Error Model to save.
        output_path: Path to save the JSON file.
    """
    dem_dict = dem_to_dict(dem)
    with open(output_path, "w") as f:
        json.dump(dem_dict, f, indent=2)


def build_parity_check_matrix(
    dem: stim.DetectorErrorModel,
    merge_hyperedges: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix H from DEM for BP decoding.

    This function extracts error mechanisms from the DEM and constructs
    the parity check matrix. When merge_hyperedges=True (default), errors
    with identical detector patterns are merged into single hyperedges
    with combined probabilities, which is required for correct BP decoding.

    Args:
        dem: Detector Error Model.
        merge_hyperedges: If True, merge errors with identical detector patterns
            into single hyperedges with combined probabilities. This is the
            correct approach for BP decoding. If False, create one column per
            error instruction (legacy behavior, not recommended).

    Returns:
        Tuple of (H, priors, obs_flip) where:
        - H: Parity check matrix, shape (num_detectors, num_hyperedges)
        - priors: Prior error probabilities, shape (num_hyperedges,)
        - obs_flip: Observable flip probabilities, shape (num_hyperedges,)
            When merge_hyperedges=True, this contains the probability that
            the observable is flipped (can be fractional).
            When merge_hyperedges=False, this is binary (0 or 1).
    """
    if merge_hyperedges:
        return _build_parity_check_matrix_hyperedge(dem)
    else:
        return _build_parity_check_matrix_legacy(dem)


def _build_parity_check_matrix_legacy(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy implementation: one column per error instruction.

    This does NOT merge duplicate detector patterns and creates an invalid
    factor graph for BP. Use build_parity_check_matrix with merge_hyperedges=True
    instead.
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()

            # Split by separator into independent components
            components = _split_error_by_separator(targets)

            for comp in components:
                errors.append({
                    "prob": prob,
                    "detectors": comp["detectors"],
                    "observables": comp["observables"],
                })

    n_detectors = dem.num_detectors
    n_errors = len(errors)

    H = np.zeros((n_detectors, n_errors), dtype=np.uint8)
    priors = np.zeros(n_errors, dtype=np.float64)
    obs_flip = np.zeros(n_errors, dtype=np.uint8)

    for j, e in enumerate(errors):
        priors[j] = e["prob"]
        for d in e["detectors"]:
            H[d, j] = 1
        if e["observables"]:
            obs_flip[j] = 1

    return H, priors, obs_flip


def _build_parity_check_matrix_hyperedge(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix with proper hyperedge merging.

    Multiple error mechanisms that trigger the same detector pattern are
    merged into a single hyperedge. Probabilities are combined using the
    XOR formula (since detectors use XOR - two errors cancel out):
        p_combined = p_old + p_new - 2 * p_old * p_new

    This is the correct approach for BP decoding because:
    1. Errors with identical syndromes are indistinguishable
    2. They should be represented as a single variable in the factor graph
    3. Detectors are XOR-based: if two errors trigger same detector, they cancel

    For observable flips, we track the probability that an odd number of
    observable-flipping errors occur (also XOR-based).
    """
    n_detectors = dem.num_detectors

    # Map from detector pattern (frozenset) to hyperedge info
    # Each hyperedge stores: (combined_prob, obs_flip_prob)
    # combined_prob = P(odd number of errors fire) - XOR probability
    # obs_flip_prob = P(odd number of obs-flipping errors fire) - XOR probability
    hyperedge_map: dict[frozenset, tuple[float, float]] = {}

    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            if prob == 0:
                continue  # Skip zero-probability errors

            targets = inst.targets_copy()

            # Split by separator into independent components
            components = _split_error_by_separator(targets)

            for comp in components:
                detectors = frozenset(comp["detectors"])
                has_obs_flip = len(comp["observables"]) > 0

                if detectors in hyperedge_map:
                    # Merge with existing hyperedge using XOR probability
                    p_old, obs_prob_old = hyperedge_map[detectors]

                    # XOR probability: P(exactly one of A or B fires)
                    # p_xor = p_old * (1 - p_new) + p_new * (1 - p_old)
                    #       = p_old + p_new - 2 * p_old * p_new
                    # This is correct because detectors use XOR:
                    # if both errors fire, the detector triggers twice = no trigger
                    p_combined = p_old + prob - 2 * p_old * prob

                    # Observable flip probability (XOR):
                    # P(obs flipped) = P(odd number of obs-flipping errors fire)
                    # Use same XOR formula for errors that flip observable
                    if has_obs_flip:
                        # New error flips observable: XOR with existing flip probability
                        obs_prob_new = obs_prob_old * (1 - prob) + prob * (1 - obs_prob_old)
                    else:
                        # New error doesn't flip observable
                        # P(obs flip) = P(old flips) * P(new doesn't fire)
                        # This preserves the obs flip only when the non-flipping error doesn't fire
                        obs_prob_new = obs_prob_old * (1 - prob)

                    hyperedge_map[detectors] = (p_combined, obs_prob_new)
                else:
                    # New hyperedge
                    obs_prob = prob if has_obs_flip else 0.0
                    hyperedge_map[detectors] = (prob, obs_prob)

    # Convert hyperedge map to arrays
    n_hyperedges = len(hyperedge_map)

    H = np.zeros((n_detectors, n_hyperedges), dtype=np.uint8)
    priors = np.zeros(n_hyperedges, dtype=np.float64)
    obs_flip = np.zeros(n_hyperedges, dtype=np.float64)

    for j, (detectors, (prob, obs_prob)) in enumerate(hyperedge_map.items()):
        priors[j] = prob
        for d in detectors:
            H[d, j] = 1
        # Store conditional probability: P(obs flip | hyperedge fires)
        # For decoding, threshold at 0.5 to determine if error flips observable
        if prob > 0:
            obs_flip[j] = obs_prob / prob
        else:
            obs_flip[j] = 0.0

    # Clip to [0, 1] to handle floating point precision issues
    priors = np.clip(priors, 0.0, 1.0)
    obs_flip = np.clip(obs_flip, 0.0, 1.0)

    return H, priors, obs_flip


def dem_to_uai(dem: stim.DetectorErrorModel) -> str:
    """
    Convert DEM to UAI format for probabilistic inference.
    the .dem file is split first and then transformed into a file .uai

    Args:
        dem: Detector Error Model to convert.

    Returns:
        String in UAI format representing the factor graph.
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()

            # Split by separator into independent components
            components = _split_error_by_separator(targets)

            for comp in components:
                errors.append({"prob": prob, "detectors": comp["detectors"]})

    n_detectors = dem.num_detectors
    lines = []
    lines.append("MARKOV")
    lines.append(str(n_detectors))
    lines.append(" ".join(["2"] * n_detectors))
    lines.append(str(len(errors)))

    for e in errors:
        dets = e["detectors"]
        lines.append(f"{len(dets)} " + " ".join(map(str, dets)))

    lines.append("")
    for e in errors:
        n_dets = len(e["detectors"])
        n_entries = 2 ** n_dets
        lines.append(str(n_entries))

        p = e["prob"]
        for i in range(n_entries):
            parity = bin(i).count("1") % 2
            if parity == 0:
                lines.append(str(1 - p))
            else:
                lines.append(str(p))
        lines.append("")

    return "\n".join(lines)


def save_uai(dem: stim.DetectorErrorModel, output_path: pathlib.Path) -> None:
    """
    Save DEM as UAI format file.

    Args:
        dem: Detector Error Model to save.
        output_path: Path to save the UAI file.
    """
    output_path.write_text(dem_to_uai(dem))


def generate_dem_from_circuit(
    circuit_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    decompose_errors: bool = True,
) -> pathlib.Path:
    """
    Generate and save DEM from a circuit file.

    Args:
        circuit_path: Path to the circuit file (.stim).
        output_path: Optional output path. If None, uses datasets/dems/ directory.
        decompose_errors: Whether to decompose errors into components.

    Returns:
        Path to the saved DEM file.
    """
    circuit = stim.Circuit.from_file(str(circuit_path))

    if output_path is None:
        dems_dir = pathlib.Path("datasets")
        dems_dir.mkdir(parents=True, exist_ok=True)
        output_path = dems_dir / circuit_path.with_suffix(".dem").name

    dem = extract_dem(circuit, decompose_errors=decompose_errors)
    save_dem(dem, output_path)

    return output_path


def generate_uai_from_circuit(
    circuit_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    decompose_errors: bool = True,
) -> pathlib.Path:
    """
    Generate and save UAI format file from a circuit file.

    Args:
        circuit_path: Path to the circuit file (.stim).
        output_path: Optional output path. If None, uses datasets/uais/ directory.
        decompose_errors: Whether to decompose errors into components.

    Returns:
        Path to the saved UAI file.
    """
    circuit = stim.Circuit.from_file(str(circuit_path))

    if output_path is None:
        uais_dir = pathlib.Path("datasets")
        uais_dir.mkdir(parents=True, exist_ok=True)
        output_path = uais_dir / circuit_path.with_suffix(".uai").name

    dem = extract_dem(circuit, decompose_errors=decompose_errors)
    save_uai(dem, output_path)

    return output_path
