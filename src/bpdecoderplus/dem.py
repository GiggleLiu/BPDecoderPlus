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

    CRITICAL: This function is required for correct BP+OSD decoding.

    In DEM format, error(p) D0 D1 ^ D2 means a correlated fault that triggers
    {D0, D1} AND {D2} simultaneously with probability p. These must be treated
    as SEPARATE columns in the parity check matrix H, each with the same
    probability p. Without this splitting:
    - The parity check matrix H has wrong structure
    - BP marginals are computed incorrectly
    - Threshold analysis produces invalid results

    Reference: PyMatching (https://github.com/oscarhiggott/PyMatching) uses
    the same approach when parsing DEM files.

    Args:
        targets: List of stim.DemTarget from an error instruction.

    Returns:
        List of component dicts, each with 'detectors' and 'observables' lists.
        Returns a list with one element if no separators are present.

    Example:
        For targets representing "D0 D1 ^ D2 L0":
        Returns [{"detectors": [0, 1], "observables": []},
                 {"detectors": [2], "observables": [0]}]
    """
    components = []
    current_detectors = []
    current_observables = []

    for t in targets:
        if t.is_separator():
            # ^ separator found - finalize current component
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

    # Don't forget the last component after final separator (or if no separator)
    if current_detectors or current_observables:
        components.append({
            "detectors": current_detectors,
            "observables": current_observables,
        })

    return components


def dem_to_dict(dem: stim.DetectorErrorModel) -> dict[str, Any]:
    """
    Convert DEM to dictionary with structured information.

    Handles ^ separators by splitting each error instruction into
    separate components (see _split_error_by_separator).

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

            # Split by ^ separator - each component becomes a separate error
            for comp in _split_error_by_separator(targets):
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
    split_by_separator: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix H from DEM for BP decoding.

    CRITICAL: By default, this function handles ^ separators in DEM error
    instructions. Each component (separated by ^) becomes a SEPARATE column
    in H with the same probability. This is required for correct BP decoding.

    Example: error(0.01) D0 D1 ^ D2 creates TWO columns (when split_by_separator=True):
    - Column 1: prob=0.01, detectors={0,1}, obs_flip=0
    - Column 2: prob=0.01, detectors={2}, obs_flip=0

    Args:
        dem: Detector Error Model.
        split_by_separator: If True (default), split error targets by ^ separator
            into separate columns. If False, treat all targets in one error
            instruction as a single column. Default True is required for correct
            BP decoding on circuit-level noise models.

    Returns:
        Tuple of (H, priors, obs_flip) where:
        - H: Parity check matrix, shape (num_detectors, num_errors)
        - priors: Prior error probabilities, shape (num_errors,)
        - obs_flip: Observable flip indicators, shape (num_errors,) binary (0 or 1)
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()

            if split_by_separator:
                # CRITICAL: Split by ^ separator - each component is a separate error
                # Without this, the parity check matrix has wrong structure and
                # BP decoding produces incorrect results.
                for comp in _split_error_by_separator(targets):
                    errors.append({
                        "prob": prob,
                        "detectors": comp["detectors"],
                        "observables": comp["observables"],
                    })
            else:
                # No splitting - treat all targets as single error (legacy behavior)
                detectors = [t.val for t in targets if t.is_relative_detector_id()]
                observables = [t.val for t in targets if t.is_logical_observable_id()]
                errors.append({
                    "prob": prob,
                    "detectors": detectors,
                    "observables": observables,
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


def dem_to_uai(dem: stim.DetectorErrorModel) -> str:
    """
    Convert DEM to UAI format for probabilistic inference.

    Handles ^ separators by splitting each error into separate factors.

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

            # Split by ^ separator - each component becomes a separate factor
            for comp in _split_error_by_separator(targets):
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
