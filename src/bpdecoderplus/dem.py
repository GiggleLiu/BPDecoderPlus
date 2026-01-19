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
            detectors = [t.val for t in targets if t.is_relative_detector_id()]
            observables = [t.val for t in targets if t.is_logical_observable_id()]

            errors.append({
                "probability": float(prob),
                "detectors": detectors,
                "observables": observables,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix H from DEM for BP decoding.

    Args:
        dem: Detector Error Model.

    Returns:
        Tuple of (H, priors, obs_flip) where:
        - H: Parity check matrix, shape (num_detectors, num_errors)
        - priors: Prior error probabilities, shape (num_errors,)
        - obs_flip: Observable flip indicators, shape (num_errors,)
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()
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
        dems_dir = pathlib.Path("datasets/dems")
        dems_dir.mkdir(parents=True, exist_ok=True)
        output_path = dems_dir / circuit_path.with_suffix(".dem").name

    dem = extract_dem(circuit, decompose_errors=decompose_errors)
    save_dem(dem, output_path)

    return output_path
