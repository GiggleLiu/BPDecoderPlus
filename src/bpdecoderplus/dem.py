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
    merge_hyperedges: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix H from DEM for BP decoding.

    CRITICAL: This function performs two important steps for correct BP decoding:

    1. **Separator splitting** (split_by_separator=True, default):
       DEM error instructions like "error(0.01) D0 D1 ^ D2" are split by the ^
       separator into separate components. This is required because ^ indicates
       correlated faults that trigger multiple detector patterns simultaneously.
       Reference: PyMatching (https://github.com/oscarhiggott/PyMatching) uses
       the same approach when parsing DEM files.

    2. **Hyperedge merging** (merge_hyperedges=True, default):
       After separator splitting, errors with IDENTICAL detector patterns are
       merged into single "hyperedges" using XOR probability combination:
           p_combined = p_old + p_new - 2 * p_old * p_new
       This is the correct approach because:
       - Errors with identical syndromes are indistinguishable to the decoder
       - Detectors are XOR-based: two errors triggering the same detector cancel
       - This reduces the factor graph size and improves threshold performance
       Reference: PyMatching merges errors after parsing DEM files.

    DO NOT REMOVE the merge_hyperedges functionality without understanding its
    impact on threshold performance. See Issue #61 and PR #62 for context.

    Args:
        dem: Detector Error Model.
        split_by_separator: If True (default), split error targets by ^ separator.
            Required for correct BP decoding on circuit-level noise models.
        merge_hyperedges: If True (default), merge errors with identical detector
            patterns into single hyperedges with XOR-combined probabilities.
            This improves threshold performance. If False, keep separate columns.

    Returns:
        Tuple of (H, priors, obs_flip) where:
        - H: Parity check matrix, shape (num_detectors, num_hyperedges)
        - priors: Prior error probabilities, shape (num_hyperedges,)
        - obs_flip: Observable flip probabilities, shape (num_hyperedges,)
            When merge_hyperedges=True, contains P(obs flip | hyperedge fires).
            When merge_hyperedges=False, binary (0 or 1).
    """
    if merge_hyperedges:
        return _build_parity_check_matrix_hyperedge(dem, split_by_separator)
    else:
        return _build_parity_check_matrix_simple(dem, split_by_separator)


def _build_parity_check_matrix_simple(
    dem: stim.DetectorErrorModel,
    split_by_separator: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix without hyperedge merging.

    Creates one column per error component. This is simpler but may have
    worse threshold performance than the hyperedge-merged version.

    Args:
        dem: Detector Error Model.
        split_by_separator: If True, split error targets by ^ separator.

    Returns:
        Tuple of (H, priors, obs_flip) with binary obs_flip values.
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()

            if split_by_separator:
                for comp in _split_error_by_separator(targets):
                    errors.append({
                        "prob": prob,
                        "detectors": comp["detectors"],
                        "observables": comp["observables"],
                    })
            else:
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


def _build_parity_check_matrix_hyperedge(
    dem: stim.DetectorErrorModel,
    split_by_separator: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix with hyperedge merging for optimal threshold.

    CRITICAL: DO NOT REMOVE THIS FUNCTION. It is required for optimal threshold
    performance. See Issue #61 and PR #62 for the history of why this exists.

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

    Reference: PyMatching (https://github.com/oscarhiggott/PyMatching) uses
    a similar approach, merging errors after parsing DEM files to build the
    decoding graph.

    Args:
        dem: Detector Error Model.
        split_by_separator: If True, split error targets by ^ separator first.

    Returns:
        Tuple of (H, priors, obs_flip) where obs_flip contains soft probabilities.
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
            if split_by_separator:
                components = _split_error_by_separator(targets)
            else:
                detectors = [t.val for t in targets if t.is_relative_detector_id()]
                observables = [t.val for t in targets if t.is_logical_observable_id()]
                components = [{"detectors": detectors, "observables": observables}]

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


def build_parity_factor(n_vars: int, syndrome_bit: int) -> list[float]:
    """
    Build XOR parity constraint factor for MAP decoding.

    Creates a factor that equals 1.0 when the parity of the input variables
    matches the syndrome bit, and 1e-30 (near-zero) otherwise.

    The near-zero value (instead of 0) avoids log(0) = -inf issues in
    the tropical tensor network computation.

    Args:
        n_vars: Number of binary variables in this factor.
        syndrome_bit: Target parity (0 or 1) from the syndrome.

    Returns:
        List of factor values with 2^n_vars entries.
    """
    n_entries = 2 ** n_vars
    values = []
    for i in range(n_entries):
        parity = bin(i).count("1") % 2
        # Factor = 1.0 if parity matches syndrome, else near-zero
        if parity == syndrome_bit:
            values.append(1.0)
        else:
            values.append(1e-30)
    return values


def dem_to_uai_for_decoding(
    dem: stim.DetectorErrorModel,
    syndrome: np.ndarray,
) -> str:
    """
    Convert DEM + syndrome to UAI model for MAP error decoding.

    Unlike dem_to_uai (where variables = detectors), this creates a model where:
    - Variables = error/hyperedge bits (after separator split + hyperedge merge)
    - Prior factors = error probabilities for each hyperedge
    - Constraint factors = syndrome parity checks (hard constraints)

    The resulting UAI model can be solved with tropical tensor network MPE
    inference to find the most likely error pattern given the syndrome.

    Args:
        dem: Detector Error Model.
        syndrome: Binary syndrome array of shape (num_detectors,).

    Returns:
        String in UAI format representing the decoding factor graph.
    """
    # Build parity check matrix with separator splitting and hyperedge merging
    H, priors, obs_flip = build_parity_check_matrix(dem)

    n_detectors, n_errors = H.shape

    # Verify syndrome length
    if len(syndrome) != n_detectors:
        raise ValueError(
            f"Syndrome length {len(syndrome)} does not match "
            f"number of detectors {n_detectors}"
        )

    lines = []

    # UAI header: MARKOV network type
    lines.append("MARKOV")

    # Number of variables = number of hyperedges (errors)
    lines.append(str(n_errors))

    # All variables are binary
    lines.append(" ".join(["2"] * n_errors))

    # Count factors: n_errors prior factors + n_detectors constraint factors
    n_factors = n_errors + n_detectors
    lines.append(str(n_factors))

    # Factor scopes
    # First: prior factors (each covers one error variable)
    for i in range(n_errors):
        lines.append(f"1 {i}")

    # Second: constraint factors (each covers errors connected to a detector)
    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        n_vars = len(error_indices)
        if n_vars > 0:
            scope_str = " ".join(str(e) for e in error_indices)
            lines.append(f"{n_vars} {scope_str}")
        else:
            # Detector with no connected errors - should not happen in valid DEM
            # but handle gracefully with empty scope
            lines.append("0")

    lines.append("")

    # Factor values
    # Prior factors: P(e_i = 0) = 1 - p_i, P(e_i = 1) = p_i
    for i in range(n_errors):
        p = priors[i]
        lines.append("2")
        lines.append(str(1.0 - p))
        lines.append(str(p))
        lines.append("")

    # Constraint factors: XOR parity check
    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        n_vars = len(error_indices)
        if n_vars > 0:
            syndrome_bit = int(syndrome[d])
            factor_values = build_parity_factor(n_vars, syndrome_bit)
            lines.append(str(len(factor_values)))
            for v in factor_values:
                lines.append(str(v))
            lines.append("")
        else:
            # Empty factor for detector with no errors
            lines.append("1")
            lines.append("1.0")
            lines.append("")

    return "\n".join(lines)


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
