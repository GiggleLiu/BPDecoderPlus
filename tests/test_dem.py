"""Tests for DEM extraction module."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import stim

from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.dem import (
    _split_error_by_separator,
    build_parity_check_matrix,
    dem_to_dict,
    dem_to_uai,
    extract_dem,
    generate_dem_from_circuit,
    generate_uai_from_circuit,
    load_dem,
    save_dem,
    save_dem_json,
    save_uai,
)


class TestExtractDem:
    """Tests for extract_dem function."""

    def test_basic_extraction(self):
        """Test basic DEM extraction."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        assert isinstance(dem, stim.DetectorErrorModel)
        assert dem.num_detectors > 0
        assert dem.num_observables == 1

    def test_decompose_errors(self):
        """Test DEM extraction with error decomposition."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit, decompose_errors=True)

        assert isinstance(dem, stim.DetectorErrorModel)

    def test_no_decompose(self):
        """Test DEM extraction without error decomposition."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit, decompose_errors=False)

        assert isinstance(dem, stim.DetectorErrorModel)


class TestSaveDem:
    """Tests for save_dem function."""

    def test_save_dem(self):
        """Test saving DEM to file."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.dem"
            save_dem(dem, output_path)

            assert output_path.exists()
            assert output_path.read_text().startswith("error")


class TestLoadDem:
    """Tests for load_dem function."""

    def test_load_dem(self):
        """Test loading DEM from file."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.dem"
            save_dem(dem, output_path)

            loaded_dem = load_dem(output_path)

            assert loaded_dem.num_detectors == dem.num_detectors
            assert loaded_dem.num_observables == dem.num_observables


class TestDemToDict:
    """Tests for dem_to_dict function."""

    def test_basic_conversion(self):
        """Test converting DEM to dictionary."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        dem_dict = dem_to_dict(dem)

        assert "num_detectors" in dem_dict
        assert "num_observables" in dem_dict
        assert "num_errors" in dem_dict
        assert "errors" in dem_dict
        assert dem_dict["num_detectors"] == dem.num_detectors
        assert dem_dict["num_observables"] == dem.num_observables

    def test_error_structure(self):
        """Test error structure in dictionary."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        dem_dict = dem_to_dict(dem)

        assert len(dem_dict["errors"]) > 0
        first_error = dem_dict["errors"][0]
        assert "probability" in first_error
        assert "detectors" in first_error
        assert "observables" in first_error


class TestSaveDemJson:
    """Tests for save_dem_json function."""

    def test_save_json(self):
        """Test saving DEM as JSON."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.json"
            save_dem_json(dem, output_path)

            assert output_path.exists()

            import json
            with open(output_path) as f:
                data = json.load(f)

            assert "num_detectors" in data
            assert "errors" in data


class TestBuildParityCheckMatrix:
    """Tests for build_parity_check_matrix function."""

    def test_basic_matrix(self):
        """Test building parity check matrix."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        H, priors, obs_flip = build_parity_check_matrix(dem)

        assert H.shape[0] == dem.num_detectors
        assert H.shape[1] > 0
        assert priors.shape[0] == H.shape[1]
        assert obs_flip.shape[0] == H.shape[1]

    def test_matrix_types(self):
        """Test matrix data types."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        H, priors, obs_flip = build_parity_check_matrix(dem)

        assert H.dtype == np.uint8
        assert priors.dtype == np.float64
        assert obs_flip.dtype == np.uint8

    def test_matrix_values(self):
        """Test matrix value ranges."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        H, priors, obs_flip = build_parity_check_matrix(dem)

        assert np.all((H == 0) | (H == 1))
        assert np.all((priors >= 0) & (priors <= 1))
        assert np.all((obs_flip == 0) | (obs_flip == 1))


class TestGenerateDemFromCircuit:
    """Tests for generate_dem_from_circuit function."""

    def test_generate_from_file(self):
        """Test generating DEM from circuit file."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            dem_path = generate_dem_from_circuit(circuit_path)

            assert dem_path.exists()
            assert dem_path.suffix == ".dem"

            # Load and verify
            loaded_dem = load_dem(dem_path)
            assert loaded_dem.num_detectors > 0

    def test_custom_output_path(self):
        """Test generating DEM with custom output path."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            custom_output = pathlib.Path(tmpdir) / "custom.dem"
            dem_path = generate_dem_from_circuit(circuit_path, output_path=custom_output)

            assert dem_path == custom_output
            assert dem_path.exists()

    def test_no_decompose(self):
        """Test generating DEM without error decomposition."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            dem_path = generate_dem_from_circuit(circuit_path, decompose_errors=False)

            assert dem_path.exists()


class TestDemToUai:
    """Tests for dem_to_uai function."""

    def test_basic_conversion(self):
        """Test basic DEM to UAI conversion."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)
        uai_str = dem_to_uai(dem)

        assert isinstance(uai_str, str)
        assert "MARKOV" in uai_str
        assert str(dem.num_detectors) in uai_str

    def test_uai_format_structure(self):
        """Test UAI format has correct structure."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)
        uai_str = dem_to_uai(dem)

        lines = uai_str.strip().split("\n")
        assert lines[0] == "MARKOV"
        assert int(lines[1]) == dem.num_detectors
        assert len(lines[2].split()) == dem.num_detectors


class TestSaveUai:
    """Tests for save_uai function."""

    def test_save_uai(self):
        """Test saving UAI file."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        with tempfile.TemporaryDirectory() as tmpdir:
            uai_path = pathlib.Path(tmpdir) / "test.uai"
            save_uai(dem, uai_path)

            assert uai_path.exists()
            content = uai_path.read_text()
            assert "MARKOV" in content


class TestGenerateUaiFromCircuit:
    """Tests for generate_uai_from_circuit function."""

    def test_generate_from_file(self):
        """Test generating UAI from circuit file."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            uai_path = generate_uai_from_circuit(circuit_path)

            assert uai_path.exists()
            assert uai_path.suffix == ".uai"

    def test_custom_output_path(self):
        """Test generating UAI with custom output path."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            custom_output = pathlib.Path(tmpdir) / "custom.uai"
            uai_path = generate_uai_from_circuit(circuit_path, output_path=custom_output)

            assert uai_path == custom_output
            assert uai_path.exists()


class TestSplitErrorBySeparator:
    """Tests for ^ separator handling in DEM parsing.

    CRITICAL: These tests verify that error instructions with ^ separators
    are correctly split into independent components. This is required for
    correct BP decoding - without it, the parity check matrix has wrong
    structure and threshold analysis produces invalid results.
    """

    def test_no_separator(self):
        """Test targets without ^ separator return single component."""
        # Simulate targets for "D0 D1 L0"
        dem = stim.DetectorErrorModel("error(0.01) D0 D1 L0")
        inst = list(dem.flattened())[0]
        targets = inst.targets_copy()

        components = _split_error_by_separator(targets)

        assert len(components) == 1
        assert components[0]["detectors"] == [0, 1]
        assert components[0]["observables"] == [0]

    def test_single_separator(self):
        """Test targets with one ^ separator split into two components."""
        # Simulate targets for "D0 D1 ^ D2"
        dem = stim.DetectorErrorModel("error(0.01) D0 D1 ^ D2")
        inst = list(dem.flattened())[0]
        targets = inst.targets_copy()

        components = _split_error_by_separator(targets)

        assert len(components) == 2
        assert components[0]["detectors"] == [0, 1]
        assert components[0]["observables"] == []
        assert components[1]["detectors"] == [2]
        assert components[1]["observables"] == []

    def test_multiple_separators(self):
        """Test targets with multiple ^ separators split correctly."""
        # Simulate targets for "D0 ^ D1 ^ D2 L0"
        dem = stim.DetectorErrorModel("error(0.01) D0 ^ D1 ^ D2 L0")
        inst = list(dem.flattened())[0]
        targets = inst.targets_copy()

        components = _split_error_by_separator(targets)

        assert len(components) == 3
        assert components[0]["detectors"] == [0]
        assert components[1]["detectors"] == [1]
        assert components[2]["detectors"] == [2]
        assert components[2]["observables"] == [0]

    def test_observable_in_first_component(self):
        """Test observable correctly assigned to first component."""
        dem = stim.DetectorErrorModel("error(0.01) D0 L0 ^ D1")
        inst = list(dem.flattened())[0]
        targets = inst.targets_copy()

        components = _split_error_by_separator(targets)

        assert len(components) == 2
        assert components[0]["detectors"] == [0]
        assert components[0]["observables"] == [0]
        assert components[1]["detectors"] == [1]
        assert components[1]["observables"] == []


class TestBuildParityCheckMatrixSeparator:
    """Test that build_parity_check_matrix handles ^ separators correctly."""

    def test_separator_creates_multiple_columns(self):
        """Verify ^ separator creates multiple columns in H matrix.

        CRITICAL: This test catches the bug where ^ separators are ignored,
        causing wrong H matrix structure and invalid decoding results.
        """
        # Create DEM with ^ separator
        dem = stim.DetectorErrorModel("""
            error(0.01) D0 D1 ^ D2
            error(0.02) D1
        """)

        H, priors, obs_flip = build_parity_check_matrix(dem, split_by_separator=True)

        # Should have 3 columns: 2 from first error (split by ^), 1 from second
        assert H.shape[1] == 3, (
            f"Expected 3 columns (2 from 'D0 D1 ^ D2', 1 from 'D1'), got {H.shape[1]}. "
            "This indicates ^ separator is not being handled correctly."
        )

        # First column: D0, D1
        assert H[0, 0] == 1 and H[1, 0] == 1 and H[2, 0] == 0
        # Second column: D2
        assert H[0, 1] == 0 and H[1, 1] == 0 and H[2, 1] == 1
        # Third column: D1
        assert H[0, 2] == 0 and H[1, 2] == 1 and H[2, 2] == 0

        # Both columns from first error share same probability
        assert priors[0] == 0.01
        assert priors[1] == 0.01
        assert priors[2] == 0.02

    def test_no_split_option(self):
        """Test split_by_separator=False keeps all targets in one column."""
        dem = stim.DetectorErrorModel("""
            error(0.01) D0 D1 ^ D2
            error(0.02) D1
        """)

        H, priors, obs_flip = build_parity_check_matrix(dem, split_by_separator=False)

        # Should have 2 columns: 1 from first error (no split), 1 from second
        assert H.shape[1] == 2, (
            f"Expected 2 columns with split_by_separator=False, got {H.shape[1]}"
        )

        # First column: D0, D1, D2 all together
        assert H[0, 0] == 1 and H[1, 0] == 1 and H[2, 0] == 1
        # Second column: D1
        assert H[0, 1] == 0 and H[1, 1] == 1 and H[2, 1] == 0

    def test_real_dem_has_separators(self):
        """Verify real surface code DEM contains ^ separators that must be handled."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        # Count error instructions with ^ separators
        separator_count = 0
        for inst in dem.flattened():
            if inst.type == "error":
                targets = inst.targets_copy()
                if any(t.is_separator() for t in targets):
                    separator_count += 1

        # Real surface code DEMs contain many ^ separators
        assert separator_count > 0, (
            "Expected real DEM to contain ^ separators. "
            "If this fails, the test DEM generation may have changed."
        )

        # Verify H matrix is built correctly
        H, priors, obs_flip = build_parity_check_matrix(dem)

        # Number of columns should exceed number of error instructions
        # because separators split instructions into multiple columns
        n_instructions = sum(1 for inst in dem.flattened() if inst.type == "error")
        assert H.shape[1] >= n_instructions, (
            f"H has {H.shape[1]} columns but DEM has {n_instructions} error instructions. "
            "With ^ separators, we expect more columns than instructions."
        )
