"""Tests for circuit generation module."""

from __future__ import annotations

import pathlib
import tempfile

import pytest
import stim

from bpdecoderplus.circuit import (
    generate_circuit,
    generate_filename,
    parse_rounds,
    prob_tag,
    run_smoke_test,
    write_circuit,
)


class TestParseRounds:
    """Tests for parse_rounds function."""

    def test_basic_list(self):
        """Test parsing a basic list of rounds."""
        result = parse_rounds([3, 5, 7])
        assert result == [3, 5, 7]

    def test_unsorted_input(self):
        """Test that output is sorted."""
        result = parse_rounds([7, 3, 5])
        assert result == [3, 5, 7]

    def test_duplicates_removed(self):
        """Test that duplicates are removed."""
        result = parse_rounds([3, 5, 3, 5, 7])
        assert result == [3, 5, 7]

    def test_negative_values_filtered(self):
        """Test that non-positive values are filtered out."""
        result = parse_rounds([3, -1, 5, 0, 7])
        assert result == [3, 5, 7]

    def test_empty_after_filter_raises(self):
        """Test that ValueError is raised when no valid rounds remain."""
        with pytest.raises(ValueError, match="At least one positive"):
            parse_rounds([-1, 0])

    def test_empty_input_raises(self):
        """Test that ValueError is raised for empty input."""
        with pytest.raises(ValueError, match="At least one positive"):
            parse_rounds([])


class TestProbTag:
    """Tests for prob_tag function."""

    def test_p001(self):
        """Test conversion of p=0.01."""
        assert prob_tag(0.01) == "p0100"

    def test_p0001(self):
        """Test conversion of p=0.001."""
        assert prob_tag(0.001) == "p0010"

    def test_p005(self):
        """Test conversion of p=0.05."""
        assert prob_tag(0.05) == "p0500"

    def test_p01(self):
        """Test conversion of p=0.1."""
        assert prob_tag(0.1) == "p1000"

    def test_p0005(self):
        """Test conversion of p=0.005."""
        assert prob_tag(0.005) == "p0050"


class TestGenerateCircuit:
    """Tests for generate_circuit function."""

    def test_basic_generation(self):
        """Test basic circuit generation."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        assert isinstance(circuit, stim.Circuit)

    def test_circuit_has_detectors(self):
        """Test that circuit contains detectors."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = circuit.detector_error_model()
        assert dem.num_detectors > 0

    def test_circuit_has_observable(self):
        """Test that circuit has a logical observable."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = circuit.detector_error_model()
        assert dem.num_observables == 1

    def test_task_x(self):
        """Test X-memory task generation."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="x")
        assert isinstance(circuit, stim.Circuit)

    def test_invalid_task_raises(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be"):
            generate_circuit(distance=3, rounds=3, p=0.01, task="y")

    def test_invalid_p_raises(self):
        """Test that invalid p raises ValueError."""
        with pytest.raises(ValueError, match="p must be in"):
            generate_circuit(distance=3, rounds=3, p=0.0, task="z")
        with pytest.raises(ValueError, match="p must be in"):
            generate_circuit(distance=3, rounds=3, p=1.0, task="z")
        with pytest.raises(ValueError, match="p must be in"):
            generate_circuit(distance=3, rounds=3, p=-0.1, task="z")

    def test_different_distances(self):
        """Test circuit generation with different distances."""
        for d in [3, 5, 7]:
            circuit = generate_circuit(distance=d, rounds=3, p=0.01, task="z")
            assert isinstance(circuit, stim.Circuit)

    def test_different_rounds(self):
        """Test circuit generation with different rounds."""
        for r in [1, 3, 5, 10]:
            circuit = generate_circuit(distance=3, rounds=r, p=0.01, task="z")
            assert isinstance(circuit, stim.Circuit)


class TestRunSmokeTest:
    """Tests for run_smoke_test function."""

    def test_valid_circuit_passes(self):
        """Test that valid circuit passes smoke test."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        # Should not raise
        run_smoke_test(circuit, shots=2)

    def test_custom_shots(self):
        """Test smoke test with custom shot count."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        # Should not raise
        run_smoke_test(circuit, shots=10)


class TestWriteCircuit:
    """Tests for write_circuit function."""

    def test_write_and_read(self):
        """Test writing circuit and reading it back."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "test.stim"
            write_circuit(circuit, path)

            # Read back and verify
            assert path.exists()
            loaded = stim.Circuit.from_file(str(path))
            assert str(circuit) == str(loaded)


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        filename = generate_filename(distance=3, rounds=5, p=0.01, task="z")
        assert filename == "sc_d3_r5_p0010_z.stim"

    def test_x_task(self):
        """Test filename with x task."""
        filename = generate_filename(distance=5, rounds=7, p=0.001, task="x")
        assert filename == "sc_d5_r7_p0001_x.stim"
