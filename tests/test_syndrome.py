"""Tests for syndrome database generation module."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import stim

from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.syndrome import (
    generate_syndrome_database_from_circuit,
    load_syndrome_database,
    sample_syndromes,
    save_syndrome_database,
)


class TestSampleSyndromes:
    """Tests for sample_syndromes function."""

    def test_basic_sampling(self):
        """Test basic syndrome sampling."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        syndromes, observables = sample_syndromes(circuit, num_shots=10)

        assert syndromes.shape[0] == 10
        assert observables.shape == (10,)
        assert syndromes.dtype in (np.uint8, np.bool_)
        assert observables.dtype in (np.uint8, np.bool_)

    def test_without_observables(self):
        """Test sampling without observables."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        syndromes, observables = sample_syndromes(
            circuit, num_shots=10, include_observables=False
        )

        assert syndromes.shape[0] == 10
        assert observables is None

    def test_num_detectors(self):
        """Test that number of detectors matches circuit."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = circuit.detector_error_model()
        syndromes, _ = sample_syndromes(circuit, num_shots=5)

        assert syndromes.shape[1] == dem.num_detectors


class TestSaveSyndromeDatabase:
    """Tests for save_syndrome_database function."""

    def test_save_with_observables(self):
        """Test saving database with observables."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)
        observables = np.random.randint(0, 2, size=10, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            save_syndrome_database(syndromes, observables, output_path)

            assert output_path.exists()

    def test_save_without_observables(self):
        """Test saving database without observables."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            save_syndrome_database(syndromes, None, output_path)

            assert output_path.exists()

    def test_save_with_metadata(self):
        """Test saving database with metadata."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)
        observables = np.random.randint(0, 2, size=10, dtype=np.uint8)
        metadata = {"distance": 3, "rounds": 3, "p": 0.01}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            save_syndrome_database(syndromes, observables, output_path, metadata)

            assert output_path.exists()


class TestLoadSyndromeDatabase:
    """Tests for load_syndrome_database function."""

    def test_load_with_observables(self):
        """Test loading database with observables."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)
        observables = np.random.randint(0, 2, size=10, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            save_syndrome_database(syndromes, observables, output_path)

            loaded_syndromes, loaded_observables, _ = load_syndrome_database(output_path)

            np.testing.assert_array_equal(loaded_syndromes, syndromes)
            np.testing.assert_array_equal(loaded_observables, observables)

    def test_load_without_observables(self):
        """Test loading database without observables."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            save_syndrome_database(syndromes, None, output_path)

            loaded_syndromes, loaded_observables, _ = load_syndrome_database(output_path)

            np.testing.assert_array_equal(loaded_syndromes, syndromes)
            assert loaded_observables is None

    def test_load_with_metadata(self):
        """Test loading database with metadata."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)
        observables = np.random.randint(0, 2, size=10, dtype=np.uint8)
        metadata = {"distance": 3, "rounds": 3, "p": 0.01}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            save_syndrome_database(syndromes, observables, output_path, metadata)

            _, _, loaded_metadata = load_syndrome_database(output_path)

            assert loaded_metadata == metadata

    def test_load_metadata_0dim_array(self):
        """Test loading metadata stored as 0-dimensional array."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)
        observables = np.random.randint(0, 2, size=10, dtype=np.uint8)
        metadata = {"test": "value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            # Save with 0-dim array (using np.array(json_str))
            import json
            np.savez(
                output_path,
                syndromes=syndromes,
                observables=observables,
                metadata=np.array(json.dumps(metadata))  # 0-dim array
            )

            _, _, loaded_metadata = load_syndrome_database(output_path)
            assert loaded_metadata == metadata

    def test_load_metadata_dict_directly(self):
        """Test loading metadata stored as pickled dict."""
        syndromes = np.random.randint(0, 2, size=(10, 24), dtype=np.uint8)
        observables = np.random.randint(0, 2, size=10, dtype=np.uint8)
        metadata = {"test": "value", "number": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.npz"
            # Save with allow_pickle and dict directly
            np.savez(
                output_path,
                syndromes=syndromes,
                observables=observables,
                metadata=np.array([metadata], dtype=object)  # 1-dim with dict
            )

            _, _, loaded_metadata = load_syndrome_database(output_path)
            assert loaded_metadata == metadata


class TestGenerateSyndromeDatabaseFromCircuit:
    """Tests for generate_syndrome_database_from_circuit function."""

    def test_generate_from_circuit_file(self):
        """Test generating database from circuit file."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            db_path = generate_syndrome_database_from_circuit(circuit_path, num_shots=20)

            assert db_path.exists()
            assert db_path.suffix == ".npz"

            # Load and verify
            syndromes, observables, metadata = load_syndrome_database(db_path)
            assert syndromes.shape[0] == 20
            assert observables.shape == (20,)
            assert metadata["num_shots"] == 20
            assert metadata["circuit_file"] == "test.stim"

    def test_custom_output_path(self):
        """Test generating database with custom output path."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

        with tempfile.TemporaryDirectory() as tmpdir:
            circuit_path = pathlib.Path(tmpdir) / "test.stim"
            circuit_path.write_text(str(circuit))

            custom_output = pathlib.Path(tmpdir) / "custom_db.npz"
            db_path = generate_syndrome_database_from_circuit(
                circuit_path, num_shots=15, output_path=custom_output
            )

            assert db_path == custom_output
            assert db_path.exists()
