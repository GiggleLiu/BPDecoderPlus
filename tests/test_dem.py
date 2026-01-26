"""Tests for DEM extraction module."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import stim
import torch

from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.dem import (
    _split_error_by_separator,
    build_decoding_uai,
    build_parity_check_matrix,
    extract_dem,
    generate_dem_from_circuit,
    load_dem,
    save_dem,
)
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder


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

        # Default: hyperedge merging produces float64 obs_flip (soft probabilities)
        H, priors, obs_flip = build_parity_check_matrix(dem)
        assert H.dtype == np.uint8
        assert priors.dtype == np.float64
        assert obs_flip.dtype == np.float64

        # Without hyperedge merging: binary obs_flip (uint8)
        H2, priors2, obs_flip2 = build_parity_check_matrix(dem, merge_hyperedges=False)
        assert H2.dtype == np.uint8
        assert priors2.dtype == np.float64
        assert obs_flip2.dtype == np.uint8

    def test_matrix_values(self):
        """Test matrix value ranges."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)

        # Default: hyperedge merging produces soft obs_flip in [0, 1]
        H, priors, obs_flip = build_parity_check_matrix(dem)
        assert np.all((H == 0) | (H == 1))
        assert np.all((priors >= 0) & (priors <= 1))
        assert np.all((obs_flip >= 0) & (obs_flip <= 1))

        # Without hyperedge merging: binary obs_flip
        H2, priors2, obs_flip2 = build_parity_check_matrix(dem, merge_hyperedges=False)
        assert np.all((H2 == 0) | (H2 == 1))
        assert np.all((priors2 >= 0) & (priors2 <= 1))
        assert np.all((obs_flip2 == 0) | (obs_flip2 == 1))


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

        # Verify H matrix is built correctly without hyperedge merging
        # (to test that separator splitting works)
        H_split, priors_split, obs_flip_split = build_parity_check_matrix(
            dem, merge_hyperedges=False
        )

        # Number of columns should exceed number of error instructions
        # because separators split instructions into multiple columns
        n_instructions = sum(1 for inst in dem.flattened() if inst.type == "error")
        assert H_split.shape[1] >= n_instructions, (
            f"H has {H_split.shape[1]} columns but DEM has {n_instructions} error instructions. "
            "With ^ separators, we expect more columns than instructions."
        )

        # With hyperedge merging (default), we get FEWER columns because
        # identical detector patterns are merged
        H_merged, priors_merged, obs_flip_merged = build_parity_check_matrix(dem)
        assert H_merged.shape[1] < H_split.shape[1], (
            f"Expected hyperedge merging to reduce columns: merged={H_merged.shape[1]} "
            f"vs split={H_split.shape[1]}"
        )


class TestHyperedgeMerging:
    """Tests for hyperedge merging functionality.

    CRITICAL: These tests protect the hyperedge merging code from being
    accidentally removed. See Issue #61 and PR #62 for context.
    """

    def test_xor_probability_combination(self):
        """Test that probabilities are combined using XOR formula."""
        # Create a simple DEM with two errors triggering the same detector
        dem_str = """
        error(0.1) D0
        error(0.2) D0
        """
        dem = stim.DetectorErrorModel(dem_str)

        H, priors, obs_flip = build_parity_check_matrix(dem)

        # Should have only 1 column (merged)
        assert H.shape[1] == 1

        # XOR probability: p_combined = p1 + p2 - 2*p1*p2
        # = 0.1 + 0.2 - 2*0.1*0.2 = 0.3 - 0.04 = 0.26
        expected_prob = 0.1 + 0.2 - 2 * 0.1 * 0.2
        assert np.isclose(priors[0], expected_prob), f"Expected {expected_prob}, got {priors[0]}"

    def test_no_merge_keeps_separate_columns(self):
        """Test that merge_hyperedges=False keeps columns separate."""
        dem_str = """
        error(0.1) D0
        error(0.2) D0
        """
        dem = stim.DetectorErrorModel(dem_str)

        H, priors, obs_flip = build_parity_check_matrix(dem, merge_hyperedges=False)

        # Should have 2 columns (not merged)
        assert H.shape[1] == 2
        assert np.isclose(priors[0], 0.1)
        assert np.isclose(priors[1], 0.2)

    def test_different_detector_patterns_not_merged(self):
        """Test that errors with different detector patterns are not merged."""
        dem_str = """
        error(0.1) D0
        error(0.2) D1
        error(0.3) D0 D1
        """
        dem = stim.DetectorErrorModel(dem_str)

        H, priors, obs_flip = build_parity_check_matrix(dem)

        # All three have different patterns, so 3 columns
        assert H.shape[1] == 3

    def test_observable_flip_probability(self):
        """Test that observable flip probability is computed correctly."""
        # Two errors with same detector: one flips observable, one doesn't
        dem_str = """
        error(0.1) D0 L0
        error(0.2) D0
        """
        dem = stim.DetectorErrorModel(dem_str)

        H, priors, obs_flip = build_parity_check_matrix(dem)

        # Should have 1 merged column
        assert H.shape[1] == 1

        # Combined prob = 0.1 + 0.2 - 2*0.1*0.2 = 0.26
        # Obs flip prob = P(error1 fires XOR error2 fires | either fires) when error1 flips obs
        # = P(only error1 fires) / P(either fires)
        # = 0.1 * (1 - 0.2) / 0.26 = 0.08 / 0.26 ≈ 0.3077
        # Actually the formula is: obs_prob = p_old_obs * (1 - p_new) if new doesn't flip
        # obs_prob_after = 0.1 * (1 - 0.2) = 0.08
        # Then obs_flip[j] = obs_prob / combined_prob = 0.08 / 0.26
        expected_obs_flip = (0.1 * (1 - 0.2)) / (0.1 + 0.2 - 2 * 0.1 * 0.2)
        assert np.isclose(obs_flip[0], expected_obs_flip, atol=0.01), (
            f"Expected obs_flip ≈ {expected_obs_flip}, got {obs_flip[0]}"
        )

    def test_merge_with_separator_splitting(self):
        """Test that separator splitting happens before hyperedge merging."""
        # Two error instructions with separators that produce same detector pattern
        dem_str = """
        error(0.1) D0 ^ D1
        error(0.2) D0 ^ D1
        """
        dem = stim.DetectorErrorModel(dem_str)

        # Without merging: 4 columns (2 instructions × 2 components each)
        H_no_merge, _, _ = build_parity_check_matrix(dem, merge_hyperedges=False)
        assert H_no_merge.shape[1] == 4

        # With merging: 2 columns (D0 errors merged, D1 errors merged)
        H_merged, _, _ = build_parity_check_matrix(dem, merge_hyperedges=True)
        assert H_merged.shape[1] == 2


class TestMergedMatrixModeThreshold:
    """Test that merged matrix mode (split_by_separator=True, merge_hyperedges=True)
    produces correct decoding behavior: logical error rate decreases with distance
    at low physical error rates.
    
    This validates the correctness of the matrix construction mode changes.
    """

    def _compute_observable_prediction_soft(self, solution: np.ndarray, obs_flip: np.ndarray) -> int:
        """
        Compute observable prediction using soft XOR probability chain.
        
        When hyperedges are merged, obs_flip stores conditional probabilities.
        This function computes P(odd number of observable flips).
        """
        p_flip = 0.0
        for i in range(len(solution)):
            if solution[i] == 1:
                p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
        return int(p_flip > 0.5)

    def _run_bposd_decoder(self, H, syndromes, observables, obs_flip, priors,
                           osd_order=10, max_iter=30):
        """Run BP+OSD decoder and return logical error rate."""
        bp_decoder = BatchBPDecoder(H, priors, device='cpu')
        osd_decoder = BatchOSDDecoder(H, device='cpu')

        batch_syndromes = torch.from_numpy(syndromes).float()
        marginals = bp_decoder.decode(batch_syndromes, max_iter=max_iter, damping=0.2)

        errors = 0
        for i in range(len(syndromes)):
            probs = marginals[i].cpu().numpy()
            solution = osd_decoder.solve(syndromes[i], probs, osd_order=osd_order)
            predicted_obs = self._compute_observable_prediction_soft(solution, obs_flip)
            if predicted_obs != observables[i]:
                errors += 1

        return errors / len(syndromes)

    @pytest.mark.slow
    def test_ler_decreases_with_distance_merged_mode(self):
        """
        Test that logical error rate decreases with code distance below threshold.
        
        The circuit-level depolarizing noise threshold for rotated surface code 
        is ~0.7% (0.007). At p=0.005 (below threshold), we expect the logical 
        error rate to decrease as code distance increases from d=3 to d=5.
        
        This test uses the "merged" matrix mode:
        - split_by_separator=True: correctly handles ^ separators in DEM
        - merge_hyperedges=True: merges identical detector patterns for efficiency
        """
        p = 0.005  # Below threshold (~0.007), LER should decrease with distance
        num_shots = 1000  # Number of syndrome samples
        
        lers = {}
        
        for distance in [3, 5]:
            rounds = distance  # Standard: rounds = distance
            
            # Generate circuit and DEM
            circuit = generate_circuit(distance=distance, rounds=rounds, p=p, task="z")
            dem = extract_dem(circuit)
            
            # Build parity check matrix with merged mode
            H, priors, obs_flip = build_parity_check_matrix(
                dem, 
                split_by_separator=True, 
                merge_hyperedges=True
            )
            
            # Sample syndromes
            sampler = circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(
                num_shots, separate_observables=True
            )
            syndromes = detection_events.astype(np.uint8)
            observables = observable_flips.flatten().astype(np.uint8)
            
            # Run decoder
            ler = self._run_bposd_decoder(
                H, syndromes, observables, obs_flip, priors,
                osd_order=10, max_iter=30
            )
            lers[distance] = ler
            
            print(f"\nd={distance}: H shape={H.shape}, LER={ler:.4f} ({num_shots} shots)")
        
        # Verify LER decreases with distance below threshold
        print(f"\nLER comparison at p={p}: d=3: {lers[3]:.4f}, d=5: {lers[5]:.4f}")
        
        # The key assertion: at p < threshold, larger distance should have lower LER
        # Allow small tolerance for statistical fluctuations
        assert lers[5] <= lers[3] + 0.02, (
            f"Expected LER to decrease with distance at p={p} (below threshold 0.007), "
            f"but got d=3: {lers[3]:.4f}, d=5: {lers[5]:.4f}. "
            f"This may indicate a problem with matrix construction mode."
        )


class TestBuildDecodingUAI:
    """Tests for build_decoding_uai function.
    
    This function builds a UAI factor graph for MAP decoding from a parity
    check matrix, priors, and syndrome.
    """

    def test_basic_uai_structure(self):
        """Test that UAI output has correct structure."""
        # Simple 2x3 matrix: 2 detectors, 3 errors
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        priors = np.array([0.1, 0.2, 0.15])
        syndrome = np.array([1, 0], dtype=np.uint8)

        uai_str = build_decoding_uai(H, priors, syndrome)

        lines = uai_str.strip().split("\n")
        
        # Check header
        assert lines[0] == "MARKOV"
        assert lines[1] == "3"  # 3 variables (errors)
        assert lines[2] == "2 2 2"  # All binary
        assert lines[3] == "5"  # 3 prior factors + 2 constraint factors

    def test_prior_factors(self):
        """Test that prior factors have correct values."""
        H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        priors = np.array([0.1, 0.3])
        syndrome = np.array([0, 0], dtype=np.uint8)

        uai_str = build_decoding_uai(H, priors, syndrome)
        lines = uai_str.strip().split("\n")

        # Find factor values section (after scopes)
        # Structure: header (4 lines) + scopes (4 lines: 2 prior + 2 constraint) + blank + values
        # Prior factor 0: should have values [1-0.1, 0.1] = [0.9, 0.1]
        # Prior factor 1: should have values [1-0.3, 0.3] = [0.7, 0.3]
        
        # Find "2" entries for prior factors
        idx = 0
        for i, line in enumerate(lines):
            if line == "" and idx == 0:
                idx = i + 1
                break
        
        # First prior factor
        assert lines[idx] == "2"
        assert float(lines[idx + 1]) == pytest.approx(0.9)
        assert float(lines[idx + 2]) == pytest.approx(0.1)

    def test_constraint_factors_syndrome_zero(self):
        """Test constraint factors when syndrome is 0 (even parity required)."""
        H = np.array([[1, 1]], dtype=np.uint8)  # 1 detector, 2 errors
        priors = np.array([0.1, 0.1])
        syndrome = np.array([0], dtype=np.uint8)  # Even parity required

        uai_str = build_decoding_uai(H, priors, syndrome)

        # Constraint factor for detector 0 should have:
        # - 00 (parity 0) -> 1.0
        # - 01 (parity 1) -> 1e-30
        # - 10 (parity 1) -> 1e-30
        # - 11 (parity 0) -> 1.0
        assert "1.0" in uai_str
        assert "1e-30" in uai_str

    def test_constraint_factors_syndrome_one(self):
        """Test constraint factors when syndrome is 1 (odd parity required)."""
        H = np.array([[1, 1]], dtype=np.uint8)  # 1 detector, 2 errors
        priors = np.array([0.1, 0.1])
        syndrome = np.array([1], dtype=np.uint8)  # Odd parity required

        uai_str = build_decoding_uai(H, priors, syndrome)

        # Constraint factor should enforce odd parity
        # - 00 (parity 0) -> 1e-30
        # - 01 (parity 1) -> 1.0
        # - 10 (parity 1) -> 1.0
        # - 11 (parity 0) -> 1e-30
        assert "1.0" in uai_str
        assert "1e-30" in uai_str

    def test_empty_detector(self):
        """Test handling of detectors with no connected errors."""
        # Detector 1 has no connected errors
        H = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        priors = np.array([0.1, 0.1])
        
        # Empty detector with syndrome 0 should be satisfiable
        syndrome_zero = np.array([0, 0], dtype=np.uint8)
        uai_str_zero = build_decoding_uai(H, priors, syndrome_zero)
        assert "1" in uai_str_zero  # Factor with single entry
        
        # Empty detector with syndrome 1 should be unsatisfiable
        syndrome_one = np.array([0, 1], dtype=np.uint8)
        uai_str_one = build_decoding_uai(H, priors, syndrome_one)
        # Both should produce valid UAI format
        assert uai_str_one.startswith("MARKOV")

    def test_real_surface_code(self):
        """Test build_decoding_uai with real surface code data."""
        circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")
        dem = extract_dem(circuit)
        H, priors, obs_flip = build_parity_check_matrix(dem)

        # Sample a syndrome
        sampler = circuit.compile_detector_sampler()
        samples = sampler.sample(1, append_observables=True)
        syndrome = samples[0, :-1].astype(np.uint8)

        uai_str = build_decoding_uai(H, priors, syndrome)

        # Verify structure
        lines = uai_str.strip().split("\n")
        assert lines[0] == "MARKOV"
        
        n_errors = H.shape[1]
        n_detectors = H.shape[0]
        assert lines[1] == str(n_errors)
        
        # Number of factors = n_errors (priors) + n_detectors (constraints)
        n_factors = n_errors + n_detectors
        assert lines[3] == str(n_factors)
