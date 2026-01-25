"""Tests for Tropical TN matching MWPM decoder behavior.

These tests verify that the Tropical TN MAP decoder produces results
consistent with pymatching's MWPM decoder on surface codes.
This was the main fix for Issue #68.
"""

import sys
from pathlib import Path

# Add src to path for bpdecoderplus imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pytest
import stim
import torch

from bpdecoderplus.dem import build_parity_check_matrix

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

from tropical_in_new.src import mpe_tropical
from tropical_in_new.src.utils import read_model_from_string


def build_uai(H, priors, syndrome):
    """Build UAI model from parity check matrix."""
    n_detectors, n_errors = H.shape
    lines = []
    lines.append("MARKOV")
    lines.append(str(n_errors))
    lines.append(" ".join(["2"] * n_errors))

    n_factors = n_errors + n_detectors
    lines.append(str(n_factors))

    for i in range(n_errors):
        lines.append(f"1 {i}")

    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        if len(error_indices) > 0:
            lines.append(f"{len(error_indices)} " + " ".join(str(e) for e in error_indices))
        else:
            lines.append("0")

    lines.append("")

    for i in range(n_errors):
        p = priors[i]
        lines.append("2")
        lines.append(str(1.0 - p))
        lines.append(str(p))
        lines.append("")

    for d in range(n_detectors):
        error_indices = np.where(H[d, :] == 1)[0]
        if len(error_indices) > 0:
            syndrome_bit = int(syndrome[d])
            n_entries = 2**len(error_indices)
            lines.append(str(n_entries))
            for i in range(n_entries):
                parity = bin(i).count("1") % 2
                if parity == syndrome_bit:
                    lines.append("1.0")
                else:
                    lines.append("1e-30")
            lines.append("")
        else:
            lines.append("1")
            if syndrome[d] == 0:
                lines.append("1.0")
            else:
                lines.append("1e-30")
            lines.append("")

    return "\n".join(lines)


class TestTropicalParityConstraint:
    """Test that Tropical TN correctly handles parity constraints."""

    def test_simple_parity_constraint_2var(self):
        """2 variables, parity = 1 (odd parity required)."""
        uai_str = """MARKOV
2
2 2
3
1 0
1 1
2 0 1

2
0.9
0.1

2
0.9
0.1

4
1e-30
1.0
1.0
1e-30
"""
        model = read_model_from_string(uai_str)
        assignment, score, info = mpe_tropical(model)

        x0 = assignment.get(1, 0)
        x1 = assignment.get(2, 0)
        parity = (x0 + x1) % 2

        assert parity == 1, f"Expected odd parity, got x0={x0}, x1={x1}"

    def test_simple_parity_constraint_3var(self):
        """3 variables, parity = 1 (odd parity required)."""
        uai_str = """MARKOV
3
2 2 2
4
1 0
1 1
1 2
3 0 1 2

2
0.99
0.01

2
0.99
0.01

2
0.99
0.01

8
1e-30
1.0
1.0
1e-30
1.0
1e-30
1e-30
1.0
"""
        model = read_model_from_string(uai_str)
        assignment, score, info = mpe_tropical(model)

        x0 = assignment.get(1, 0)
        x1 = assignment.get(2, 0)
        x2 = assignment.get(3, 0)
        parity = (x0 + x1 + x2) % 2

        assert parity == 1, f"Expected odd parity, got x0={x0}, x1={x1}, x2={x2}"
        # Should fire exactly 1 error (minimum weight)
        assert x0 + x1 + x2 == 1, "Should fire exactly 1 error"


@pytest.mark.skipif(not HAS_PYMATCHING, reason="pymatching not installed")
class TestTropicalMatchesMWPM:
    """Test that Tropical TN matches MWPM on surface code decoding."""

    def test_surface_code_d3_agreement(self):
        """Tropical TN should agree with MWPM on d=3 surface code."""
        distance = 3
        error_rate = 0.01

        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_z',
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=error_rate,
        )
        dem = circuit.detector_error_model(decompose_errors=True)

        # Use bpdecoderplus.dem with merge_hyperedges=True for faster computation
        # The connected components fix ensures all factors are included
        H, priors, obs_flip = build_parity_check_matrix(
            dem, split_by_separator=True, merge_hyperedges=True
        )
        
        # MWPM matcher for comparison
        matcher = pymatching.Matching.from_detector_error_model(dem)

        # Sample syndromes
        sampler = circuit.compile_detector_sampler()
        samples = sampler.sample(50, append_observables=True)
        syndromes = samples[:, :-1].astype(np.uint8)
        observables = samples[:, -1].astype(np.int32)

        # MWPM decode
        mwpm_preds = matcher.decode_batch(syndromes)
        if mwpm_preds.ndim > 1:
            mwpm_preds = mwpm_preds.flatten()

        # Tropical TN decode
        agrees = 0
        for i in range(len(syndromes)):
            syndrome = syndromes[i]
            mwpm_pred = int(mwpm_preds[i])

            uai_str = build_uai(H, priors, syndrome)
            model = read_model_from_string(uai_str)
            assignment, score, info = mpe_tropical(model)

            solution = np.zeros(H.shape[1], dtype=np.int32)
            for j in range(H.shape[1]):
                solution[j] = assignment.get(j + 1, 0)

            # Threshold obs_flip at 0.5 for soft values from hyperedge merging
            obs_flip_binary = (obs_flip > 0.5).astype(int)
            tropical_pred = int(np.dot(solution, obs_flip_binary) % 2)

            if tropical_pred == mwpm_pred:
                agrees += 1

        agreement_rate = agrees / len(syndromes)
        # Should agree on at least 90% of samples
        # (some disagreement possible due to degeneracy and different graph structures)
        assert agreement_rate >= 0.90, (
            f"Tropical TN agrees with MWPM on only {agreement_rate*100:.1f}% of samples"
        )

    def test_surface_code_single_detector_active(self):
        """Single active detector should be correctly decoded."""
        distance = 3
        error_rate = 0.01

        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_z',
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=error_rate,
        )
        dem = circuit.detector_error_model(decompose_errors=True)

        # Use bpdecoderplus.dem with merge_hyperedges=True for faster computation
        H, priors, obs_flip = build_parity_check_matrix(
            dem, split_by_separator=True, merge_hyperedges=True
        )

        # Create syndrome with only detector 0 active
        syndrome = np.zeros(H.shape[0], dtype=np.uint8)
        syndrome[0] = 1

        # Build and solve
        uai_str = build_uai(H, priors, syndrome)
        model = read_model_from_string(uai_str)
        assignment, score, info = mpe_tropical(model)

        solution = np.zeros(H.shape[1], dtype=np.int32)
        for j in range(H.shape[1]):
            solution[j] = assignment.get(j + 1, 0)

        # Verify syndrome is satisfied
        computed_syndrome = (H @ solution) % 2
        assert np.array_equal(computed_syndrome, syndrome), (
            "Tropical TN solution doesn't satisfy syndrome"
        )

    def test_all_factors_included_in_contraction(self):
        """Verify all factors are included in contraction tree."""
        distance = 3
        error_rate = 0.01

        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_z',
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=error_rate,
        )
        dem = circuit.detector_error_model(decompose_errors=True)

        # Use bpdecoderplus.dem with merge_hyperedges=True for faster computation
        H, priors, obs_flip = build_parity_check_matrix(
            dem, split_by_separator=True, merge_hyperedges=True
        )

        syndrome = np.zeros(H.shape[0], dtype=np.uint8)
        syndrome[0] = 1

        uai_str = build_uai(H, priors, syndrome)
        model = read_model_from_string(uai_str)

        from tropical_in_new.src.utils import build_tropical_factors
        from tropical_in_new.src.network import build_network
        from tropical_in_new.src.contraction import get_omeco_tree

        factors = build_tropical_factors(model, evidence={})
        nodes = build_network(factors)
        tree_dict = get_omeco_tree(nodes)

        def count_leaves(tree):
            if "tensor_index" in tree:
                return 1
            args = tree.get("args", tree.get("children", []))
            return sum(count_leaves(a) for a in args)

        num_leaves = count_leaves(tree_dict)
        assert num_leaves == len(nodes), (
            f"Tree has {num_leaves} leaves but there are {len(nodes)} nodes"
        )
