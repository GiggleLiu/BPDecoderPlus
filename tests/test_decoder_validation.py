"""
Decoder validation tests for BP+OSD decoder.

This module contains pytest tests for validating the BP+OSD decoder
effectiveness and comparing with the ldpc library.

Usage:
    uv run pytest tests/test_decoder_validation.py -v -s
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import pytest

from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder

# Check if ldpc is available
try:
    from ldpc import BpOsdDecoder
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False


def run_ldpc_decoder(H, syndromes, observables, obs_flip, error_rate=0.01,
                     osd_order=10, max_iter=20):
    """
    Run ldpc library BP+OSD decoder.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        obs_flip: Observable flip indicators per error
        error_rate: Physical error rate for BP
        osd_order: OSD search depth
        max_iter: Maximum BP iterations

    Returns:
        Logical error rate
    """
    ldpc_decoder = BpOsdDecoder(
        H.astype(np.uint8),
        error_rate=error_rate,
        max_iter=max_iter,
        bp_method='product_sum',
        osd_method='osd_e',
        osd_order=osd_order
    )

    errors = 0
    for i, syndrome in enumerate(syndromes):
        result = ldpc_decoder.decode(syndrome.astype(np.uint8))
        predicted_obs = np.dot(result, obs_flip) % 2
        if predicted_obs != observables[i]:
            errors += 1

    return errors / len(syndromes)


def run_bpdecoderplus(H, syndromes, observables, obs_flip, priors,
                      osd_order=10, max_iter=20):
    """
    Run BPDecoderPlus BP+OSD decoder.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        obs_flip: Observable flip indicators per error
        priors: Per-qubit error probabilities
        osd_order: OSD search depth
        max_iter: Maximum BP iterations

    Returns:
        Logical error rate
    """
    bp_decoder = BatchBPDecoder(H, priors, device='cpu')
    osd_decoder = BatchOSDDecoder(H, device='cpu')

    batch_syndromes = torch.from_numpy(syndromes).float()
    marginals = bp_decoder.decode(batch_syndromes, max_iter=max_iter, damping=0.2)

    errors = 0
    for i in range(len(syndromes)):
        probs = marginals[i].cpu().numpy()
        result = osd_decoder.solve(syndromes[i], probs, osd_order=osd_order)
        predicted_obs = np.dot(result, obs_flip) % 2
        if predicted_obs != observables[i]:
            errors += 1

    return errors / len(syndromes)


class TestDecoderValidation:
    """Validation tests for d=3 dataset."""

    @pytest.fixture
    def load_d3_data(self):
        """Load d=3 dataset."""
        dem = load_dem('datasets/sc_d3_r3_p0100_z.dem')
        syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0100_z.npz')
        H, priors, obs_flip = build_parity_check_matrix(dem)
        return H, syndromes, observables, priors, obs_flip

    def test_bp_only_effectiveness(self, load_d3_data):
        """Test BP-only decoder is effective."""
        H, syndromes, observables, priors, obs_flip = load_d3_data
        num_samples = 500

        bp_decoder = BatchBPDecoder(H, priors, device='cpu')
        batch_syndromes = torch.from_numpy(syndromes[:num_samples]).float()
        marginals = bp_decoder.decode(batch_syndromes, max_iter=20, damping=0.2)

        errors = 0
        for i in range(num_samples):
            probs = marginals[i].cpu().numpy()
            pred = np.dot((probs > 0.5).astype(int), obs_flip) % 2
            if pred != observables[i]:
                errors += 1

        ler_bp = errors / num_samples
        baseline_ler = np.mean(observables[:num_samples])

        print(f"\nBP-only LER: {ler_bp:.2%}, Baseline: {baseline_ler:.2%}")
        assert ler_bp <= baseline_ler, f"BP-only is making things worse: {ler_bp:.2%} > {baseline_ler:.2%}"

    def test_bp_osd10_effectiveness(self, load_d3_data):
        """Test BP+OSD-10 decoder is effective."""
        H, syndromes, observables, priors, obs_flip = load_d3_data
        num_samples = 500

        ler = run_bpdecoderplus(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=10
        )
        baseline_ler = np.mean(observables[:num_samples])

        print(f"\nBP+OSD-10 LER: {ler:.2%}, Baseline: {baseline_ler:.2%}")
        assert ler <= baseline_ler, f"BP+OSD-10 is making things worse: {ler:.2%} > {baseline_ler:.2%}"

    def test_bp_osd15_effectiveness(self, load_d3_data):
        """Test BP+OSD-15 decoder is effective."""
        H, syndromes, observables, priors, obs_flip = load_d3_data
        num_samples = 500

        ler = run_bpdecoderplus(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=15
        )
        baseline_ler = np.mean(observables[:num_samples])

        print(f"\nBP+OSD-15 LER: {ler:.2%}, Baseline: {baseline_ler:.2%}")
        assert ler <= baseline_ler, f"BP+OSD-15 is making things worse: {ler:.2%} > {baseline_ler:.2%}"


@pytest.mark.skipif(not LDPC_AVAILABLE, reason="ldpc library not installed")
class TestLdpcValidation:
    """Validate against ldpc library."""

    @pytest.fixture
    def load_d3_data(self):
        """Load d=3 dataset."""
        dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
        syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
        H, priors, obs_flip = build_parity_check_matrix(dem)
        return H, syndromes, observables, priors, obs_flip

    def test_ldpc_comparison(self, load_d3_data):
        """Compare BPDecoderPlus with ldpc library."""
        H, syndromes, observables, priors, obs_flip = load_d3_data
        num_samples = 200

        ldpc_ler = run_ldpc_decoder(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip,
            error_rate=0.01, osd_order=10
        )
        bp_ler = run_bpdecoderplus(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=10
        )

        print(f"\n{'='*60}")
        print(f"Logical Error Rate Comparison ({num_samples} samples)")
        print(f"{'='*60}")
        print(f"ldpc (C++):           LER={ldpc_ler:.2%}")
        print(f"BPDecoderPlus (CPU):  LER={bp_ler:.2%}")
        print(f"{'='*60}")

        # Both should be effective (better than baseline ~17.6%)
        assert ldpc_ler < 0.15, f"ldpc LER too high: {ldpc_ler}"
        assert bp_ler < 0.15, f"BPDecoderPlus LER too high: {bp_ler}"

        # Results should be similar (within 5% absolute difference)
        assert abs(ldpc_ler - bp_ler) < 0.05, \
            f"LER difference too large: ldpc={ldpc_ler:.2%}, BP={bp_ler:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
