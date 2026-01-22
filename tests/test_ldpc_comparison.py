"""
Validation test comparing BPDecoderPlus with ldpc library.

This test verifies that our BP+OSD implementation produces similar
logical error rates to the reference ldpc library.

To run this test, first install the ldpc library:
    pip install ldpc

Then run:
    uv run pytest tests/test_ldpc_comparison.py -v -s
"""
import numpy as np
import torch
import pytest
import time

try:
    from ._path import add_project_root_to_path
except ImportError:
    from _path import add_project_root_to_path

add_project_root_to_path()

# Check if ldpc is available
try:
    from ldpc import BpOsdDecoder
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.osd import OSDDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder


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
        Tuple of (logical error rate, total time in seconds)
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
    start_time = time.perf_counter()
    for i, syndrome in enumerate(syndromes):
        result = ldpc_decoder.decode(syndrome.astype(np.uint8))
        predicted_obs = np.dot(result, obs_flip) % 2
        if predicted_obs != observables[i]:
            errors += 1
    end_time = time.perf_counter()

    return errors / len(syndromes), end_time - start_time


def run_bpdecoderplus(H, syndromes, observables, obs_flip, priors,
                      osd_order=10, max_iter=20):
    """
    Run BPDecoderPlus BP+OSD decoder (CPU).

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        obs_flip: Observable flip indicators per error
        priors: Per-qubit error probabilities
        osd_order: OSD search depth
        max_iter: Maximum BP iterations

    Returns:
        Tuple of (logical error rate, total time in seconds)
    """
    bp_decoder = BatchBPDecoder(H, priors, device='cpu')
    osd_decoder = OSDDecoder(H)

    start_time = time.perf_counter()
    batch_syndromes = torch.from_numpy(syndromes).float()
    marginals = bp_decoder.decode(batch_syndromes, max_iter=max_iter, damping=0.2)

    errors = 0
    for i in range(len(syndromes)):
        probs = marginals[i].cpu().numpy()
        result = osd_decoder.solve(syndromes[i], probs, osd_order=osd_order)
        predicted_obs = np.dot(result, obs_flip) % 2
        if predicted_obs != observables[i]:
            errors += 1
    end_time = time.perf_counter()

    return errors / len(syndromes), end_time - start_time


def run_bpdecoderplus_gpu(H, syndromes, observables, obs_flip, priors,
                          osd_order=10, max_iter=20):
    """
    Run BPDecoderPlus with GPU-accelerated OSD decoder.

    Args:
        H: Parity check matrix
        syndromes: Array of syndromes to decode
        observables: Ground truth observable values
        obs_flip: Observable flip indicators per error
        priors: Per-qubit error probabilities
        osd_order: OSD search depth
        max_iter: Maximum BP iterations

    Returns:
        Tuple of (logical error rate, total time in seconds)
    """
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    bp_decoder = BatchBPDecoder(H, priors, device=device)
    osd_decoder = BatchOSDDecoder(H, device=device)

    start_time = time.perf_counter()
    batch_syndromes = torch.from_numpy(syndromes).float().to(device)
    marginals = bp_decoder.decode(batch_syndromes, max_iter=max_iter, damping=0.2)

    errors = 0
    for i in range(len(syndromes)):
        probs = marginals[i].cpu().numpy()
        result = osd_decoder.solve(syndromes[i], probs, osd_order=osd_order)
        predicted_obs = np.dot(result, obs_flip) % 2
        if predicted_obs != observables[i]:
            errors += 1
    end_time = time.perf_counter()

    return errors / len(syndromes), end_time - start_time


@pytest.mark.skipif(not LDPC_AVAILABLE, reason="ldpc library not installed")
class TestLdpcComparison:
    """Test class comparing BPDecoderPlus with ldpc library."""

    @pytest.fixture
    def load_data(self):
        """Load test dataset."""
        dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
        syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
        H, priors, obs_flip = build_parity_check_matrix(dem)
        return H, syndromes, observables, priors, obs_flip

    def test_logical_error_rate_comparison(self, load_data):
        """Compare logical error rates between ldpc and BPDecoderPlus."""
        H, syndromes, observables, priors, obs_flip = load_data
        num_samples = 200  # Use subset for faster testing

        # Run both decoders
        ldpc_ler, ldpc_time = run_ldpc_decoder(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip,
            error_rate=0.01, osd_order=10, max_iter=20
        )
        bp_ler, bp_time = run_bpdecoderplus(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=10, max_iter=20
        )

        print(f"\n{'='*60}")
        print(f"Logical Error Rate Comparison ({num_samples} samples)")
        print(f"{'='*60}")
        print(f"ldpc (C++):           LER={ldpc_ler:.2%}, Time={ldpc_time:.3f}s ({num_samples/ldpc_time:.1f} samples/s)")
        print(f"BPDecoderPlus (CPU):  LER={bp_ler:.2%}, Time={bp_time:.3f}s ({num_samples/bp_time:.1f} samples/s)")
        print(f"Speedup: {bp_time/ldpc_time:.2f}x slower")
        print(f"{'='*60}")

        # Both should be effective (better than baseline ~17.6%)
        assert ldpc_ler < 0.15, f"ldpc LER too high: {ldpc_ler}"
        assert bp_ler < 0.15, f"BPDecoderPlus LER too high: {bp_ler}"

        # Results should be similar (within 5% absolute difference)
        assert abs(ldpc_ler - bp_ler) < 0.05, \
            f"LER difference too large: ldpc={ldpc_ler:.2%}, BP={bp_ler:.2%}"

    def test_syndrome_satisfaction(self, load_data):
        """Verify both decoders produce valid codewords (satisfy syndrome)."""
        H, syndromes, observables, priors, obs_flip = load_data
        num_samples = 50  # Smaller subset for this test

        # Initialize decoders
        ldpc_decoder = BpOsdDecoder(
            H.astype(np.uint8),
            error_rate=0.01,
            max_iter=20,
            bp_method='product_sum',
            osd_method='osd_e',
            osd_order=10
        )
        bp_decoder = BatchBPDecoder(H, priors, device='cpu')
        osd_decoder = OSDDecoder(H)

        batch_syndromes = torch.from_numpy(syndromes[:num_samples]).float()
        marginals = bp_decoder.decode(batch_syndromes, max_iter=20, damping=0.2)

        ldpc_valid = 0
        bp_valid = 0

        for i in range(num_samples):
            syndrome = syndromes[i]

            # ldpc decoder
            ldpc_result = ldpc_decoder.decode(syndrome.astype(np.uint8))
            ldpc_syndrome = (H @ ldpc_result) % 2
            if np.array_equal(ldpc_syndrome, syndrome):
                ldpc_valid += 1

            # BPDecoderPlus
            probs = marginals[i].cpu().numpy()
            bp_result = osd_decoder.solve(syndrome, probs, osd_order=10)
            bp_syndrome = (H @ bp_result) % 2
            if np.array_equal(bp_syndrome, syndrome):
                bp_valid += 1

        print(f"\nldpc valid codewords: {ldpc_valid}/{num_samples} ({ldpc_valid/num_samples:.1%})")
        print(f"BPDecoderPlus valid codewords: {bp_valid}/{num_samples} ({bp_valid/num_samples:.1%})")

        # Both should produce valid codewords most of the time
        assert ldpc_valid / num_samples >= 0.95, \
            f"ldpc syndrome satisfaction too low: {ldpc_valid/num_samples:.1%}"
        assert bp_valid / num_samples >= 0.95, \
            f"BPDecoderPlus syndrome satisfaction too low: {bp_valid/num_samples:.1%}"


@pytest.mark.skipif(not LDPC_AVAILABLE, reason="ldpc library not installed")
class TestLdpcComparisonExtended:
    """Extended comparison tests with more samples."""

    @pytest.fixture
    def load_data(self):
        """Load test dataset."""
        dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
        syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
        H, priors, obs_flip = build_parity_check_matrix(dem)
        return H, syndromes, observables, priors, obs_flip

    def test_full_dataset_comparison(self, load_data):
        """Compare on full dataset (500+ samples) for more accurate comparison."""
        H, syndromes, observables, priors, obs_flip = load_data
        num_samples = min(500, len(syndromes))

        ldpc_ler, ldpc_time = run_ldpc_decoder(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip,
            error_rate=0.01, osd_order=10, max_iter=20
        )
        bp_ler, bp_time = run_bpdecoderplus(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=10, max_iter=20
        )

        print(f"\nFull dataset comparison ({num_samples} samples):")
        print(f"ldpc LER: {ldpc_ler:.2%}, Time: {ldpc_time:.3f}s")
        print(f"BPDecoderPlus LER: {bp_ler:.2%}, Time: {bp_time:.3f}s")
        print(f"Difference: {abs(ldpc_ler - bp_ler):.2%}")

        # Both should be effective
        assert ldpc_ler < 0.15, f"ldpc LER too high: {ldpc_ler}"
        assert bp_ler < 0.15, f"BPDecoderPlus LER too high: {bp_ler}"


@pytest.mark.skipif(not LDPC_AVAILABLE, reason="ldpc library not installed")
class TestGPUAcceleration:
    """Test GPU-accelerated OSD decoder performance."""

    @pytest.fixture
    def load_data(self):
        """Load test dataset."""
        dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
        syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
        H, priors, obs_flip = build_parity_check_matrix(dem)
        return H, syndromes, observables, priors, obs_flip

    def test_gpu_osd_correctness(self, load_data):
        """Verify GPU OSD produces same results as CPU OSD."""
        H, syndromes, observables, priors, obs_flip = load_data
        num_samples = 50  # Small subset for correctness test

        # Run BP to get marginals
        bp_decoder = BatchBPDecoder(H, priors, device='cpu')
        batch_syndromes = torch.from_numpy(syndromes[:num_samples]).float()
        marginals = bp_decoder.decode(batch_syndromes, max_iter=20, damping=0.2)

        # Initialize both decoders
        cpu_osd = OSDDecoder(H)
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        gpu_osd = BatchOSDDecoder(H, device=device)

        matches = 0
        for i in range(num_samples):
            probs = marginals[i].cpu().numpy()
            cpu_result = cpu_osd.solve(syndromes[i], probs, osd_order=10, random_seed=42)
            gpu_result = gpu_osd.solve(syndromes[i], probs, osd_order=10, random_seed=42)

            if np.array_equal(cpu_result, gpu_result):
                matches += 1

        print(f"\nGPU OSD correctness: {matches}/{num_samples} ({matches/num_samples:.1%}) match CPU")

        # Allow some tolerance due to floating point differences
        assert matches / num_samples >= 0.90, \
            f"GPU OSD results differ too much from CPU: {matches}/{num_samples}"

    def test_gpu_timing_comparison(self, load_data):
        """Compare timing between ldpc, CPU OSD, and GPU OSD."""
        H, syndromes, observables, priors, obs_flip = load_data
        num_samples = 200

        # Run all three decoders
        ldpc_ler, ldpc_time = run_ldpc_decoder(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip,
            error_rate=0.01, osd_order=10, max_iter=20
        )
        cpu_ler, cpu_time = run_bpdecoderplus(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=10, max_iter=20
        )
        gpu_ler, gpu_time = run_bpdecoderplus_gpu(
            H, syndromes[:num_samples],
            observables[:num_samples], obs_flip, priors,
            osd_order=10, max_iter=20
        )

        print(f"\n{'='*70}")
        print(f"Performance Comparison ({num_samples} samples, OSD-10)")
        print(f"{'='*70}")
        print(f"{'Decoder':<25} {'LER':>10} {'Time (s)':>12} {'Throughput':>15}")
        print(f"{'-'*70}")
        print(f"{'ldpc (C++)':<25} {ldpc_ler:>9.2%} {ldpc_time:>12.3f} {num_samples/ldpc_time:>12.1f} syn/s")
        print(f"{'BPDecoderPlus (CPU)':<25} {cpu_ler:>9.2%} {cpu_time:>12.3f} {num_samples/cpu_time:>12.1f} syn/s")
        device_name = 'GPU' if CUDA_AVAILABLE else 'CPU (no CUDA)'
        print(f"{'BPDecoderPlus (' + device_name + ')':<25} {gpu_ler:>9.2%} {gpu_time:>12.3f} {num_samples/gpu_time:>12.1f} syn/s")
        print(f"{'-'*70}")
        print(f"CPU vs ldpc: {cpu_time/ldpc_time:.1f}x slower")
        print(f"GPU vs CPU:  {cpu_time/gpu_time:.2f}x speedup")
        print(f"GPU vs ldpc: {gpu_time/ldpc_time:.1f}x slower")
        print(f"{'='*70}")

        # All should produce similar LER
        assert abs(ldpc_ler - gpu_ler) < 0.05, \
            f"GPU LER differs too much from ldpc: {gpu_ler:.2%} vs {ldpc_ler:.2%}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_high_osd_order(self, load_data):
        """Test GPU acceleration benefit with higher OSD order."""
        H, syndromes, observables, priors, obs_flip = load_data
        num_samples = 50  # Fewer samples for higher OSD order

        for osd_order in [10, 12, 15]:
            cpu_ler, cpu_time = run_bpdecoderplus(
                H, syndromes[:num_samples],
                observables[:num_samples], obs_flip, priors,
                osd_order=osd_order, max_iter=20
            )
            gpu_ler, gpu_time = run_bpdecoderplus_gpu(
                H, syndromes[:num_samples],
                observables[:num_samples], obs_flip, priors,
                osd_order=osd_order, max_iter=20
            )

            speedup = cpu_time / gpu_time
            print(f"\nOSD-{osd_order}: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, Speedup={speedup:.2f}x")
            print(f"  CPU LER: {cpu_ler:.2%}, GPU LER: {gpu_ler:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
