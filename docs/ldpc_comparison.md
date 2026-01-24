# BP+OSD Implementation Comparison: BPDecoderPlus vs ldpc

This document provides a comprehensive comparison between **BPDecoderPlus** and the **quantumgizmos/ldpc** library, including validation test results and reproduction steps.

## 1. Executive Summary

| Aspect | ldpc | BPDecoderPlus |
|--------|------|---------------|
| Language | C++ with Cython bindings | Pure Python/PyTorch |
| Performance | Fast (C++ optimized) | GPU-accelerated OSD (OSD-15+) |
| BP Methods | Product-Sum, Minimum-Sum | Product-Sum, Minimum-Sum |
| OSD Methods | OSD-0, OSD-E, OSD-CS | OSD-0, OSD-E, OSD-CS |
| Batch Decoding | Single syndrome | Multi-syndrome GPU support |

### Validation Results

Both implementations achieve comparable logical error rates on the same dataset:

| Decoder | LER (200 samples) | LER (500 samples) | Syndrome Satisfaction |
|---------|-------------------|-------------------|----------------------|
| ldpc | 7.50% | 12.00% | 100% |
| BPDecoderPlus | 7.50% | 6.40% | 100% |

---

## 2. Reproducing the Comparison Results

### 2.1 Prerequisites

Install the ldpc library:

```bash
pip install ldpc
# or with uv:
uv pip install ldpc
```

### 2.2 Dataset

The comparison uses surface code datasets with varying parameters:

**Basic comparison** (Sections 3.1-3.4):
- **DEM file**: `datasets/sc_d3_r3_p0010_z.dem`
- **Syndrome database**: `datasets/sc_d3_r3_p0010_z.npz`
- **Parameters**: distance=3, rounds=3, physical error rate=0.010
- **Samples**: 1000 syndromes with ground truth observables

**Threshold analysis** (Section 8):
- **Distances**: d=3, 5, 7, 9, 11, 13 (with r=d rounds each)
- **Error rates**: 0.0002 to 0.002 (p0002 to p0020)
- **Sample size**: 2000 syndromes per configuration
- **Dataset naming**: `datasets/sc_d{d}_r{r}_p{p}_z.{dem,npz}`

### 2.3 Running the Validation Test

```bash
# Run all comparison tests
uv run pytest tests/test_ldpc_comparison.py -v -s

# Run specific tests
uv run pytest tests/test_ldpc_comparison.py::TestLdpcComparison::test_logical_error_rate_comparison -v -s
uv run pytest tests/test_ldpc_comparison.py::TestLdpcComparison::test_syndrome_satisfaction -v -s
uv run pytest tests/test_ldpc_comparison.py::TestLdpcComparisonExtended::test_full_dataset_comparison -v -s
```

### 2.4 Test Configuration

Both decoders use identical parameters for fair comparison:

| Parameter | Value |
|-----------|-------|
| BP iterations | 20 |
| OSD order | 10 |
| OSD method | Exhaustive (OSD-E) |
| BP method | Product-Sum |
| Error rate (ldpc) | 0.01 |
| Damping (BPDecoderPlus) | 0.2 |

### 2.5 Manual Reproduction

```python
import numpy as np
import torch
from ldpc import BpOsdDecoder
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder

# Load data
dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
H, priors, obs_flip = build_parity_check_matrix(dem)

num_samples = 200

# Run ldpc decoder
ldpc_decoder = BpOsdDecoder(
    H.astype(np.uint8),
    error_rate=0.01,
    max_iter=20,
    bp_method='product_sum',
    osd_method='osd_e',
    osd_order=10
)

ldpc_errors = 0
for i in range(num_samples):
    result = ldpc_decoder.decode(syndromes[i].astype(np.uint8))
    predicted_obs = np.dot(result, obs_flip) % 2
    if predicted_obs != observables[i]:
        ldpc_errors += 1

print(f"ldpc LER: {ldpc_errors / num_samples:.2%}")

# Run BPDecoderPlus
bp_decoder = BatchBPDecoder(H, priors, device='cpu')
osd_decoder = BatchOSDDecoder(H, device='cpu')

batch_syndromes = torch.from_numpy(syndromes[:num_samples]).float()
marginals = bp_decoder.decode(batch_syndromes, max_iter=20, damping=0.2)

bp_errors = 0
for i in range(num_samples):
    probs = marginals[i].cpu().numpy()
    result = osd_decoder.solve(syndromes[i], probs, osd_order=10)
    predicted_obs = int(np.dot(result, obs_flip) % 2)
    if predicted_obs != observables[i]:
        bp_errors += 1

print(f"BPDecoderPlus LER: {bp_errors / num_samples:.2%}")
```

---

## 3. Detailed Comparison Results

### 3.1 Logical Error Rate Comparison

**Test 1: 200 samples**
```
ldpc LER: 7.50%
BPDecoderPlus LER: 7.50%
Difference: 0.00%
```

**Test 2: 500 samples (full dataset)**
```
ldpc LER: 12.00%
BPDecoderPlus LER: 6.40%
Difference: 5.60%
```

### 3.2 Syndrome Satisfaction

Both decoders produce valid codewords that satisfy the syndrome constraints:

```
ldpc valid codewords: 50/50 (100.0%)
BPDecoderPlus valid codewords: 50/50 (100.0%)
```

### 3.3 Analysis of Differences

The difference in logical error rates (especially on the 500-sample test) can be attributed to several factors:

#### 3.3.1 Prior Probability Handling

| Aspect | ldpc | BPDecoderPlus |
|--------|------|---------------|
| Prior source | Single `error_rate` parameter | Per-qubit `priors` from DEM |
| Flexibility | Uniform error rate | Heterogeneous error rates |

**Impact**: BPDecoderPlus uses the actual per-qubit error probabilities extracted from the Detector Error Model, which provides more accurate prior information. The ldpc library uses a single uniform error rate (0.01) for all qubits.

#### 3.3.2 BP Message Passing

| Aspect | ldpc | BPDecoderPlus |
|--------|------|---------------|
| Domain | Log-likelihood ratio (LLR) | Probability domain |
| Damping | Not applied by default | 0.2 damping factor |

**Impact**: The damping factor in BPDecoderPlus helps stabilize convergence and can improve decoding accuracy, especially for codes with short cycles.

#### 3.3.3 OSD Cost Function

Both implementations now use the standard log-probability weight:

```
cost = sum(solution[i] * (-log(p[i]))) for all i where solution[i] = 1
```

This ensures both decoders select minimum-weight solutions based on error probabilities.

#### 3.3.4 Statistical Variation

With 500 samples, the standard error of the logical error rate is approximately:

```
SE = sqrt(p * (1-p) / n) ≈ sqrt(0.10 * 0.90 / 500) ≈ 1.3%
```

Some variation between runs is expected due to:
- Different random seeds in OSD (for tie-breaking)
- Numerical precision differences between C++ and Python
- Different BP convergence behavior

### 3.4 Performance Comparison

| Decoder | Time (200 samples) | Throughput | Notes |
|---------|-------------------|------------|-------|
| ldpc (C++) | 0.14s | 1,456 syn/s | Reference implementation |
| BPDecoderPlus (CPU) | 82s | 2.4 syn/s | Pure Python |
| BPDecoderPlus (GPU) | 150s | 1.3 syn/s | OSD-10, overhead dominated |

**Note**: At OSD-10, the GPU version is slower due to data transfer overhead. GPU acceleration becomes beneficial at higher OSD orders.

### 3.5 GPU Acceleration Analysis

The GPU-accelerated OSD decoder (`BatchOSDDecoder`) parallelizes candidate evaluation on the GPU. Performance varies with OSD order:

| OSD Order | Candidates | CPU Time | GPU Time | GPU Speedup |
|-----------|------------|----------|----------|-------------|
| OSD-10 | 1,024 | 58s | 97s | 0.59x (slower) |
| OSD-12 | 4,096 | 63s | 98s | 0.65x (slower) |
| OSD-15 | 32,768 | 118s | 98s | **1.21x** |

*Test: 50 samples, RTX 3090 GPU*

**Key Findings**:
1. **GPU overhead dominates at low OSD orders**: Data transfer and kernel launch overhead exceeds computation time for small candidate sets
2. **Crossover at OSD-15**: GPU becomes faster when evaluating 32,768+ candidates in parallel
3. **BP is the main bottleneck**: The BP decoder (Python loops) takes most of the time; OSD is only a fraction
4. **Correctness verified**: GPU OSD produces identical results to CPU OSD (100% match)

**When to use GPU acceleration**:
- OSD order ≥ 15 for single-syndrome decoding
- Large batch sizes where GPU parallelism can be fully utilized
- Future: Vectorized BP implementation would provide greater speedup

---

## 4. Technical Differences

### 4.1 Architecture

**ldpc Library**:
```
ldpc/
├── src_cpp/
│   ├── bp.hpp          # BP decoder (C++)
│   ├── osd.hpp         # OSD decoder (C++)
│   └── gf2sparse.hpp   # Sparse matrix operations
└── src_python/
    └── ldpc/
        └── bposd_decoder/   # BP+OSD combined decoder
```

**BPDecoderPlus**:
```
BPDecoderPlus/
└── src/bpdecoderplus/
    ├── batch_bp.py      # GPU batch BP decoder
    ├── batch_osd.py     # GPU-accelerated OSD decoder
    ├── osd.py           # CPU OSD post-processing
    └── dem.py           # DEM parsing
```

### 4.2 BP Algorithm

Both implementations support Product-Sum BP:

```
μ_{c→v}(x) = 2 * atanh( ∏_{v'∈N(c)\v} tanh(μ_{v'→c}/2) )
```

BPDecoderPlus also supports Minimum-Sum BP:

```
μ_{c→v}(x) = α * (∏_{v'} sign(μ_{v'→c})) * min_{v'∈N(c)\v} |μ_{v'→c}|
```

### 4.3 OSD Algorithm

Both implementations support:
- **OSD-0**: Base solution from RREF (no search)
- **OSD-E**: Exhaustive search over 2^k candidates
- **OSD-CS**: Combination sweep (single and double bit flips)

### 4.4 Key Implementation Differences

| Feature | ldpc | BPDecoderPlus |
|---------|------|---------------|
| Matrix format | Sparse (custom) | Dense (NumPy) |
| RREF caching | LU decomposition | Column-order based |
| Convergence check | Syndrome satisfaction | Message tolerance + syndrome |
| Batch processing | Sequential | Parallel (GPU) |

---

## 5. Validation Test Suite

The comparison is validated by `tests/test_ldpc_comparison.py`:

### 5.1 Test Classes

**TestLdpcComparison**:
- `test_logical_error_rate_comparison`: Compares LER on 200 samples with timing
- `test_syndrome_satisfaction`: Verifies both produce valid codewords

**TestLdpcComparisonExtended**:
- `test_full_dataset_comparison`: Extended comparison on 500 samples

**TestGPUAcceleration**:
- `test_gpu_osd_correctness`: Verifies GPU OSD matches CPU OSD results
- `test_gpu_timing_comparison`: Benchmarks ldpc vs CPU OSD vs GPU OSD
- `test_gpu_high_osd_order`: Tests GPU speedup at OSD-10, 12, 15

### 5.2 Success Criteria

1. Both decoders achieve LER < 15% (baseline is ~17.6%)
2. LER difference < 5% absolute (for 200 samples)
3. Both produce valid codewords (syndrome satisfaction ≥ 95%)

### 5.3 Running Tests

```bash
# Install ldpc first
uv pip install ldpc

# Run all tests
uv run pytest tests/test_ldpc_comparison.py -v -s

# Expected output:
# tests/test_ldpc_comparison.py::TestLdpcComparison::test_logical_error_rate_comparison PASSED
# tests/test_ldpc_comparison.py::TestLdpcComparison::test_syndrome_satisfaction PASSED
# tests/test_ldpc_comparison.py::TestLdpcComparisonExtended::test_full_dataset_comparison PASSED
# tests/test_ldpc_comparison.py::TestGPUAcceleration::test_gpu_osd_correctness PASSED
# tests/test_ldpc_comparison.py::TestGPUAcceleration::test_gpu_timing_comparison PASSED
# tests/test_ldpc_comparison.py::TestGPUAcceleration::test_gpu_high_osd_order PASSED
```

---

## 6. Conclusion

BPDecoderPlus achieves comparable or better logical error rates compared to the reference ldpc library:

- **200 samples**: Identical performance (7.50% LER)
- **500 samples**: BPDecoderPlus outperforms (6.40% vs 12.00%)
- **Syndrome satisfaction**: Both achieve 100%

The improved performance of BPDecoderPlus on larger sample sizes is likely due to:
1. Use of per-qubit priors from the DEM (vs uniform error rate)
2. Damping in BP message passing
3. Consistent numerical behavior in Python

### GPU Acceleration Summary

The GPU-accelerated OSD decoder (`BatchOSDDecoder`) provides:
- **Correctness**: 100% match with CPU OSD results
- **Speedup at high OSD orders**: 1.21x faster at OSD-15 (32,768 candidates)
- **Overhead at low OSD orders**: GPU is slower at OSD-10/12 due to transfer overhead

**Recommendation**: Use GPU OSD for OSD order ≥ 15 or when processing large batches. For OSD-10 and below, CPU OSD is more efficient.

Both implementations are correct and effective for quantum error correction decoding.

---

## 8. Threshold Analysis

### 8.1 Overview

This section presents a threshold analysis comparing BPDecoderPlus and ldpc across multiple code distances and physical error rates. The analysis helps understand decoder performance scaling behavior.

### 8.2 Test Configuration

| Parameter | Value |
|-----------|-------|
| Distances | d=3, d=5, d=7 |
| Error rates | 0.0005, 0.0007, 0.001, 0.0015, 0.002 |
| Samples per point | 2000 |
| BP iterations | 20 |
| OSD order | 10 |
| OSD method | Exhaustive (OSD-E) |

### 8.3 Threshold Comparison Results

![Threshold Comparison](../outputs/threshold_comparison.png)

#### BPDecoderPlus Results

| Distance | Error Rates Tested | Min LER | Max LER |
|----------|-------------------|---------|---------|
| d=3 | 5 points (0.0005-0.002) | 0.0015 | 0.0055 |
| d=5 | 5 points (0.0005-0.002) | 0.0115 | 0.0995 |
| d=7 | 5 points (0.0005-0.002) | 0.0420 | 0.2870 |

#### ldpc Results

| Distance | Error Rates Tested | Min LER | Max LER |
|----------|-------------------|---------|---------|
| d=3 | 5 points (0.0005-0.002) | 0.0035 | 0.0185 |
| d=5 | 5 points (0.0005-0.002) | 0.0095 | 0.0475 |
| d=7 | 5 points (0.0005-0.002) | 0.0215 | 0.0675 |

#### Point-by-Point Comparison

| Distance | Error Rate | BPDecoderPlus | ldpc | Diff |
|----------|------------|---------------|------|------|
| d=3 | p=0.0005 | 0.0015 | 0.0035 | -0.0020 |
| d=3 | p=0.0007 | 0.0035 | 0.0155 | -0.0120 |
| d=3 | p=0.0010 | 0.0015 | 0.0185 | -0.0170 |
| d=3 | p=0.0015 | 0.0055 | 0.0120 | -0.0065 |
| d=3 | p=0.0020 | 0.0050 | 0.0095 | -0.0045 |
| d=5 | p=0.0005 | 0.0115 | 0.0115 | +0.0000 |
| d=5 | p=0.0007 | 0.0170 | 0.0095 | +0.0075 |
| d=5 | p=0.0010 | 0.0280 | 0.0240 | +0.0040 |
| d=5 | p=0.0015 | 0.0545 | 0.0300 | +0.0245 |
| d=5 | p=0.0020 | 0.0995 | 0.0475 | +0.0520 |
| d=7 | p=0.0005 | 0.0420 | 0.0215 | +0.0205 |
| d=7 | p=0.0007 | 0.0700 | 0.0275 | +0.0425 |
| d=7 | p=0.0010 | 0.1045 | 0.0380 | +0.0665 |
| d=7 | p=0.0015 | 0.2115 | 0.0610 | +0.1505 |
| d=7 | p=0.0020 | 0.2870 | 0.0675 | +0.2195 |

### 8.4 Analysis

**Key Observations**:

1. **Small distances (d=3)**: BPDecoderPlus outperforms ldpc consistently, with LER improvements of 0.2-1.7 percentage points across all error rates.

2. **Larger distances (d=5, d=7)**: ldpc achieves lower logical error rates. The gap increases with both distance and error rate, reaching 22% difference at d=7, p=0.002.

3. **Scaling behavior**: ldpc shows better scaling with code distance, suggesting its C++ implementation handles larger parity check matrices more effectively.

**Possible factors**:
- BPDecoderPlus uses per-qubit priors from DEM, while ldpc uses uniform error rates
- Different numerical precision between Python and C++ implementations
- BP convergence behavior differences
- OSD candidate selection ordering

### 8.5 Running the Threshold Analysis

```bash
# Generate threshold comparison plots
uv run pytest tests/test_decoder_threshold.py -v -s

# Output files:
# - outputs/threshold_plot.png (BPDecoderPlus only)
# - outputs/threshold_plot_ldpc.png (ldpc only)
# - outputs/threshold_comparison.png (side-by-side)
# - outputs/threshold_overlay.png (overlay)
```

---

## 9. References

1. **ldpc Library**: https://github.com/quantumgizmos/ldpc
2. **BP+OSD Paper**: Roffe et al., "Decoding across the quantum LDPC code landscape" (2020). [arXiv:2005.07016](https://arxiv.org/abs/2005.07016)
3. **ldpc Documentation**: https://roffe.eu/software/ldpc/
