# BP+OSD Decoder Fix: From Broken to Working

## Executive Summary

This document explains the critical bug in the original BP+OSD implementation that caused OSD post-processing to **increase** the logical error rate instead of decreasing it, and how it was fixed.

| Decoder | Logical Error Rate | Status |
|---------|-------------------|--------|
| BP-only | 10.9% | Baseline |
| BP+OSD (before fix) | ~40%+ | **Broken** |
| BP+OSD-15 (after fix) | 6.8% | **37.6% improvement** |

## Reproducibility

### Prerequisites

```bash
# Clone and setup
git clone https://github.com/your-org/BPDecoderPlus.git
cd BPDecoderPlus
git checkout fix/osd-decoder-improvements

# Install dependencies using uv
make setup
```

### Reproduce Results

```bash
# Run the demonstration (takes ~38 minutes on CPU)
uv run python run_batch_demo.py
```

**Expected Output:**
```
Device: cpu
Parity check matrix H: 24 checks x 286 error mechanisms
Test samples: 1000

Running BP (iter=20) + OSD-15 on 1000 samples...
OSD search space: 2^15 = 32,768 candidates per syndrome

  50/1000 (116s)
  100/1000 (233s)
  ...
  1000/1000 (2310s)

============================================================
Results (n=1000, BP iter=20, OSD order=15)
============================================================
BP-only Logical Error Rate:    0.1090 (109 errors)
BP+OSD-15 Logical Error Rate:  0.0680 (68 errors)
============================================================

Improvement: 37.6% reduction in logical errors
Total time: 2310s

Validating OSD produces valid codewords...
Valid syndromes: 100/100 (100% expected)
```

### Quick Validation (5 minutes)

For a quick test with fewer samples:

```bash
uv run python -c "
import numpy as np
import torch
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database
from bpdecoderplus.batch_bp import BatchBPDecoder
from bpdecoderplus.batch_osd import BatchOSDDecoder

dem = load_dem('datasets/sc_d3_r3_p0010_z.dem')
syndromes, observables, _ = load_syndrome_database('datasets/sc_d3_r3_p0010_z.npz')
H, priors, obs_flip = build_parity_check_matrix(dem)

bp_decoder = BatchBPDecoder(H, priors, device='cpu')
osd_decoder = BatchOSDDecoder(H, device='cpu')

# Test 100 samples
num_samples = 100
batch_syndromes = torch.from_numpy(syndromes[:num_samples]).float()
marginals = bp_decoder.decode(batch_syndromes, max_iter=20, damping=0.2)

pred_bp, pred_osd = [], []
for i in range(num_samples):
    probs = marginals[i].cpu().numpy()
    pred_bp.append(int(np.dot((probs > 0.5).astype(int), obs_flip) % 2))
    pred_osd.append(int(np.dot(osd_decoder.solve(syndromes[i], probs, osd_order=10), obs_flip) % 2))

ler_bp = np.mean(np.array(pred_bp) != observables[:num_samples])
ler_osd = np.mean(np.array(pred_osd) != observables[:num_samples])
print(f'BP-only: {ler_bp:.2%}, BP+OSD-10: {ler_osd:.2%}')
print(f'OSD improves BP: {ler_osd < ler_bp}')
"
```

**Expected:** OSD should improve (or match) BP performance. If OSD makes it worse, the fix is not applied.

## The Problem: Why Was OSD Making Things Worse?

### Original Implementation Issue

The original OSD implementation used **Hamming weight** (counting the number of 1s) to select the best candidate solution:

```python
# BROKEN: Old implementation
w = np.sum(cand_sol)  # Just counts number of errors
if w < min_weight:
    min_weight = w
    best_solution_sorted = cand_sol.copy()
```

**Why this is wrong:** Hamming weight treats all errors as equally likely, completely ignoring the soft information from BP. This means:

1. An error on a qubit with 99% confidence is treated the same as one with 51% confidence
2. OSD might select a solution that contradicts BP's high-confidence decisions
3. The decoder effectively "forgets" everything BP learned about error probabilities

### Concrete Example

Consider two candidate solutions for 4 qubits with BP probabilities [0.01, 0.99, 0.50, 0.50]:

| Solution | Hamming Weight | Interpretation |
|----------|---------------|----------------|
| [0,1,0,0] | 1 | Agrees with BP (q1 has 99% error prob) |
| [1,0,0,0] | 1 | Contradicts BP (q0 has only 1% error prob) |

With Hamming weight, both solutions look equally good (weight=1). But clearly [0,1,0,0] is far more likely given BP's beliefs!

## The Fix: Soft-Weighted Cost Function

### Mathematical Foundation

According to the BP+OSD paper (Panteleev & Kalachev, arXiv:2005.07016), the correct cost function uses **log-likelihood ratios (LLRs)**:

```
LLR_i = log((1 - p_i) / p_i)
```

Where `p_i` is BP's posterior probability that qubit `i` has an error.

- LLR > 0: BP thinks no error (p < 0.5)
- LLR < 0: BP thinks error (p > 0.5)
- |LLR| large: BP is confident
- |LLR| small: BP is uncertain

### The Soft-Weighted Cost

The cost penalizes solutions that **disagree with BP's confident decisions**:

```python
def _compute_soft_weight(self, solution: np.ndarray, llrs: np.ndarray) -> float:
    """
    Cost = sum of |LLR| for positions where solution disagrees with BP.
    
    - Disagreeing with a confident BP decision (large |LLR|) = high cost
    - Disagreeing with an uncertain BP decision (small |LLR|) = low cost
    """
    bp_hard_decision = (llrs < 0).astype(int)  # BP's best guess
    disagreement = (solution != bp_hard_decision).astype(float)
    cost = np.sum(disagreement * np.abs(llrs))
    return cost
```

### Why This Works

1. **Respects BP's confidence**: High-confidence errors contribute more to the cost if contradicted
2. **Allows correction of uncertain decisions**: OSD can flip low-confidence bits cheaply
3. **Maintains syndrome constraint**: All candidates satisfy H·e = s (valid codewords)
4. **Combines the best of both**: BP's soft information + OSD's guaranteed validity

## Code Changes Summary

### 1. Added Soft-Weighted Cost Function

```python
# NEW: Compute LLRs from BP probabilities
llrs = np.log((1 - probs) / probs)

# NEW: Use soft-weighted cost instead of Hamming weight
cost = self._compute_soft_weight(cand_sol, llrs_sorted)
if cost < min_cost:
    min_cost = cost
    best_solution_sorted = cand_sol.copy()
```

### 2. Simplified Interface

**Before (complex dictionary-based):**
```python
marginals_dict = {}
error_var_start = num_detectors + 1
for j in range(len(obs_flip)):
    var_idx = error_var_start + j
    marginals_dict[var_idx] = torch.tensor([1-p1, p1])
estimated = osd_decoder.solve(syndrome, marginals_dict, error_var_start, osd_order=10)
```

**After (simple numpy array):**
```python
error_probs = marginals.cpu().numpy()
estimated = osd_decoder.solve(syndrome, error_probs, osd_order=15)
```

## Experimental Results

Tested on surface code d=3, r=3, p=0.001 (1000 samples):

| Configuration | Logical Error Rate | Errors/1000 |
|--------------|-------------------|-------------|
| BP-only (iter=20) | 10.9% | 109 |
| BP+OSD-15 (iter=20) | 6.8% | 68 |

**Improvement: 37.6% reduction in logical errors**

### Performance Notes

- OSD-15 searches 2^15 = 32,768 candidates per syndrome
- Total runtime: ~38 minutes for 1000 samples on CPU
- Higher OSD order (e.g., 20) gives better results but exponentially slower

## Files Modified

1. **`src/bpdecoderplus/osd.py`**
   - Added `_compute_soft_weight()` method
   - Changed `solve()` to accept `error_probs: np.ndarray` directly
   - Replaced Hamming weight with soft-weighted cost in OSD-E search

2. **`run_batch_demo.py`**
   - Updated to use simplified OSD interface
   - Added comparison of BP-only vs BP+OSD-0 vs BP+OSD-15

## References

- Panteleev, P., & Kalachev, G. (2021). Degenerate Quantum LDPC Codes With Good Finite Length Performance. *Quantum*, 5, 585. [arXiv:2005.07016](https://arxiv.org/abs/2005.07016)

## Conclusion

The key insight is that **OSD should respect BP's soft information**, not just count errors. By using a soft-weighted cost function based on LLRs, we achieve:

1. **Correct behavior**: OSD now improves upon BP instead of degrading it
2. **Significant gains**: 37.6% reduction in logical error rate
3. **Theoretical soundness**: Follows the methodology from the original BP+OSD paper
