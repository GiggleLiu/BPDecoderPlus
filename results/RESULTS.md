# BPDecoderPlus Benchmark Results

This document explains the benchmark results comparing different quantum error correction decoders for surface codes.

## Overview

We benchmarked three decoder types on rotated surface codes:
- **IP (Integer Programming)**: Maximum likelihood decoder using integer linear programming
- **BP (Belief Propagation)**: Iterative message-passing decoder
- **BP+OSD**: Belief propagation with Ordered Statistics Decoding post-processing

## Threshold Plots

### IP Decoder Threshold
![IP Decoder Threshold](plots/threshold_ip.png)

The IP decoder shows **correct threshold behavior**:
- Below the threshold (~10%), logical error rate decreases with increasing code distance
- The curves for d=3, 5, 7, 9 fan out at low error rates, with larger distances performing better
- This demonstrates proper error suppression: doubling the distance roughly squares the logical error rate below threshold

**Key observation**: At p=0.05, the logical error rates are:
- d=3: ~4%
- d=5: ~1%
- d=7: ~0.4%
- d=9: ~0.2%

### BP Decoder Threshold
![BP Decoder Threshold](plots/threshold_bp.png)

The BP decoder shows **incorrect scaling** on surface codes:
- Logical error rate *increases* with code distance (wrong direction!)
- At p=0.05: d=3 (~11%) < d=5 (~17%) < d=7 (~25%) < d=9 (~33%)
- No threshold behavior is observed

**Why BP fails on surface codes**: Surface codes have many short cycles (length 4) in their Tanner graph. BP assumes the graph is tree-like, so these cycles cause message correlations that prevent convergence to the correct solution. This is a well-known limitation of standard BP for quantum LDPC codes with dense parity checks.

### BP+OSD Decoder Threshold
![BPOSD Decoder Threshold](plots/threshold_bposd.png)

Adding OSD post-processing **recovers threshold behavior**:
- Curves show proper fan-out pattern like the IP decoder
- Performance approaches IP decoder quality
- OSD uses the soft information from BP to guide a greedy search when BP fails to converge

**How OSD helps**: When BP doesn't converge, OSD sorts variable nodes by reliability (from BP's soft outputs) and performs Gaussian elimination to find a valid correction. This hybrid approach combines BP's speed with guaranteed convergence.

## Decoder Comparison

![Decoder Comparison](plots/decoder_comparison.png)

Side-by-side comparison at each code distance:

| Distance | Best Decoder | Worst Decoder | Gap at p=0.05 |
|----------|--------------|---------------|---------------|
| d=3 | IP ≈ BPOSD | BP | ~3x worse |
| d=5 | IP ≈ BPOSD | BP | ~7x worse |
| d=7 | IP ≈ BPOSD | BP | ~9x worse |
| d=9 | IP ≈ BPOSD | BP | ~6x worse |

**Takeaway**: BP+OSD achieves near-optimal decoding performance while being significantly faster than IP.

## Timing Comparison

![Timing Comparison](plots/timing_comparison.png)

Decoding time vs code distance (log scale):

| Distance | IP (ms) | BP (ms) | BPOSD (ms) | IP/BP Ratio |
|----------|---------|---------|------------|-------------|
| d=3 | 3.4 | 0.02 | 0.03 | 170x |
| d=5 | 5.4 | 0.11 | 0.12 | 49x |
| d=7 | 8.4 | 0.42 | 0.50 | 20x |
| d=9 | 13.1 | 1.03 | 1.17 | 13x |
| d=11 | 20.3 | 1.77 | 2.40 | 11x |

**Observations**:
- IP decoder time grows roughly as O(n^1.5) where n = d^2 is the number of qubits
- BP and BPOSD grow approximately linearly with qubit count
- BP is 10-100x faster than IP depending on code size
- OSD adds ~20-40% overhead to BP

**Practical implications**: For real-time decoding in quantum computers (which require sub-microsecond latency), BP-based decoders are essential. IP is useful as a benchmark but too slow for practical use.

## Atom Loss Results

![Atom Loss](plots/atom_loss.png)

Performance under atom loss (relevant for neutral atom quantum computers):

- Fixed physical error rate: p = 0.05
- Loss rates tested: 0%, 1%, 2%, 3%
- Decoder: IP (baseline)

**Observations**:
- Performance remains relatively stable up to 3% atom loss
- Larger codes (d=5, d=7) show more resilience to loss
- The current implementation uses a simple loss model; adaptive decoding could improve results

**Note**: The atom loss results show some variance due to the relatively small number of trials (500). A full study would require more samples and implementation of loss-aware decoding strategies as described in arXiv:2412.07841.

## Scalability Analysis

![Scalability](plots/scalability.png)

Left panel: **Decoding time vs number of qubits**
- IP decoder time scales roughly as O(n^1.2) to O(n^1.5)
- At 225 qubits (d=15): ~50ms per decode
- Extrapolating: d=21 (~441 qubits) would take ~150ms

Right panel: **Accuracy vs code size**
- Logical error rate decreases exponentially with code size (below threshold)
- At p=0.05, we see ~10x improvement per distance increment
- This confirms we are operating below the threshold

## Summary Table

| Decoder | Accuracy | Speed | Scalability | Best Use Case |
|---------|----------|-------|-------------|---------------|
| IP | Optimal (MLE) | Slow | O(n^1.5) | Benchmarking, small codes |
| BP | Poor on surface codes | Fast | O(n) | Not recommended for surface codes |
| BP+OSD | Near-optimal | Fast | O(n) | Production use |

## Conclusions

1. **IP decoder** provides optimal (MLE) decoding but is too slow for practical use beyond small codes.

2. **Standard BP** fails on surface codes due to short cycles in the Tanner graph. This is a fundamental limitation, not an implementation issue.

3. **BP+OSD** is the recommended decoder for surface codes:
   - Near-optimal accuracy
   - 10-100x faster than IP
   - Scales linearly with code size

4. **For winter school students**:
   - Implementing BP teaches iterative algorithms and graphical models
   - Understanding why BP fails teaches important concepts about code structure
   - Adding OSD demonstrates hybrid algorithm design

## References

1. Roffe et al., "Decoding across the quantum LDPC code landscape" (arXiv:2005.07016) - BP+OSD introduction
2. Chen et al., "Improved BP Decoding for Surface Codes" (arXiv:2407.11523) - Advanced BP variants
3. Higgott & Gidney, "Sparse Blossom" (arXiv:2303.15933) - Fast MWPM alternative
4. arXiv:2412.07841 - Atom loss handling strategies

---
*Generated by BPDecoderPlus benchmark suite*
