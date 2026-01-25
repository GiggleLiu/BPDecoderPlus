# Getting the Threshold Plot

This guide walks through reproducing the circuit-level threshold plot for the BP+OSD decoder on rotated surface codes.

## Prerequisites

```bash
# Install the package and dependencies
uv sync

# Optional: install ldpc for comparison plots
uv pip install ldpc
```

## Step 1: Generate Threshold Datasets

Generate syndrome datasets for multiple code distances and error rates:

```bash
uv run python scripts/generate_threshold_datasets.py
```

This generates datasets for:
- **Distances:** d = 3, 5, 7, 9, 11
- **Error rates:** p = 0.0001 to 0.015
- **Samples:** 20,000 shots per configuration
- **Output:** `datasets/sc_d{d}_r{d}_p{pppp}_z.{dem,npz}`

Each dataset contains:
- `.dem` file: Detector Error Model from stim
- `.npz` file: Syndrome samples with observable ground truth

## Step 2: Run Threshold Analysis

```bash
uv run python scripts/analyze_threshold.py
```

This script:
1. Loads each dataset (d=3,5,7,9 at p=0.001 to 0.015)
2. Runs the BP+OSD decoder (GPU-accelerated if CUDA available)
3. Computes logical error rates (LER)
4. Generates threshold plots in `outputs/`

### Output Files

| File | Description |
|------|-------------|
| `outputs/threshold_plot.png` | BPDecoderPlus LER vs physical error rate |
| `outputs/threshold_plot_ldpc.png` | ldpc library LER (if installed) |
| `outputs/threshold_comparison.png` | Side-by-side comparison |
| `outputs/threshold_overlay.png` | Both decoders overlaid |

## Step 3: Interpret Results

The threshold is the physical error rate where LER curves for different distances cross. Below threshold, larger codes perform better; above threshold, they perform worse.

### Expected Output

```
d=3, p=0.001: LER~0.001    d=3, p=0.007: LER~0.034
d=5, p=0.001: LER~0.0004   d=5, p=0.007: LER~0.037
d=7, p=0.001: LER~0.0002   d=7, p=0.007: LER~0.031
d=9, p=0.001: LER~0.0000   d=9, p=0.007: LER~0.036
```

The crossing point near p ~ 0.007-0.009 indicates the threshold.

## Reference Validation

The circuit-level depolarizing noise threshold for the rotated surface code with BP+OSD decoding is approximately **0.7%** (p ~ 0.007).

This is consistent with the literature:

- **Bravyi et al. (Nature, 2024)** [arXiv:2308.07915](https://arxiv.org/abs/2308.07915):
  > "The pseudo-threshold p0 is defined as a solution of the break-even equation
  > pL(p) = kp. [...] BB codes offer a pseudo-threshold close to 0.7%, which is
  > nearly the same as the error threshold of the surface code."

  The surface code threshold of ~0.7% under circuit-level depolarizing noise is
  a well-established benchmark (Reference 49 in Bravyi et al.).

- **Higgott & Breuckmann (2023)** [arXiv:2303.15933](https://arxiv.org/abs/2303.15933):
  Report circuit-level thresholds for surface codes under depolarizing noise.

- **Dennis et al. (2002)**: The theoretical threshold for the surface code under
  independent depolarizing noise is ~10.3% (code capacity). Under circuit-level
  noise with measurement errors, this drops to ~0.5-1%.

Our BP+OSD decoder achieves performance consistent with these references, with
the LER curves crossing near p ~ 0.7%, validating the implementation.

## Decoder Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| BP iterations | 60 | Higher than typical due to complex circuit-level factor graphs |
| BP method | min-sum | Matches ldpc library default |
| Damping | 0.2 | Prevents oscillation |
| OSD order | 10 | OSD-CS with up to 2-bit flips in top-10 variables |
| Samples | 5000 | Per (distance, error_rate) point |

## DEM Parsing: Two-Stage Processing

Building the parity check matrix H from a DEM requires two processing stages for optimal BP decoding performance:

### Stage 1: Separator Splitting

The DEM format uses `^` separators to indicate correlated faults. For example:

```
error(0.01) D0 D1 ^ D2
```

This means a single fault event with probability 0.01 that triggers **both** detector patterns `{D0, D1}` **and** `{D2}` simultaneously. The first stage splits each error instruction by `^` separators:

- `D0 D1` becomes component 1 with prob=0.01
- `D2` becomes component 2 with prob=0.01

The `_split_error_by_separator` function handles this parsing. It was critical to restore this function (see Issue #61) because without it, correlated errors were incorrectly handled, leading to wrong H matrix structure.

### Stage 2: Hyperedge Merging

After separator splitting, **errors with identical detector patterns** are merged into single "hyperedges". This is the approach used by [PyMatching](https://github.com/oscarhiggott/PyMatching) when building decoding graphs from DEM files.

**Why hyperedge merging is required:**

1. **Errors with identical syndromes are indistinguishable** to the decoder
2. **Detectors are XOR-based**: if two errors trigger the same detector, they cancel
3. **Reduces factor graph size** for more efficient BP inference

**Probability combination (XOR formula):**

```python
p_combined = p_old + p_new - 2 * p_old * p_new
```

This formula gives P(odd number of errors fire), which is the correct probability for the merged hyperedge since detectors use XOR logic.

**Observable flip tracking:**

When merging hyperedges, we track P(observable flipped | hyperedge fires) as a soft probability (0.0-1.0) rather than binary. The decoder thresholds this at 0.5 for the final prediction.

### Implementation in `dem.py`

The `build_parity_check_matrix` function has two key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `split_by_separator` | True | Split by `^` separator (Stage 1) |
| `merge_hyperedges` | True | Merge identical detector patterns (Stage 2) |

**DO NOT REMOVE** the `merge_hyperedges` functionality. It is required for optimal threshold performance. See Issue #61 and PR #62 for the full context and history.

### Example: Effect on Matrix Size

For a d=3 surface code DEM:

| Processing | H columns | Description |
|------------|-----------|-------------|
| No splitting | ~286 | One per error instruction (wrong) |
| Split only | ~556 | One per component (correct but suboptimal) |
| Split + merge | ~400 | Merged hyperedges (optimal) |

The merged version has fewer columns because errors with identical detector patterns are combined, while still correctly representing the factor graph structure.

### Reference Implementation

PyMatching (https://github.com/oscarhiggott/PyMatching) uses a similar two-stage approach:
1. Parse DEM errors with separator handling
2. Build a decoding graph with merged edges for identical detector pairs

This is the standard approach for BP-based decoders on circuit-level noise models.

## Troubleshooting

### Missing Datasets

If `analyze_threshold.py` reports missing datasets:

```bash
# Regenerate specific datasets
uv run python scripts/generate_threshold_datasets.py
```

### GPU Memory Issues

For large batches, reduce `chunk_size` in `run_bpdecoderplus_gpu_batch()` or use CPU:

```python
device = 'cpu'  # Instead of 'cuda'
```
