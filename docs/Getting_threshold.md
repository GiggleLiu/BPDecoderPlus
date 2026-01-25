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

## DEM Parsing: Separator Splitting

The Detector Error Model (DEM) format uses `^` separators to indicate correlated faults. For example:

```
error(0.01) D0 D1 ^ D2
```

This means a single fault event with probability 0.01 that triggers **both** detector patterns `{D0, D1}` **and** `{D2}` simultaneously. The current implementation uses **separator splitting**: each component separated by `^` becomes a separate column in the parity check matrix H, sharing the same probability.

### Why Separator Splitting is Correct

For BP decoding, errors with `^` separators represent correlated faults where multiple qubits are affected simultaneously. By splitting these into separate columns:

1. Each component can be independently estimated by BP
2. The parity check matrix H correctly represents the syndrome patterns
3. OSD post-processing finds valid error patterns that satisfy the syndrome

The `_split_error_by_separator` function in `dem.py` handles this parsing. It was critical to restore this function (see Issue #61) because without it, correlated errors were incorrectly merged, leading to wrong H matrix structure.

### Alternative: Hyperedge Merging

An alternative approach (used in early development) merges errors with **identical detector patterns** into single "hyperedges" using XOR probability combination:

```python
p_combined = p_old + p_new - 2 * p_old * p_new
```

With this approach:
- `obs_flip` contains soft probabilities (0.0-1.0) representing P(observable flip | hyperedge fires)
- Observable prediction uses soft XOR probability chain instead of binary mod-2

Both approaches are mathematically valid. The current separator splitting approach is simpler and produces equivalent decoding results. Small LER differences (~0.001-0.003) between implementations are within statistical tolerance given sample sizes of 5000.

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
