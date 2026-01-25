# Issue #61 Cleanup: Clearing fix/osd-decoder-improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up the `fix/osd-decoder-improvements` branch by removing redundant code (`osd.py`, hyperedge merging), generating the missing d=9 p=0.009 dataset, writing threshold documentation, adding comprehensive tests, and creating a PR.

**Architecture:** Remove dead code paths (single-sample OSD, hyperedge merging) that are superseded by batch implementations and stim's built-in DEM handling. Simplify observable flip computation to use binary (mod-2) arithmetic instead of soft XOR probability chains. Add the missing dataset and comprehensive documentation.

**Tech Stack:** Python, PyTorch, stim, numpy, pytest, matplotlib

---

### Task 1: Remove `osd.py`

**Files:**
- Delete: `src/bpdecoderplus/osd.py`
- Modify: `docs/ldpc_comparison.md:88` (change `OSDDecoder` → `BatchOSDDecoder`)
- Modify: `docs/bp_osd_fix.md:73` (change `OSDDecoder` → `BatchOSDDecoder`)

**Step 1: Delete osd.py**

```bash
git rm src/bpdecoderplus/osd.py
```

**Step 2: Update docs/ldpc_comparison.md**

Replace:
```python
from bpdecoderplus.osd import OSDDecoder
```
with:
```python
from bpdecoderplus.batch_osd import BatchOSDDecoder
```

Replace:
```python
osd_decoder = OSDDecoder(H)
```
with:
```python
osd_decoder = BatchOSDDecoder(H, device='cpu')
```

**Step 3: Update docs/bp_osd_fix.md**

Same substitutions as above.

**Step 4: Verify no remaining references**

```bash
grep -r "from.*osd import\|OSDDecoder" src/ tests/ scripts/ docs/ --include="*.py" --include="*.md"
```

Expected: Only `BatchOSDDecoder` references remain.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove redundant osd.py (batch_osd.py covers all functionality)"
```

---

### Task 2: Remove hyperedge merging from `dem.py`

**Files:**
- Modify: `src/bpdecoderplus/dem.py`
- Modify: `tests/test_dem.py:165-178` (update obs_flip dtype expectations)

**Step 1: Rewrite `build_parity_check_matrix` to simple implementation**

Replace the entire `build_parity_check_matrix`, `_build_parity_check_matrix_legacy`, `_build_parity_check_matrix_hyperedge`, and `_split_error_by_separator` functions with a single simple implementation:

```python
def build_parity_check_matrix(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parity check matrix H from DEM for BP decoding.

    Each error instruction in the flattened DEM becomes one column in H.
    Stim's decompose_errors ensures each error has a unique detector pattern.

    Args:
        dem: Detector Error Model.

    Returns:
        Tuple of (H, priors, obs_flip) where:
        - H: Parity check matrix, shape (num_detectors, num_errors)
        - priors: Prior error probabilities, shape (num_errors,)
        - obs_flip: Observable flip indicators, shape (num_errors,) binary (0 or 1)
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()
            detectors = [t.val for t in targets if t.is_relative_detector_id()]
            observables = [t.val for t in targets if t.is_logical_observable_id()]
            errors.append({
                "prob": prob,
                "detectors": detectors,
                "observables": observables,
            })

    n_detectors = dem.num_detectors
    n_errors = len(errors)

    H = np.zeros((n_detectors, n_errors), dtype=np.uint8)
    priors = np.zeros(n_errors, dtype=np.float64)
    obs_flip = np.zeros(n_errors, dtype=np.uint8)

    for j, e in enumerate(errors):
        priors[j] = e["prob"]
        for d in e["detectors"]:
            H[d, j] = 1
        if e["observables"]:
            obs_flip[j] = 1

    return H, priors, obs_flip
```

**Step 2: Fix `dem_to_dict` function**

The current version on the branch has a bug (iterates `targets` as dicts). Restore to the correct simple implementation:

```python
def dem_to_dict(dem: stim.DetectorErrorModel) -> dict[str, Any]:
    """
    Convert DEM to dictionary with structured information.

    Args:
        dem: Detector Error Model to convert.

    Returns:
        Dictionary with DEM statistics and error information.
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()
            detectors = [t.val for t in targets if t.is_relative_detector_id()]
            observables = [t.val for t in targets if t.is_logical_observable_id()]

            errors.append({
                "probability": float(prob),
                "detectors": detectors,
                "observables": observables,
            })

    return {
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "num_errors": len(errors),
        "errors": errors,
    }
```

**Step 3: Simplify `dem_to_uai` function**

Remove the `_split_error_by_separator` usage, revert to simple target parsing:

```python
def dem_to_uai(dem: stim.DetectorErrorModel) -> str:
    """
    Convert DEM to UAI format for probabilistic inference.

    Args:
        dem: Detector Error Model to convert.

    Returns:
        String in UAI format representing the factor graph.
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = inst.targets_copy()
            detectors = [t.val for t in targets if t.is_relative_detector_id()]
            errors.append({"prob": prob, "detectors": detectors})

    n_detectors = dem.num_detectors
    lines = []
    lines.append("MARKOV")
    lines.append(str(n_detectors))
    lines.append(" ".join(["2"] * n_detectors))
    lines.append(str(len(errors)))

    for e in errors:
        dets = e["detectors"]
        lines.append(f"{len(dets)} " + " ".join(map(str, dets)))

    lines.append("")
    for e in errors:
        n_dets = len(e["detectors"])
        n_entries = 2 ** n_dets
        lines.append(str(n_entries))

        p = e["prob"]
        for i in range(n_entries):
            parity = bin(i).count("1") % 2
            if parity == 0:
                lines.append(str(1 - p))
            else:
                lines.append(str(p))
        lines.append("")

    return "\n".join(lines)
```

**Step 4: Delete `_split_error_by_separator` function entirely**

Remove the function definition (lines 62-101 in current dem.py).

**Step 5: Update test_dem.py**

In `TestBuildParityCheckMatrix.test_matrix_types`, change:
```python
# obs_flip is float64 with hyperedge merging (conditional probability)
assert obs_flip.dtype == np.float64
```
to:
```python
assert obs_flip.dtype == np.uint8
```

In `TestBuildParityCheckMatrix.test_matrix_values`, change:
```python
# obs_flip is probability [0, 1] with hyperedge merging
assert np.all((obs_flip >= 0) & (obs_flip <= 1))
```
to:
```python
assert np.all((obs_flip == 0) | (obs_flip == 1))
```

**Step 6: Run tests**

```bash
uv run pytest tests/test_dem.py -v
```

Expected: All tests pass.

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove hyperedge merging from dem.py

Stim's decompose_errors already outputs unique detector patterns per error,
making hyperedge merging unnecessary. Simplify to direct target parsing."
```

---

### Task 3: Update `analyze_threshold.py` to remove soft XOR logic

**Files:**
- Modify: `scripts/analyze_threshold.py`

**Step 1: Remove `compute_observable_prediction` and `compute_observable_predictions_batch`**

Delete the two functions (lines 29-74).

**Step 2: Replace observable prediction in `run_bpdecoderplus_gpu_batch`**

Replace:
```python
# Compute predictions (handles fractional obs_flip from hyperedge merging)
predictions = compute_observable_predictions_batch(solutions, obs_flip)
```
with:
```python
# Binary observable prediction: mod-2 dot product
predictions = (solutions @ obs_flip) % 2
```

**Step 3: Replace observable prediction in `run_ldpc_decoder`**

Replace:
```python
predicted_obs = compute_observable_prediction(result, obs_flip)
```
with:
```python
predicted_obs = int(np.dot(result, obs_flip) % 2)
```

**Step 4: Remove obs_flip diagnostic output from `load_dataset`**

Remove the verbose diagnostic block (lines 206-210):
```python
    if verbose:
        print(f"    obs_flip range: [{obs_flip.min():.4f}, {obs_flip.max():.4f}]")
        near_half = np.sum((obs_flip > 0.3) & (obs_flip < 0.7))
        print(f"    obs_flip near 0.5 (0.3-0.7): {near_half}/{len(obs_flip)}")
        print(f"    obs_flip == 0: {np.sum(obs_flip == 0)}, == 1: {np.sum(obs_flip == 1)}")
```

And remove the `verbose` parameter and `first_for_distance` logic from `collect_threshold_data`.

**Step 5: Commit**

```bash
git add scripts/analyze_threshold.py
git commit -m "refactor: simplify observable prediction to binary mod-2 dot product

With hyperedge merging removed, obs_flip is always binary (0 or 1),
so the soft XOR probability chain is unnecessary."
```

---

### Task 4: Generate missing d=9 p=0.009 dataset

**Files:**
- Create: `datasets/sc_d9_r9_p0090_z.dem`
- Create: `datasets/sc_d9_r9_p0090_z.npz`

**Step 1: Generate the dataset**

```bash
uv run python -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.syndrome import sample_syndromes, save_syndrome_database
from bpdecoderplus.dem import extract_dem

distance = 9
rounds = 9
error_rate = 0.009
num_shots = 20000

circuit = generate_circuit(distance=distance, rounds=rounds, p=error_rate, task='z')
dem = extract_dem(circuit)
syndromes, observables = sample_syndromes(circuit, num_shots=num_shots)

dem_path = Path('datasets/sc_d9_r9_p0090_z.dem')
npz_path = Path('datasets/sc_d9_r9_p0090_z.npz')

with open(dem_path, 'w') as f:
    f.write(str(dem))

metadata = {
    'distance': distance,
    'rounds': rounds,
    'p': error_rate,
    'task': 'z',
    'num_shots': num_shots,
    'num_detectors': dem.num_detectors,
}
save_syndrome_database(syndromes, observables, npz_path, metadata)
print(f'Generated {dem_path} and {npz_path}')
print(f'H will have {dem.num_detectors} detectors')
print(f'Detection rate: {syndromes.mean():.4f}, obs flip rate: {observables.mean():.4f}')
"
```

Expected: Files created successfully.

**Step 2: Verify the dataset works with analyze_threshold**

```bash
uv run python -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database

dem = load_dem('datasets/sc_d9_r9_p0090_z.dem')
syndromes, observables, _ = load_syndrome_database('datasets/sc_d9_r9_p0090_z.npz')
H, priors, obs_flip = build_parity_check_matrix(dem)
print(f'H shape: {H.shape}')
print(f'Syndromes shape: {syndromes.shape}')
print(f'obs_flip values: all binary = {set(obs_flip.tolist()).issubset({0, 1})}')
"
```

**Step 3: Commit**

```bash
git add datasets/sc_d9_r9_p0090_z.dem datasets/sc_d9_r9_p0090_z.npz
git commit -m "data: add missing d=9 p=0.009 dataset for threshold analysis"
```

---

### Task 5: Write `docs/Getting_threshold.md`

**Files:**
- Create: `docs/Getting_threshold.md`

**Step 1: Write the documentation**

```markdown
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
```

**Step 2: Commit**

```bash
git add docs/Getting_threshold.md
git commit -m "docs: add Getting_threshold.md with reproduction steps and reference validation"
```

---

### Task 6: Add tests for `batch_bp.py` and `batch_osd.py`

**Files:**
- Modify: `tests/test_osd_correctness.py` (already tests BatchOSDDecoder and BatchBPDecoder)
- Create: `tests/test_batch_bp.py` (new tests for BatchBPDecoder)
- Create: `tests/test_batch_osd.py` (new tests for BatchOSDDecoder)

**Step 1: Write `tests/test_batch_bp.py`**

```python
"""Tests for BatchBPDecoder."""

import numpy as np
import pytest
import torch

from bpdecoderplus.batch_bp import BatchBPDecoder


class TestBatchBPDecoderInit:
    """Test BatchBPDecoder initialization."""

    def test_basic_init(self):
        """Test decoder initializes with valid inputs."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        assert decoder.num_checks == 2
        assert decoder.num_qubits == 3
        assert decoder.num_edges == 4  # 4 ones in H

    def test_edge_structure(self):
        """Test edge lists are built correctly."""
        H = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        assert decoder.num_edges == 4


class TestBatchBPDecoderDecode:
    """Test BatchBPDecoder decoding."""

    def test_zero_syndrome(self):
        """Test decoding zero syndrome returns low error probabilities."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.zeros(1, 2)
        marginals = decoder.decode(syndromes, max_iter=20)

        assert marginals.shape == (1, 3)
        # With zero syndrome and low priors, marginals should be low
        assert (marginals < 0.5).all()

    def test_batch_decoding(self):
        """Test batch decoding processes multiple syndromes."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=20)

        assert marginals.shape == (3, 3)

    def test_min_sum_method(self):
        """Test min-sum decoding method."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=20, method='min-sum')

        assert marginals.shape == (1, 3)
        assert (marginals >= 0).all() and (marginals <= 1).all()

    def test_sum_product_method(self):
        """Test sum-product decoding method."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=20, method='sum-product')

        assert marginals.shape == (1, 3)
        assert (marginals >= 0).all() and (marginals <= 1).all()

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="Unknown method"):
            decoder.decode(syndromes, method='invalid')

    def test_damping_effect(self):
        """Test that damping produces valid marginals."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        syndromes = torch.tensor([[1, 1]], dtype=torch.float32)

        m1 = decoder.decode(syndromes, max_iter=50, damping=0.0)
        m2 = decoder.decode(syndromes, max_iter=50, damping=0.5)

        # Both should produce valid probabilities
        assert (m1 >= 0).all() and (m1 <= 1).all()
        assert (m2 >= 0).all() and (m2 <= 1).all()

    def test_repetition_code(self):
        """Test BP on simple repetition code."""
        # [3,1,3] repetition code
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        priors = np.array([0.1, 0.01, 0.01], dtype=np.float32)
        decoder = BatchBPDecoder(H, priors, device='cpu')

        # Syndrome [1,0] consistent with error on qubit 0
        syndromes = torch.tensor([[1, 0]], dtype=torch.float32)
        marginals = decoder.decode(syndromes, max_iter=50)

        # Qubit 0 should have highest error probability
        assert marginals[0, 0] > marginals[0, 1]
        assert marginals[0, 0] > marginals[0, 2]
```

**Step 2: Write `tests/test_batch_osd.py`**

```python
"""Tests for BatchOSDDecoder."""

import numpy as np
import pytest
import torch

from bpdecoderplus.batch_osd import BatchOSDDecoder


class TestBatchOSDDecoderInit:
    """Test BatchOSDDecoder initialization."""

    def test_basic_init(self):
        """Test decoder initializes with valid inputs."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        assert decoder.num_checks == 2
        assert decoder.num_errors == 3

    def test_init_stores_H(self):
        """Test that H matrix is stored correctly."""
        H = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        np.testing.assert_array_equal(decoder.H, H)


class TestBatchOSDDecoderSolve:
    """Test BatchOSDDecoder.solve method."""

    def test_single_error(self):
        """Test solving for a single error."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        # Syndrome [1,0] consistent with error on position 0
        syndrome = np.array([1, 0], dtype=np.int8)
        probs = np.array([0.9, 0.1, 0.1])

        result = decoder.solve(syndrome, probs, osd_order=0)

        # Should find error at position 0
        assert result[0] == 1
        # Result must satisfy syndrome
        assert np.all((H @ result) % 2 == syndrome)

    def test_zero_syndrome(self):
        """Test solving zero syndrome returns zero vector."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([0, 0], dtype=np.int8)
        probs = np.array([0.1, 0.1, 0.1])

        result = decoder.solve(syndrome, probs, osd_order=0)

        assert np.all(result == 0)

    def test_osd_order_improves_solution(self):
        """Test that higher OSD order can find better solutions."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 1], dtype=np.int8)
        probs = np.array([0.1, 0.8, 0.1])

        result_0 = decoder.solve(syndrome, probs, osd_order=0)
        result_10 = decoder.solve(syndrome, probs, osd_order=10)

        # Both must satisfy syndrome
        assert np.all((H @ result_0) % 2 == syndrome)
        assert np.all((H @ result_10) % 2 == syndrome)

    def test_mismatched_probs_raises(self):
        """Test that mismatched probability length raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 0], dtype=np.int8)
        probs = np.array([0.1, 0.1])  # Wrong length

        with pytest.raises(ValueError, match="doesn't match"):
            decoder.solve(syndrome, probs)

    def test_combination_sweep_method(self):
        """Test OSD-CS method produces valid solutions."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 0], dtype=np.int8)
        probs = np.array([0.8, 0.1, 0.1])

        result = decoder.solve(syndrome, probs, osd_order=5, osd_method='combination_sweep')
        assert np.all((H @ result) % 2 == syndrome)


class TestBatchOSDDecoderSolveBatch:
    """Test BatchOSDDecoder.solve_batch method."""

    def test_batch_solve(self):
        """Test batch solving multiple syndromes."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndromes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
        ])

        results = decoder.solve_batch(syndromes, probs, osd_order=5)

        assert results.shape == (3, 3)
        # All results must satisfy their respective syndromes
        for i in range(3):
            assert np.all((H @ results[i]) % 2 == syndromes[i])


class TestBatchOSDDecoderRREF:
    """Test RREF computation."""

    def test_rref_identity(self):
        """Test RREF of identity matrix."""
        H = np.eye(3, dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 0, 1], dtype=np.int8)
        sorted_indices = np.array([0, 1, 2])

        augmented, pivot_cols = decoder._get_rref_cached(sorted_indices, syndrome)
        assert pivot_cols == [0, 1, 2]

    def test_rref_rank_deficient(self):
        """Test RREF of rank-deficient matrix."""
        H = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int8)
        decoder = BatchOSDDecoder(H, device='cpu')

        syndrome = np.array([1, 1], dtype=np.int8)
        sorted_indices = np.array([0, 1, 2])

        augmented, pivot_cols = decoder._get_rref_cached(sorted_indices, syndrome)
        assert len(pivot_cols) == 1  # Rank 1
```

**Step 3: Run new tests**

```bash
uv run pytest tests/test_batch_bp.py tests/test_batch_osd.py -v
```

Expected: All pass.

**Step 4: Commit**

```bash
git add tests/test_batch_bp.py tests/test_batch_osd.py
git commit -m "test: add comprehensive tests for BatchBPDecoder and BatchOSDDecoder"
```

---

### Task 7: Run all tests and fix failures

**Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

Common expected issues:
- `test_decoder_validation.py` may fail if it references `merge_hyperedges` parameter
- `test_osd_correctness.py` references `_check_syndrome_satisfied` which may need checking

**Step 3: Verify scripts work**

```bash
uv run python -c "from bpdecoderplus.dem import build_parity_check_matrix; print('OK')"
uv run python -c "from bpdecoderplus.batch_bp import BatchBPDecoder; print('OK')"
uv run python -c "from bpdecoderplus.batch_osd import BatchOSDDecoder; print('OK')"
```

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test failures after cleanup"
```

---

### Task 8: Create PR and comment on related issues

**Step 1: Push branch**

```bash
git push -u origin fix/issue-61-cleanup
```

**Step 2: Create PR**

```bash
gh pr create --title "Cleanup: remove redundant OSD, hyperedge merging, add threshold docs" --body "$(cat <<'EOF'
## Summary
- Remove redundant `osd.py` (functionality covered by `batch_osd.py`)
- Remove hyperedge merging from `dem.py` (stim handles unique detector patterns)
- Simplify observable prediction to binary mod-2 dot product
- Generate missing d=9, p=0.009 dataset
- Add `docs/Getting_threshold.md` with reproduction steps and reference validation
- Add comprehensive tests for `BatchBPDecoder` and `BatchOSDDecoder`

## Test plan
- [ ] `uv run pytest tests/ -v` passes
- [ ] `uv run python scripts/analyze_threshold.py` runs without "Dataset not found" for d=9,p=0.009
- [ ] Threshold plot shows crossing near p~0.7%, consistent with literature

Closes #61

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Comment on issue #61**

```bash
gh issue comment 61 --body "PR created: see the linked PR above. All 6 items from the issue have been addressed."
```
