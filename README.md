# PyTorch Belief Propagation

This module provides a PyTorch implementation of belief propagation (BP)
for discrete factor graphs defined in the UAI format. The design follows
the tensor-contraction style used in TensorInference.jl.
See https://github.com/TensorBFS/TensorInference.jl for reference.

## Quick Start

```bash
pip install -r requirements.txt
```

## Environment Setup

Run commands from the repo root so imports resolve correctly.

Windows (PowerShell):

```powershell
$env:PYTHONPATH=(Get-Location).Path
```

macOS/Linux:

```bash
export PYTHONPATH="$(pwd)"
```

Alternative (works everywhere): install the package in editable mode.

```bash
pip install -e .
```

```python
from pytorch_bp import read_model_file, BeliefPropagation, belief_propagate, compute_marginals

model = read_model_file("examples/simple_model.uai")
bp = BeliefPropagation(model)
state, info = belief_propagate(bp)
print(info)
print(compute_marginals(state, bp))
```

## Examples

Run from the repo root so `pytorch_bp` is on the import path:

```bash
python examples/simple_example.py
python examples/evidence_example.py
```

## Tests

```bash
python -m unittest discover -s tests
```

What each unit test covers:

| Test file | Test case | Functions under test |
| --- | --- | --- |
| `tests/test_uai_parser.py` | `test_read_model_from_string` | `read_model_from_string` |
| `tests/test_uai_parser.py` | `test_read_evidence_file` | `read_evidence_file` |
| `tests/test_bp_basic.py` | `test_bp_matches_exact_tree` | `belief_propagate`, `compute_marginals` |
| `tests/test_bp_basic.py` | `test_apply_evidence` | `apply_evidence`, `belief_propagate`, `compute_marginals` |
| `tests/test_integration.py` | `test_example_file_runs` | `read_model_file`, `BeliefPropagation`, `belief_propagate`, `compute_marginals` |
| `tests/test_integration.py` | `test_example_with_evidence` | `read_evidence_file`, `apply_evidence` + BP pipeline |
| `tests/testcase.py` | `test_unary_factor_marginal` | `belief_propagate`, `compute_marginals` (unary factor) |
| `tests/testcase.py` | `test_chain_three_vars_exact` | `belief_propagate`, `compute_marginals` (tree exactness) |
| `tests/testcase.py` | `test_message_normalization` | `collect_message`, `process_message` (normalization) |
