## PyTorch Belief Propagation Usage

This guide shows how to parse a UAI file, run BP, and apply evidence.
The implementation follows the tensor-contraction viewpoint in
TensorInference.jl: https://github.com/TensorBFS/TensorInference.jl

### Quick Start

```python
from pytorch_bp import read_model_file, BeliefPropagation, belief_propagate, compute_marginals

model = read_model_file("examples/simple_model.uai")
bp = BeliefPropagation(model)
state, info = belief_propagate(bp, max_iter=50, tol=1e-8, damping=0.1)
print(info)

marginals = compute_marginals(state, bp)
print(marginals[1])
```

### Evidence

```python
from pytorch_bp import read_model_file, read_evidence_file, apply_evidence
from pytorch_bp import BeliefPropagation, belief_propagate, compute_marginals

model = read_model_file("examples/simple_model.uai")
evidence = read_evidence_file("examples/simple_model.evid")
bp = apply_evidence(BeliefPropagation(model), evidence)
state, info = belief_propagate(bp)
marginals = compute_marginals(state, bp)
```

### Tips

- For loopy graphs, use damping between 0.1 and 0.5.
- Normalize messages to avoid numerical underflow.
- Use float64 for consistent comparisons in tests.
