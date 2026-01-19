## Belief Propagation (BP) Overview

This document summarizes the BP message-passing rules implemented in
`src/bpdecoderplus/pytorch_bp/belief_propagation.py` for discrete factor graphs. The approach
mirrors the tensor-contraction perspective used in TensorInference.jl.
See https://github.com/TensorBFS/TensorInference.jl for the Julia reference.

### Factor Graph Notation

- Variables are indexed by x_i with domain size d_i.
- Factors are indexed by f and connect a subset of variables.
- Each factor has a tensor (potential) phi_f defined over its variables.

### Messages

Factor to variable message:

mu_{f->x}(x) = sum_{all y in ne(f), y != x} phi_f(x, y, ...) * product_{y != x} mu_{y->f}(y)

Variable to factor message:

mu_{x->f}(x) = product_{g in ne(x), g != f} mu_{g->x}(x)

### Damping

To improve stability on loopy graphs, a damping update is applied:

mu_new = damping * mu_old + (1 - damping) * mu_candidate

### Convergence

We use an L1 difference threshold between consecutive factor->variable
messages to determine convergence.

### Marginals

After convergence, variable marginals are computed as:

b(x) = (1 / Z) * product_{f in ne(x)} mu_{f->x}(x)

The normalization constant Z is obtained by summing the unnormalized vector.
