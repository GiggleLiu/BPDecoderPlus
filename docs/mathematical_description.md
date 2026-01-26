## Belief Propagation (BP) Overview

This document summarizes the BP message-passing rules implemented in
`src/bpdecoderplus/pytorch_bp/belief_propagation.py` for discrete factor graphs. The approach
mirrors the tensor-contraction perspective used in TensorInference.jl.
See https://github.com/TensorBFS/TensorInference.jl for the Julia reference.

### Factor Graph Notation

- Variables are indexed by \(x_i\) with domain size \(d_i\).
- Factors are indexed by \(f\) and connect a subset of variables.
- Each factor has a tensor (potential) \(\phi_f\) defined over its variables.

### Messages

**Factor to variable message:**

\[
\mu_{f \to x}(x) = \sum_{\{y \in \text{ne}(f), y \neq x\}} \phi_f(x, y, \ldots) \prod_{y \neq x} \mu_{y \to f}(y)
\]

**Variable to factor message:**

\[
\mu_{x \to f}(x) = \prod_{g \in \text{ne}(x), g \neq f} \mu_{g \to x}(x)
\]

### Damping

To improve stability on loopy graphs, a damping update is applied:

\[
\mu_{\text{new}} = \alpha \cdot \mu_{\text{old}} + (1 - \alpha) \cdot \mu_{\text{candidate}}
\]

where \(\alpha\) is the damping factor (typically between 0 and 1).

### Convergence

We use an \(L_1\) difference threshold between consecutive factor-to-variable
messages to determine convergence:

\[
\max_{f,x} \| \mu_{f \to x}^{(t)} - \mu_{f \to x}^{(t-1)} \|_1 < \epsilon
\]

### Marginals

After convergence, variable marginals are computed as:

\[
b(x) = \frac{1}{Z} \prod_{f \in \text{ne}(x)} \mu_{f \to x}(x)
\]

The normalization constant \(Z\) is obtained by summing the unnormalized vector:

\[
Z = \sum_x \prod_{f \in \text{ne}(x)} \mu_{f \to x}(x)
\]
