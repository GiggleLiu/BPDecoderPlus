## Tropical Tensor Network for MPE

We compute the Most Probable Explanation (MPE) by treating the factor
graph as a tensor network in the tropical semiring (max-plus).

### Tropical Semiring

For log-potentials, multiplication becomes addition and summation becomes
maximization:

- `a ⊗ b = a + b`
- `a ⊕ b = max(a, b)`

### MPE Objective

Let `x` be the set of variables and `phi_f(x_f)` the factor potentials.
We maximize the log-score:

`x* = argmax_x sum_f log(phi_f(x_f))`

### Contraction

Each factor becomes a tensor over its variable scope. Contraction combines
two tensors by adding their log-values and reducing (max) over eliminated
variables. A greedy elimination order (min-fill or min-degree) controls
the intermediate tensor sizes.

### Backpointers

During each reduction, we store the argmax index for eliminated variables.
Traversing the contraction tree with these backpointers recovers the MPE
assignment.
