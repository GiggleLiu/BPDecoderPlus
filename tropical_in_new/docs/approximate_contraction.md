# Approximate Tensor Network Contraction

This document describes the approximate contraction methods implemented for
tropical tensor networks, enabling scalable MPE decoding for surface codes
with code distance d >= 5.

## Background

Exact tensor network contraction has complexity exponential in the tree width
of the underlying factor graph. For circuit-level noise surface codes:

| Distance | Tree Width | Memory Required |
|----------|------------|-----------------|
| d=3      | ~9         | ~4 KB           |
| d=5      | ~36        | ~550 GB         |
| d=7      | ~80        | ~10^15 GB       |

This makes exact contraction infeasible for d >= 5. Approximate methods
trade exactness for scalability by limiting the bond dimension χ.

## Methods

### 1. MPS Boundary Contraction

Based on Bravyi et al. [arXiv:1405.4883], this method:

1. Arranges the tensor network into rows
2. Initializes a boundary MPS from the first row
3. For each subsequent row:
   - Constructs an MPO from the row tensors
   - Applies the MPO to the MPS
   - Truncates to bond dimension χ
4. Contracts the final MPS to a scalar

```python
from tropical_in_new.src import mpe_tropical

# Use MPS method with bond dimension 32
assignment, score, info = mpe_tropical(model, method="mps", chi=32)
```

### 2. Sweep Contraction

Based on Chubb [arXiv:2101.04125], this method:

1. Computes a 2D layout of tensors using spectral/force-directed embedding
2. Sweeps a line across the network in a chosen direction
3. Maintains a boundary MPS with bounded bond dimension
4. Contracts tensors as encountered by the sweep line

```python
from tropical_in_new.src import mpe_tropical

# Use sweep method with bond dimension 32
assignment, score, info = mpe_tropical(model, method="sweep", chi=32)
```

### 3. Auto Selection

The `method="auto"` option estimates contraction complexity and
automatically selects between exact and approximate methods:

```python
# Automatically select method based on problem size
assignment, score, info = mpe_tropical(model, method="auto")
```

## API Reference

### Main Functions

#### `mpe_tropical`

```python
def mpe_tropical(
    model: UAIModel,
    evidence: Dict[int, int] | None = None,
    method: Literal["exact", "mps", "sweep", "auto"] = "exact",
    chi: Optional[int] = None,
) -> tuple[Dict[int, int], float, Dict[str, int | tuple[int, ...]]]:
    """
    Perform MPE inference on a tropical tensor network.
    
    Args:
        model: UAI model to solve
        evidence: Dictionary of observed variable assignments
        method: Contraction method
        chi: Bond dimension for approximate methods
        
    Returns:
        Tuple of (assignment, score, info)
    """
```

#### `mpe_tropical_approximate`

```python
def mpe_tropical_approximate(
    model: UAIModel,
    evidence: Dict[int, int] | None = None,
    method: Literal["mps", "sweep"] = "mps",
    chi: Optional[int] = None,
) -> tuple[Dict[int, int], float, Dict[str, int | tuple[int, ...]]]:
    """
    Approximate MPE using MPS-based contraction.
    
    This function enables MPE inference for large tensor networks
    where exact contraction is infeasible.
    """
```

### Classes

#### `TropicalMPS`

Matrix Product State representation in the tropical semiring.

```python
from tropical_in_new.src import TropicalMPS

# Create from tensor
tensor = torch.randn(4, 4)
mps = TropicalMPS.from_tensor(tensor, chi=8)

# Convert back to tensor
recovered = mps.to_tensor()
```

#### `TropicalMPO`

Matrix Product Operator for row-by-row contraction.

```python
from tropical_in_new.src import TropicalMPO

# Create from row of tensors
mpo = TropicalMPO.from_tensor_row(tensors, connections)
```

### Utility Functions

#### `boundary_contract`

```python
from tropical_in_new.src import boundary_contract

result = boundary_contract(
    tensors=[...],      # List of tensors in log domain
    vars_list=[...],    # Variables for each tensor
    chi=32,             # Maximum bond dimension
    track_assignment=True,
)
print(f"Value: {result.value}, Chi used: {result.chi_used}")
```

#### `sweep_contract`

```python
from tropical_in_new.src import sweep_contract, SweepDirection

result = sweep_contract(
    nodes=[...],        # List of TensorNode objects
    chi=32,             # Maximum bond dimension
    direction=SweepDirection.LEFT_TO_RIGHT,
)
```

#### `estimate_required_chi`

```python
from tropical_in_new.src import estimate_required_chi

chi = estimate_required_chi(nodes, target_accuracy=0.99)
```

## Choosing Bond Dimension χ

The bond dimension χ controls the trade-off between accuracy and memory:

- **Higher χ**: Better approximation, more memory
- **Lower χ**: Faster, less memory, more approximation error

Guidelines:

| Network Size | Suggested χ | Memory (approx) |
|--------------|-------------|-----------------|
| Small (d=3)  | 16-32       | < 1 MB          |
| Medium (d=5) | 32-64       | 1-10 MB         |
| Large (d=7)  | 64-128      | 10-100 MB       |

Use `adaptive_sweep_contract` to automatically find a good χ:

```python
from tropical_in_new.src import adaptive_sweep_contract

result = adaptive_sweep_contract(
    nodes,
    chi_start=8,
    chi_max=128,
    chi_step=8,
)
```

## Tropical Semiring Considerations

In the tropical (max-plus) semiring:

- Multiplication → Addition: `a ⊗ b = a + b`
- Addition → Maximum: `a ⊕ b = max(a, b)`

SVD truncation in this semiring approximates by keeping the top-χ paths
with highest log-probability values.

## References

1. Bravyi et al., "Efficient algorithms for maximum likelihood decoding
   in the surface code" [arXiv:1405.4883](https://arxiv.org/abs/1405.4883)

2. Chubb, "General tensor network decoding of 2D Pauli codes"
   [arXiv:2101.04125](https://arxiv.org/abs/2101.04125)

3. Schotte et al., "Quantum error correction thresholds for the
   universal Fibonacci Turaev-Viro code"
   [arXiv:2012.04610](https://arxiv.org/abs/2012.04610)
