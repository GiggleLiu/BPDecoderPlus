# BPDecoderPlus: Quantum Error Correction with Belief Propagation

[![Tests](https://github.com/TensorBFS/BPDecoderPlus/actions/workflows/test.yml/badge.svg)](https://github.com/TensorBFS/BPDecoderPlus/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/TensorBFS/BPDecoderPlus/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/BPDecoderPlus)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://tensorbfs.github.io/BPDecoderPlus/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python package for circuit-level decoding of surface codes using belief propagation decoders.

## Features

- **Noisy Circuit Generation**: Generate rotated surface-code memory circuits with Stim
- **Detector Error Models**: Export detector error models (DEMs) for decoder development
- **Syndrome Sampling**: Pre-generate syndrome databases for training and testing
- **PyTorch BP Module**: Belief propagation on factor graphs with PyTorch backend
- **CLI Tools**: Command-line interface for dataset generation
- **Production Ready**: Modern Python packaging, CI/CD, comprehensive tests

## Prerequisites

- **Programming**: Python 3.10+, familiarity with NumPy and PyTorch
- **Mathematics**: Linear algebra, probability theory
- **QEC Background**: Stabilizer formalism, surface codes (helpful but not required)

## Key Concepts

### Detection Events

In circuit-level quantum error correction, we don't use raw syndrome measurements directly. Instead, we use **detection events** — the XOR (difference) between consecutive syndrome measurements.

**Why detection events instead of raw syndromes?**

In code-capacity noise (simplified model), syndromes directly indicate errors. But in circuit-level noise:
- Measurement errors exist and can randomly flip syndrome values
- A syndrome value of 1 could mean "real data error" or "measurement error"
- Detection events localize changes in space-time

```
Round 1 syndrome: [0, 0, 1, 0]
Round 2 syndrome: [0, 1, 1, 0]
                   ───────────
Detection event:  [0, 1, 0, 0]  ← Only the CHANGE matters
```

A detection event = 1 means "something happened in this space-time region" (data qubit error or measurement error). The decoder's job is to figure out which.

### Observable Flip

An **observable flip** indicates whether the logical qubit's value changed from initialization to final measurement.

For a surface code doing Z-memory:
- The logical observable Z̄ is a product of Z operators along a path
- Initialize in |0⟩_L (eigenstate of Z̄ with eigenvalue +1)
- If final measurement gives Z̄ = -1, that's an observable flip → logical error

**The decoding problem:**

```
Physical errors occur during circuit execution
                    ↓
Input:  Detection events (what we observe)
                    ↓
                 Decoder
                    ↓
Output: Predicted observable flip (0 or 1)
                    ↓
        Compare with actual observable flip
                    ↓
           Match → Success
           Mismatch → Logical error
```

In the Detector Error Model (DEM), errors are annotated with which detectors and observables they affect:

```
error(0.001) D0 D1       # Triggers detectors 0,1 but NOT the observable
error(0.001) D2 D3 L0    # Triggers detectors 2,3 AND flips logical observable L0
```

Errors that include `L0` form logical error chains — these are what the decoder must identify.

## Must-Read Papers

Before starting, please read these foundational papers:

### 1. BP+OSD Decoder (Foundational)
**"Decoding across the quantum LDPC code landscape"** - Roffe et al. (2020)
- Introduces BP+OSD, the key algorithm for this project
- [arXiv:2005.07016](https://arxiv.org/abs/2005.07016)

### 2. Improved BP for Surface Codes
**"Improved Belief Propagation Decoding Algorithms for Surface Codes"** - Chen et al. (2024)
- State-of-the-art BP improvements (Momentum-BP, AdaGrad-BP)
- [arXiv:2407.11523](https://arxiv.org/abs/2407.11523)

### 3. Circuit-Level Noise Decoding
**"Exact Decoding of Repetition Code under Circuit Level Noise"** - (2025)
- Explains circuit-level noise models and exact MLE decoding
- [arXiv:2501.03582](https://arxiv.org/abs/2501.03582)

### 4. Atom Loss Error Correction
**"Quantum Error Correction resilient against Atom Loss"** - (2024)
- Core paper for the atom loss extension
- [arXiv:2412.07841](https://arxiv.org/abs/2412.07841)

### 5. Decoder Review (Optional)
**"Decoding algorithms for surface codes"** - Quantum Journal (2024)
- Comprehensive review of all surface code decoders
- [Quantum 8:1498](https://quantum-journal.org/papers/q-2024-10-10-1498/)

## Installation

### With uv (recommended)

```bash
# Install the package
uv pip install bpdecoderplus

# For development
git clone https://github.com/TensorBFS/BPDecoderPlus.git
cd BPDecoderPlus
uv sync --dev
```

### With pip

```bash
# Install from source
git clone https://github.com/TensorBFS/BPDecoderPlus.git
cd BPDecoderPlus
pip install -e .[dev,docs]
```

## Quick Start

### Generate Noisy Circuits

```bash
# Generate surface code circuits with noise
generate-noisy-circuits --distance 3 5 7 --p 0.01 --rounds 3 5 7 --task z
```

### Python API

```python
from bpdecoderplus.pytorch_bp import (
    read_model_file,
    BeliefPropagation,
    belief_propagate,
    compute_marginals,
)

# Load UAI model
model = read_model_file("examples/simple_model.uai")

# Run belief propagation
bp = BeliefPropagation(model)
state, info = belief_propagate(bp)

# Get marginals
marginals = compute_marginals(state, bp)
print(marginals)
```

### Run Examples

```bash
python examples/simple_example.py
python examples/evidence_example.py
python examples/minimal_example.py
```

### Tanner Graph Decoding Tutorial

For a comprehensive walkthrough of using Belief Propagation on Tanner graphs for surface code decoding, see the [Tanner Graph Walkthrough](https://giggleliu.github.io/BPDecoderPlus/tanner_graph_walkthrough/) documentation.

The walkthrough covers:

- **Tanner graph theory** - Bipartite graph representation of parity check codes
- **Complete decoding pipeline** - From circuit generation to BP decoding and evaluation
- **Visualization** - Interactive graph structures and convergence analysis
- **Parameter tuning** - Damping, tolerance, and iteration optimization
- **Hands-on examples** - Runnable code with d=3 surface code datasets

**Run the companion script:**

```bash
uv run python examples/tanner_graph_walkthrough.py
```

## Project Structure

```
BPDecoderPlus/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── pyproject.toml              # Package configuration
├── Makefile                     # Build automation
├── src/bpdecoderplus/          # Main package
│   ├── __init__.py
│   ├── circuit.py              # Circuit generation with Stim
│   ├── dem.py                  # Detector error model export
│   ├── syndrome.py             # Syndrome sampling
│   ├── cli.py                  # Command-line interface
│   └── pytorch_bp/             # PyTorch BP module
│       ├── belief_propagation.py
│       └── uai_parser.py
├── datasets/                    # Sample datasets
│   ├── README.md
│   └── *.stim, *.dem, *.uai   # Generated data
├── docs/                        # MkDocs documentation
│   ├── index.md
│   ├── getting_started.md
│   ├── usage_guide.md
│   └── api_reference.md
├── examples/                    # Example scripts
│   ├── simple_example.py
│   ├── evidence_example.py
│   └── minimal_example.py
├── tests/                       # Test suite
│   ├── test_circuit.py
│   ├── test_dem.py
│   ├── test_syndrome.py
│   ├── test_bp_basic.py
│   └── test_integration.py
└── .github/workflows/          # CI/CD
    ├── test.yml
    └── docs.yml
```

## Command-Line Interface

The `generate-noisy-circuits` command generates surface code datasets:

```bash
# Basic usage
generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 5 --task z

# Multiple distances and error rates
generate-noisy-circuits --distance 3 5 7 --p 0.005 0.01 0.015 --rounds 3 5 7 --task z

# Specify output directory
generate-noisy-circuits --distance 5 --p 0.01 --rounds 5 --task z --output datasets/
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=bpdecoderplus --cov-report=html

# Run specific test file
uv run pytest tests/test_circuit.py
```

## Building Documentation

```bash
# Serve documentation locally
make docs-serve

# Build documentation
make docs
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install dependencies with uv |
| `make test` | Run unit tests |
| `make docs` | Build documentation |
| `make docs-serve` | Serve documentation locally |
| `make clean` | Remove generated files |
| `make help` | Show available targets |

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/TensorBFS/BPDecoderPlus.git
cd BPDecoderPlus

# Install with development dependencies
uv sync --dev

# Install pre-commit hooks (if configured)
uv run pre-commit install
```

### Code Style

This project uses:
- **Ruff** for linting and formatting
- **pytest** for testing
- **Type hints** for public APIs

## Resources

### Reference Implementations
- [Stim](https://github.com/quantumlib/Stim) - Fast stabilizer circuit simulator (used by this package)
- [bp_osd](https://github.com/quantumgizmos/bp_osd) - Python BP+OSD implementation
- [ldpc](https://github.com/quantumgizmos/ldpc) - LDPC decoder library
- [TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl) - Julia quantum error correction library

### Documentation
- [Stim Documentation](https://github.com/quantumlib/Stim/blob/main/doc/index.md)
- [Error Correction Zoo](https://errorcorrectionzoo.org/)
- [BPDecoderPlus Documentation](https://tensorbfs.github.io/BPDecoderPlus/)

## Troubleshooting

### Out of memory on large codes

Reduce code distance or number of samples:

```bash
# Use smaller codes for testing
generate-noisy-circuits --distance 3 --rounds 3 5 --p 0.01 --task z
```

### Import errors

Ensure the package is installed correctly:

```bash
# Reinstall in development mode
uv pip install -e .

# Or with pip
pip install -e .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use BPDecoderPlus in your research, please cite:

```bibtex
@software{bpdecoderplus2025,
  title = {BPDecoderPlus: Belief Propagation Decoders for Surface Codes},
  author = {Liu, Jinguo and Contributors},
  year = {2025},
  url = {https://github.com/TensorBFS/BPDecoderPlus}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses:
- [Stim](https://github.com/quantumlib/Stim) for efficient quantum circuit simulation
- [PyTorch](https://pytorch.org/) for the belief propagation implementation
- [uv](https://github.com/astral-sh/uv) for fast Python package management

## Related Projects

- [TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl) - Julia quantum error correction library
- [QEC Visualization](https://github.com/nzy1997/qec-thrust) - Quantum error correction visualization tools
