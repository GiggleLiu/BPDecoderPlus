# BPDecoderPlus

[![Tests](https://github.com/GiggleLiu/BPDecoderPlus/actions/workflows/test.yml/badge.svg)](https://github.com/GiggleLiu/BPDecoderPlus/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/BPDecoderPlus/branch/main/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/BPDecoderPlus)

**Quantum Error Correction with Belief Propagation**

!!! note "Work in Progress"
    This WIP project is for AI + Quantum winter school training.

A winter school project on circuit-level decoding of surface codes using belief propagation and integer programming decoders, with extensions for atom loss in neutral atom quantum computers.

## Project Goals

| Level | Task | Description |
|-------|------|-------------|
| **Basic** | MLE Decoder | Reproduce the integer programming (MLE) decoder as the baseline |
| **Challenge** | Atom Loss | Handle atom loss errors in neutral atom systems |
| **Extension** | QEC Visualization | https://github.com/nzy1997/qec-thrust |

!!! info
    We also want to explore the boundary of vibe coding, which may lead to a scipost paper.

## Learning Objectives

After completing this project, students will:

- Understand surface code structure and syndrome extraction
- Implement and compare different decoding algorithms
- Analyze decoder performance through threshold plots
- Learn about practical QEC challenges (atom loss, circuit-level noise)

## Prerequisites

- **Programming**: Julia basics, familiarity with Python for plotting
- **Mathematics**: Linear algebra, probability theory
- **QEC Background**: Stabilizer formalism, surface codes (helpful but not required)

## Quick Start

Install the package:

```bash
# Clone the repository
git clone https://github.com/GiggleLiu/BPDecoderPlus.git
cd BPDecoderPlus

# Install dependencies
make setup

# Run tests
make test
```

Generate a dataset:

```bash
python -m bpdecoderplus.cli \
  --distance 3 \
  --p 0.01 \
  --rounds 3 5 7 \
  --generate-dem \
  --generate-syndromes 1000
```

## Features

### Python Module (bpdecoderplus)

- **Noisy Circuit Generation**: Create surface code circuits with realistic noise models
- **Detector Error Model (DEM)**: Extract error models for belief propagation
- **UAI Format Support**: Export to UAI format for probabilistic inference with TensorInference.jl
- **Syndrome Database**: Generate training datasets from circuit simulations
- **PyTorch BP Implementation**: Belief propagation solver for factor graphs

### Julia Module (TensorQEC.jl Integration)

- Multiple decoder implementations (IP, BP, BP+OSD, Matching)
- Comprehensive benchmarking tools
- Performance visualization

## Documentation

- [Getting Started](getting_started.md) - Quick start guide and pipeline overview
- [Usage Guide](usage_guide.md) - Detailed usage examples
- [API Reference](api_reference.md) - Complete API documentation
- [Mathematical Description](mathematical_description.md) - Mathematical background

## Available Decoders

| Decoder | Symbol | Description |
|---------|--------|-------------|
| IP (MLE) | `:IP` | Integer programming decoder - finds minimum weight error |
| BP | `:BP` | Belief propagation without post-processing |
| BP+OSD | `:BPOSD` | BP with Ordered Statistics Decoding post-processing |
| Matching | `:Matching` | Minimum weight perfect matching (via TensorQEC) |

## Resources

### Core Library
- [TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl) - QEC library we build on

### Reference Implementations
- [bp_osd](https://github.com/quantumgizmos/bp_osd) - Python BP+OSD implementation
- [ldpc](https://github.com/quantumgizmos/ldpc) - LDPC decoder library

### Documentation
- [TensorQEC Documentation](https://nzy1997.github.io/TensorQEC.jl/dev/)
- [Error Correction Zoo](https://errorcorrectionzoo.org/)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This project is built on [TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl) by nzy1997.
