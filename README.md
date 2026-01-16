# BPDecoderPlus: Quantum Error Correction with Belief Propagation

A winter school project on circuit-level decoding of surface codes using belief propagation and integer programming decoders, with extensions for atom loss in neutral atom quantum computers.

## Project Goals

| Level | Task | Description |
|-------|------|-------------|
| **Basic** | MLE Decoder | Reproduce the integer programming (MLE) decoder as the baseline |
| **Challenge** | Atom Loss | Handle atom loss errors in neutral atom systems |
| **Extension** | QEC Visualization | https://github.com/nzy1997/qec-thrust |

Note: we also want to explore the boundary of vibe coding, which may lead to a scipost paper.

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

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd BPDecoderPlus
```

### 2. Install Dependencies

```bash
# Install Julia dependencies
make setup-julia

# Install Python dependencies (for visualization)
make setup-python

# Or install both at once
make setup
```

### 3. Run Tests

```bash
make test
```

### 4. Quick Demo

```bash
# Run a quick benchmark to verify everything works
make quick
```

Or interactively in Julia:
```julia
using BPDecoderPlus

# Quick benchmark with IP decoder
result = quick_benchmark(distance=5, p=0.05, n_trials=100)
println("Logical error rate: ", result["logical_error_rate"])

# Compare decoders
compare_decoders([3, 5], [0.02, 0.05, 0.08], 100)
```

## Project Structure

```
BPDecoderPlus/
├── README.md                    # This file
├── Makefile                     # Build automation
├── Project.toml                 # Julia dependencies
├── src/
│   └── BPDecoderPlus.jl         # Main module (uses TensorQEC.jl)
├── benchmark/
│   ├── generate_data.jl         # Generate benchmark data
│   ├── run_benchmarks.jl        # Run decoder timing tests
│   └── data/                    # Output data (JSON)
├── python/
│   ├── requirements.txt         # Python dependencies
│   └── visualize.py             # Plotting scripts
├── results/
│   └── plots/                   # Generated plots
├── test/
│   └── runtests.jl              # Unit tests
└── note/
    └── belief_propagation_qec_plan.tex
```

## Available Decoders

| Decoder | Symbol | Description |
|---------|--------|-------------|
| IP (MLE) | `:IP` | Integer programming decoder - finds minimum weight error |
| BP | `:BP` | Belief propagation without post-processing |
| BP+OSD | `:BPOSD` | BP with Ordered Statistics Decoding post-processing |
| Matching | `:Matching` | Minimum weight perfect matching (via TensorQEC) |

## Tasks

### Basic Task: Reproduce MLE Decoder

1. Understand how surface codes work using TensorQEC
2. Run the IP decoder on different code distances
3. Generate threshold plots (logical vs physical error rate)
4. Analyze how performance scales with code distance

```julia
using BPDecoderPlus

# Create surface code
code = SurfaceCode(5, 5)
tanner = CSSTannerGraph(code)

# Create error model
em = iid_error(0.03, 0.03, 0.03, 25)

# Decode with IP
decoder = IPDecoder()
compiled = compile(decoder, tanner)

ep = random_error_pattern(em)
syn = syndrome_extraction(ep, tanner)
result = decode(compiled, syn)

# Check for logical error
x_err, z_err, y_err = check_logical_error(tanner, ep, result.error_pattern)
```

### Challenge Task: Implement and Compare BP Decoder

1. Run the BP decoder and compare with IP
2. Understand when BP fails and when OSD helps
3. Compare threshold performance
4. Analyze timing differences

```julia
# Compare decoders
results = compare_decoders(
    [3, 5, 7],           # distances
    0.01:0.02:0.15,       # error rates
    1000;                 # trials
    decoders=[:IP, :BP, :BPOSD]
)
```

### Extension: Handle Atom Loss

1. Understand the atom loss model
2. Compare naive vs loss-aware decoding
3. Analyze threshold degradation with loss

```julia
# Simulate atom loss
loss_model = AtomLossModel(0.02)  # 2% loss rate
tanner, lost_qubits = apply_atom_loss(tanner, loss_model)

# Decode with loss information
result = decode_with_atom_loss(tanner, syn, decoder, lost_qubits)
```

## Running Benchmarks

### Full Benchmark (takes longer, more accurate)
```bash
make benchmark
make visualize
```

### Quick Benchmark (for testing)
```bash
make quick
```

### Individual Decoder Test
```bash
make benchmark-ip
make benchmark-bp
make benchmark-bposd
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install all dependencies |
| `make test` | Run unit tests |
| `make benchmark` | Generate full benchmark data |
| `make quick` | Quick benchmark (fewer trials) |
| `make visualize` | Generate plots from data |
| `make all` | Full pipeline: test -> benchmark -> visualize |
| `make clean` | Remove generated files |
| `make help` | Show available targets |

## Output Plots

After running benchmarks and visualization, you'll find:

- `results/plots/threshold_ip.png` - IP decoder threshold curve
- `results/plots/threshold_bp.png` - BP decoder threshold curve
- `results/plots/threshold_bposd.png` - BP+OSD decoder threshold curve
- `results/plots/decoder_comparison.png` - Side-by-side comparison
- `results/plots/timing_comparison.png` - Decoding speed comparison
- `results/plots/atom_loss.png` - Effect of atom loss
- `results/plots/scalability.png` - Scalability analysis

## Evaluation Criteria

Your submission will be evaluated on:

1. **Correctness** (40%): Do your decoders produce valid corrections?
2. **Analysis** (30%): Quality of threshold plots and performance analysis
3. **Code Quality** (20%): Clean, documented, well-tested code
4. **Extension** (10%): Atom loss handling or other improvements

## Resources

### Core Library
- [TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl) - QEC library we build on

### Reference Implementations
- [bp_osd](https://github.com/quantumgizmos/bp_osd) - Python BP+OSD implementation
- [ldpc](https://github.com/quantumgizmos/ldpc) - LDPC decoder library

### Documentation
- [TensorQEC Documentation](https://nzy1997.github.io/TensorQEC.jl/dev/)
- [Error Correction Zoo](https://errorcorrectionzoo.org/)

## Troubleshooting

### TensorQEC fails to precompile
This is a known issue with YaoSym. It still works at runtime:
```julia
# Ignore precompilation warnings and proceed
using TensorQEC
```

### Out of memory on large codes
Reduce code distance or number of trials:
```julia
# Use smaller codes for testing
results = quick_benchmark(distance=3, n_trials=100)
```

### Visualization fails
Ensure Python dependencies are installed:
```bash
pip install -r python/requirements.txt
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This project is built on [TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl) by nzy1997.
