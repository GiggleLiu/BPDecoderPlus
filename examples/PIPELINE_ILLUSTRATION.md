# Getting Started: Generating Data for BP Decoding

## Overview

This guide shows you how to generate training and test data for belief propagation (BP) decoding of quantum error correction codes. The package handles all the complex quantum circuit details automatically - you just need to specify a few parameters.

## Quick Start

Generate a complete dataset with one command:

```bash
uv run generate-noisy-circuits \
  --distance 3 \
  --p 0.01 \
  --rounds 3 \
  --task z \
  --generate-dem \
  --generate-syndromes 1000
```

This creates three files in `datasets/noisy_circuits/`:
- `sc_d3_r3_p0010_z.stim` - The noisy quantum circuit
- `sc_d3_r3_p0010_z.dem` - The detector error model
- `sc_d3_r3_p0010_z.npz` - 1000 syndrome samples

## Understanding the Pipeline

The data generation happens in four steps:

1. Generate Noisy Circuit - Creates a surface code circuit with noise
2. Extract Detector Error Model (DEM) - Analyzes which errors trigger which detectors
3. Build Parity Check Matrix - Converts DEM to matrix form for BP decoder
4. Sample Syndromes - Runs the circuit many times to generate training data

## Step-by-Step Guide

### Step 1: Generate Noisy Circuit

The first step creates a quantum error correction circuit with realistic noise.

**What you specify:**
- `distance` - Size of the surface code (3, 5, 7, etc.)
- `rounds` - Number of error correction cycles
- `p` - Physical error rate (e.g., 0.01 = 1% error per operation)
- `task` - Type of logical operation ("z" for memory experiment)

**What it creates:**
A `.stim` file containing the complete quantum circuit with:
- Qubit initialization
- Syndrome measurement operations
- Noise on every gate
- Logical observable measurement

**Example:**
```bash
uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 --task z
```

Creates: `datasets/noisy_circuits/sc_d3_r3_p0010_z.stim`

### Step 2: Extract Detector Error Model (DEM)

The DEM tells us which errors trigger which syndrome detectors.

**Why we need it:**
The circuit describes quantum operations, but the decoder needs to know:
- What errors can occur?
- Which detectors fire when each error happens?
- Does the error flip the logical qubit?

**What it creates:**
A `.dem` file with entries like:
```
error(0.01) D0 D5 L0
```
This means: "There's a 1% chance of an error that triggers detectors 0 and 5, and flips the logical observable"

**How to generate:**
Add `--generate-dem` flag to the command above, or use:
```bash
uv run extract-dem datasets/noisy_circuits/sc_d3_r3_p0010_z.stim
```

### Step 3: Build Parity Check Matrix

The parity check matrix H is the mathematical representation needed for BP decoding.

**What it does:**
Converts the DEM into matrix form where:
- H[i,j] = 1 if error j triggers detector i
- priors[j] = probability of error j occurring
- obs_flip[j] = 1 if error j flips the logical observable

**Example:**
For distance-3, rounds-3 code:
- H matrix: (24 detectors × 286 error mechanisms)
- Each row represents a detector
- Each column represents a possible error

**In Python:**
```python
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix

dem = extract_dem(circuit)
H, priors, obs_flip = build_parity_check_matrix(dem)
```

### Step 4: Sample Syndromes

Generate training/test data by running the circuit many times.

**What happens:**
Each "shot" runs the full circuit:
1. Initialize qubits
2. Apply gates (errors occur randomly based on noise rate)
3. Measure syndromes (which detectors fire?)
4. Measure logical qubit (did it flip?)

**What you get:**
A `.npz` file containing:
- `syndromes`: Binary array (num_shots × num_detectors)
- `observables`: Binary array (num_shots,) - 1 means logical error
- `metadata`: Circuit parameters

**Example:**
```bash
uv run generate-noisy-circuits \
  --distance 3 --p 0.01 --rounds 3 --task z \
  --generate-syndromes 10000
```

Creates 10,000 syndrome samples for training/testing the decoder.

## What the BP Decoder Will Do (Coming Soon)

Once implemented, the BP decoder will take the generated data and perform quantum error correction.

### Input to Decoder

The decoder needs:
- H matrix (from DEM) - which errors trigger which detectors
- priors (from DEM) - probability of each error
- syndrome (from sampling) - which detectors actually fired

### How BP Decoding Works

Belief Propagation is an iterative algorithm that:

1. **Initialize beliefs** - Start with prior probabilities for each error
2. **Message passing** - Detectors and errors exchange information:
   - Detectors tell errors: "Given what I observed, how likely are you?"
   - Errors tell detectors: "Given my probability, what should you expect?"
3. **Iterate** - Repeat message passing until beliefs converge
4. **Decode** - Choose the most likely error pattern
5. **Predict** - Determine if errors flip the logical observable

### Expected API

```python
from bpdecoderplus.decoder import BPDecoder

# Load the data
H, priors, obs_flip = build_parity_check_matrix(dem)
data = np.load('datasets/noisy_circuits/sc_d3_r3_p0010_z.npz')
syndromes = data['syndromes']
actual_observables = data['observables']

# Create and run decoder
decoder = BPDecoder(H, priors, obs_flip)
predicted_observables = decoder.decode(syndromes)

# Evaluate performance
accuracy = (predicted_observables == actual_observables).mean()
logical_error_rate = 1 - accuracy
print(f"Logical error rate: {logical_error_rate:.4f}")
```

### Success Criteria

- **Correct decoding**: Predicted observable matches actual observable
- **Logical error**: Predicted observable differs from actual observable

The goal is to achieve a logical error rate much lower than the physical error rate (p), demonstrating the power of quantum error correction.

## File Formats Reference

### Circuit File (.stim)
Contains quantum operations and noise model. You don't need to read this directly - the package handles it.

### DEM File (.dem)
Lists all error mechanisms and their effects:
```
error(0.01) D0 D1      # Error triggers detectors 0 and 1
error(0.01) D1 D2      # Error triggers detectors 1 and 2
error(0.01) D0 D2 L0   # Error triggers detectors 0, 2 and flips logical
```

### Syndrome Database (.npz)
NumPy archive with three components:
```python
data = np.load('sc_d3_r3_p0010_z.npz')
syndromes = data['syndromes']      # Shape: (num_shots, num_detectors)
observables = data['observables']  # Shape: (num_shots,)
metadata = data['metadata']        # JSON string with parameters
```

## Complete Example Workflow

### 1. Generate all data at once

```bash
uv run generate-noisy-circuits \
  --distance 3 \
  --p 0.01 \
  --rounds 3 \
  --task z \
  --generate-dem \
  --generate-syndromes 10000
```

This creates:
```
datasets/noisy_circuits/
├── sc_d3_r3_p0010_z.stim   # Circuit (~2.5 KB)
├── sc_d3_r3_p0010_z.dem    # DEM (~15 KB)
└── sc_d3_r3_p0010_z.npz    # Syndromes (~30 KB for 10k shots)
```

### 2. Use in Python

```python
import numpy as np
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix
import stim

# Load circuit and extract DEM
circuit = stim.Circuit.from_file('datasets/noisy_circuits/sc_d3_r3_p0010_z.stim')
dem = extract_dem(circuit)

# Build parity check matrix
H, priors, obs_flip = build_parity_check_matrix(dem)
print(f"H matrix shape: {H.shape}")  # (24, 286) for d=3, r=3

# Load syndrome data
data = np.load('datasets/noisy_circuits/sc_d3_r3_p0010_z.npz')
syndromes = data['syndromes']        # (10000, 24)
observables = data['observables']    # (10000,)

# Ready for BP decoder!
# decoder = BPDecoder(H, priors, obs_flip)  # Coming soon
# predictions = decoder.decode(syndromes)
```

### 3. Generate multiple datasets

For comprehensive testing, generate data at different noise levels:

```bash
# Low noise
uv run generate-noisy-circuits --distance 3 --p 0.001 --rounds 3 --task z \
  --generate-dem --generate-syndromes 10000

# Medium noise
uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 --task z \
  --generate-dem --generate-syndromes 10000

# High noise
uv run generate-noisy-circuits --distance 3 --p 0.05 --rounds 3 --task z \
  --generate-dem --generate-syndromes 10000
```

## Expected Performance

For a distance-3 surface code with p=0.01:

- **Number of detectors**: 24 (8 per round × 3 rounds)
- **Number of error mechanisms**: ~286
- **Detection rate**: 3-5% of shots have non-zero syndrome
- **Logical error rate (no decoder)**: ~0.5-1%
- **Logical error rate (with BP decoder)**: ~0.1-0.3% (expected)

The BP decoder should reduce the logical error rate by 3-10x compared to no decoding.

## Next Steps

1. **Generate your first dataset** using the Quick Start command
2. **Explore the data** by loading the .npz file and examining syndromes
3. **Wait for BP decoder implementation** (Issue #3)
4. **Evaluate decoder performance** on your generated datasets

## Advanced Options

### Generate multiple rounds

```bash
uv run generate-noisy-circuits \
  --distance 3 \
  --p 0.01 \
  --rounds 3 5 7 \
  --task z \
  --generate-dem \
  --generate-syndromes 10000
```

This creates datasets for 3, 5, and 7 rounds to study how performance scales.

### Different code distances

```bash
# Small code (fast, lower threshold)
uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 --task z \
  --generate-dem --generate-syndromes 10000

# Larger code (slower, higher threshold)
uv run generate-noisy-circuits --distance 5 --p 0.01 --rounds 5 --task z \
  --generate-dem --generate-syndromes 10000
```

## Troubleshooting

**Q: The command is slow**
A: Syndrome sampling is the slowest part. Start with fewer shots (e.g., 1000) for testing.

**Q: Files are large**
A: The .npz files scale with num_shots. Use 1000-10000 shots for development, more for final evaluation.

**Q: What's a good starting point?**
A: Use distance=3, rounds=3, p=0.01, and 10000 shots. This runs quickly and gives good statistics.

## References

- See `examples/MINIMUM_WORKING_EXAMPLE.md` for a minimal code example
- See `docs/SYNDROME_DATABASE.md` for detailed data format documentation
- See `docs/DEM_GENERATION.md` for DEM extraction details
