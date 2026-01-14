# Circuit-Level Syndrome Datasets

This directory contains circuit-level syndrome datasets for quantum error correction decoder benchmarking. The data is generated using [Stim](https://github.com/quantumlib/Stim), Google's fast stabilizer circuit simulator.

## Table of Contents

1. [Overview](#overview)
2. [Circuit-Level vs Code-Capacity Noise](#circuit-level-vs-code-capacity-noise)
3. [Key Concepts](#key-concepts)
4. [Data Generation Process](#data-generation-process)
5. [File Formats](#file-formats)
6. [Dataset Structure](#dataset-structure)
7. [Usage Examples](#usage-examples)

---

## Overview

Circuit-level decoding is more realistic than code-capacity decoding because it models:
- **Gate errors**: Errors during CNOT, Hadamard, and other gates
- **Measurement errors**: Faulty syndrome measurements
- **Reset errors**: Imperfect qubit initialization
- **Idle errors**: Decoherence while qubits wait

This makes the decoding problem significantly harder but more representative of real quantum hardware.

---

## Circuit-Level vs Code-Capacity Noise

| Aspect | Code-Capacity | Circuit-Level |
|--------|---------------|---------------|
| Error location | Data qubits only | Data + ancilla qubits |
| Time dimension | Single snapshot | Multiple rounds |
| Measurement | Perfect | Noisy (can flip) |
| Syndrome | Direct parity check | Detection events (differences) |
| Decoding graph | 2D (spatial) | 3D (space + time) |

### Why Detection Events?

In circuit-level noise, measurement errors can cause the raw syndrome to flip randomly. To handle this, we use **detection events** instead of raw syndromes:

```
Detection Event = Syndrome[round t] ⊕ Syndrome[round t-1]
```

A detection event fires (=1) when something changes between rounds. This converts measurement errors into localized events that decoders can handle.

---

## Key Concepts

### 1. Detectors

A **detector** is a parity check that should be deterministic (always 0) in the absence of errors. In a surface code memory experiment:

- Each stabilizer measurement defines a detector
- Detectors compare measurements across time (detection events)
- For `d` rounds of a distance-`d` code: `num_detectors ≈ d × (d²-1)`

### 2. Logical Observables

A **logical observable** tracks whether a logical error has occurred. For a surface code memory experiment:

- Usually 1 observable (the stored logical qubit)
- Observable = 1 means a logical error happened
- This is the ground truth label for decoder evaluation

### 3. Detector Error Model (DEM)

The DEM describes the probabilistic relationship between errors and detectors:

```
error(p) D0 D1      # Error with probability p triggers detectors 0 and 1
error(p) D2 L0      # Error triggers detector 2 AND flips logical observable
```

The DEM is a **Tanner graph** that decoders use to find the most likely error pattern.

---

## Data Generation Process

### Step 1: Generate Noisy Circuit

```python
import stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",  # Code type
    distance=5,                        # Code distance
    rounds=5,                          # Syndrome extraction rounds
    after_clifford_depolarization=0.001,  # Gate error rate
    before_measure_flip_probability=0.001, # Measurement error rate
    after_reset_flip_probability=0.001,    # Reset error rate
)
```

This generates a complete circuit with:
- Qubit initialization (resets)
- Repeated syndrome extraction rounds
- CNOT gates between data and ancilla qubits
- Ancilla measurements
- Detector and observable annotations

### Step 2: Extract Detector Error Model

```python
dem = circuit.detector_error_model(decompose_errors=True)
```

The DEM is extracted by:
1. Propagating each possible error through the circuit
2. Recording which detectors it triggers
3. Computing the probability of each error mechanism

`decompose_errors=True` breaks hyperedges (errors triggering >2 detectors) into graphlike edges for matching-based decoders.

### Step 3: Sample Detection Events

```python
sampler = circuit.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(
    num_shots=100000,
    separate_observables=True
)
```

This simulates the noisy circuit many times, recording:
- **Detection events**: Which detectors fired (syndrome data)
- **Observable flips**: Whether a logical error occurred (labels)

---

## File Formats

### `.stim` - Stim Circuit

Human-readable circuit description:

```
QUBIT_COORDS(1, 1) 1          # Qubit 1 at position (1,1)
R 1 3 5 8 10                   # Reset qubits
X_ERROR(0.001) 1 3 5 8 10      # Apply X errors with p=0.001
H 2 11 16                      # Hadamard gates
CX 2 3 16 17                   # CNOT gates
M 2 9 11                       # Measurements
DETECTOR(2, 2, 0) rec[-7]      # Detector from measurement record
OBSERVABLE_INCLUDE(0) rec[-1]  # Logical observable
```

### `.dem` - Detector Error Model

Probabilistic error model for decoders:

```
error(0.00193) D0              # Single-detector error
error(0.00193) D0 D1           # Two-detector error (edge)
error(0.00411) D1 L0           # Error affecting logical observable
error(0.00053) D2 D3 ^ D6      # Decomposed hyperedge
detector(1, 1, 0) D0           # Detector coordinates
```

**Syntax:**
- `D#`: Detector index
- `L#`: Logical observable index
- `^`: Separator for hyperedge decomposition
- Coordinates help visualize the decoding graph

### `_events.01` - Detection Events (Text)

One line per shot, one character per detector:

```
000000000000000000000000    # No detectors fired
000000010000000000100000    # Detectors 7 and 19 fired
110000000000000000000001    # Detectors 0, 1, and 23 fired
```

- `0` = detector did not fire
- `1` = detector fired (something changed)
- Line length = number of detectors

### `_events.b8` - Detection Events (Binary)

Same data as `.01` but packed into bytes (8 detectors per byte, little-endian). More space-efficient for large datasets.

```python
import numpy as np
events = np.fromfile("events.b8", dtype=np.uint8)
events = np.unpackbits(events, bitorder='little')
events = events.reshape(num_shots, -1)[:, :num_detectors]
```

### `_obs.01` - Observable Flips

One line per shot, one character per observable:

```
0    # No logical error
0    # No logical error
1    # Logical error occurred!
0    # No logical error
```

This is the **ground truth** for evaluating decoder accuracy.

### `_metadata.json` - Dataset Metadata

```json
{
  "code": "surface_code:rotated_memory_z",
  "distance": 5,
  "rounds": 5,
  "p_error": 0.001,
  "noise_model": "depolarizing",
  "num_shots": 100000,
  "num_detectors": 120,
  "num_observables": 1,
  "logical_error_rate": 0.0558,
  "seed": 42,
  "files": {
    "circuit": "surface_d5_r5_p0.0010.stim",
    "dem": "surface_d5_r5_p0.0010.dem",
    "events_01": "surface_d5_r5_p0.0010_events.01",
    "events_b8": "surface_d5_r5_p0.0010_events.b8",
    "observables": "surface_d5_r5_p0.0010_obs.01"
  }
}
```

---

## Dataset Structure

### Naming Convention

```
surface_d{distance}_r{rounds}_p{error_rate}.*
```

Example: `surface_d5_r5_p0.0010` = distance 5, 5 rounds, p=0.001

### Generated Datasets

| Dataset | Distance | Rounds | Detectors | Error Rate | Shots |
|---------|----------|--------|-----------|------------|-------|
| d3_r3_p0.0010 | 3 | 3 | 24 | 0.1% | 10,000 |
| d3_r3_p0.0050 | 3 | 3 | 24 | 0.5% | 10,000 |
| d3_r3_p0.0100 | 3 | 3 | 24 | 1.0% | 10,000 |
| d5_r5_p0.0010 | 5 | 5 | 120 | 0.1% | 10,000 |
| d5_r5_p0.0050 | 5 | 5 | 120 | 0.5% | 10,000 |
| d5_r5_p0.0100 | 5 | 5 | 120 | 1.0% | 10,000 |

### Detector Count Formula

For a rotated surface code with `d` distance and `r` rounds:
```
num_detectors = (d² - 1) × r + boundary_corrections
```

---

## Usage Examples

### 1. Load Data in Python

```python
import numpy as np
import json

# Load metadata
with open("surface_d5_r5_p0.0010_metadata.json") as f:
    meta = json.load(f)

# Load detection events (.01 format)
with open("surface_d5_r5_p0.0010_events.01") as f:
    events = np.array([[int(c) for c in line.strip()]
                       for line in f], dtype=np.uint8)

# Load observable flips
with open("surface_d5_r5_p0.0010_obs.01") as f:
    labels = np.array([int(line.strip()) for line in f], dtype=np.uint8)

print(f"Events shape: {events.shape}")  # (10000, 120)
print(f"Labels shape: {labels.shape}")  # (10000,)
```

### 2. Use with PyMatching

```python
import stim
from pymatching import Matching

# Load DEM
dem = stim.DetectorErrorModel.from_file("surface_d5_r5_p0.0010.dem")

# Create decoder
matcher = Matching.from_detector_error_model(dem)

# Decode
predictions = matcher.decode_batch(events)

# Evaluate accuracy
errors = np.sum(predictions != labels)
print(f"Logical error rate: {errors / len(labels):.4f}")
```

### 3. Use with Tesseract Decoder

```bash
./tesseract \
    --dem surface_d5_r5_p0.0010.dem \
    --in surface_d5_r5_p0.0010_events.01 \
    --in-format 01 \
    --out predictions.01 \
    --out-format 01
```

### 4. Train a Neural Network Decoder

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Prepare data
X = torch.tensor(events, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Train your model...
```

---

## Generating More Data

### Quick Mode (10k shots, d=3,5)
```bash
make circuit-data-quick
```

### Full Mode (100k shots, d=3,5,7,9)
```bash
make circuit-data
```

### Custom Parameters
```bash
make circuit-data-custom D=7 R=7 P=0.002 N=500000
```

### Direct Python Usage
```bash
python python/generate_circuit_data.py \
    --distance 9 \
    --rounds 9 \
    --p-error 0.005 \
    --shots 1000000 \
    --noise-model depolarizing \
    --output-dir benchmark/circuit_data
```

---

## References

1. **Stim**: [github.com/quantumlib/Stim](https://github.com/quantumlib/Stim) - Fast stabilizer circuit simulator
2. **Tesseract**: [github.com/quantumlib/tesseract-decoder](https://github.com/quantumlib/tesseract-decoder) - Search-based QEC decoder
3. **PyMatching**: [github.com/oscarhiggott/PyMatching](https://github.com/oscarhiggott/PyMatching) - MWPM decoder
4. **DEM Format**: [Stim DEM documentation](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)

---

*Generated by BPDecoderPlus*
