# Pipeline Illustration: From Circuit to Decoder

## Overview

This document illustrates the complete pipeline for quantum error correction with belief propagation decoding.

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Generate Noisy Circuit                         │
│  ─────────────────────────────────────────────────────  │
│  Input:  distance=3, rounds=3, p=0.01, task="z"        │
│  Output: Stim circuit (.stim file)                      │
│                                                          │
│  What it does:                                          │
│  - Creates a surface code circuit                       │
│  - Adds depolarizing noise (p=0.01) to all operations  │
│  - Includes syndrome measurements and logical readout   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Extract Detector Error Model (DEM)            │
│  ─────────────────────────────────────────────────────  │
│  Input:  Circuit                                        │
│  Output: DEM (.dem file)                                │
│                                                          │
│  What it does:                                          │
│  - Analyzes circuit to find all error mechanisms        │
│  - Determines which errors trigger which detectors      │
│  - Creates error(p) D1 D2 ... instructions             │
│                                                          │
│  Example DEM entry:                                     │
│    error(0.01) D0 D5 L0                                 │
│    ↑          ↑  ↑  ↑                                   │
│    │          │  │  └─ Flips logical observable        │
│    │          │  └──── Triggers detector 5              │
│    │          └─────── Triggers detector 0              │
│    └──────────────────── Probability 1%                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Build Parity Check Matrix H                   │
│  ─────────────────────────────────────────────────────  │
│  Input:  DEM                                            │
│  Output: H matrix, priors, obs_flip                     │
│                                                          │
│  What it does:                                          │
│  - Converts DEM to matrix form for BP decoder           │
│  - H[i,j] = 1 if error j triggers detector i           │
│  - priors[j] = probability of error j                   │
│  - obs_flip[j] = 1 if error j flips observable         │
│                                                          │
│  Example:                                               │
│    H = [[1, 0, 1, 0, ...],    ← Detector 0             │
│         [0, 1, 1, 0, ...],    ← Detector 1             │
│         ...]                                            │
│         ↑  ↑  ↑  ↑                                      │
│         Error mechanisms                                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Sample Syndromes                              │
│  ─────────────────────────────────────────────────────  │
│  Input:  Circuit, num_shots                             │
│  Output: Syndromes, Observables (.npz file)             │
│                                                          │
│  What it does:                                          │
│  - Runs circuit num_shots times                         │
│  - Records which detectors fire (syndrome)              │
│  - Records if logical qubit flips (observable)          │
│                                                          │
│  Example (1 shot):                                      │
│    Syndrome:    [0, 1, 1, 0, 0, 1, 0, ...]             │
│                  ↑  ↑  ↑        ↑                       │
│                  Detectors 1, 2, 5 fired                │
│    Observable:  0  (no logical error)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Decode (BP Decoder)                           │
│  ─────────────────────────────────────────────────────  │
│  Input:  H, priors, syndrome                            │
│  Output: Predicted observable flip                      │
│                                                          │
│  What it does:                                          │
│  - Uses belief propagation on H matrix                  │
│  - Infers most likely error pattern                     │
│  - Predicts if errors flip logical observable           │
│                                                          │
│  Algorithm:                                             │
│    1. Initialize beliefs from priors                    │
│    2. Pass messages between detectors and errors        │
│    3. Iterate until convergence                         │
│    4. Decode to most likely error pattern               │
│    5. Check if errors flip observable                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6: Evaluate                                       │
│  ─────────────────────────────────────────────────────  │
│  Input:  Predicted observable, Actual observable        │
│  Output: Success/Failure                                │
│                                                          │
│  Success criterion:                                     │
│    Predicted == Actual → Decoding succeeded            │
│    Predicted != Actual → Logical error                 │
│                                                          │
│  Metrics:                                               │
│    Logical error rate = (# failures) / (# shots)        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
                   ┌───────┐
                   │  END  │
                   └───────┘
```

## Data Flow Diagram

```
Circuit Parameters          Noisy Circuit              Detector Error Model
─────────────────          ──────────────             ────────────────────
distance = 3        ──→    .stim file         ──→     .dem file
rounds = 3                 (quantum ops)              (error→detector map)
p = 0.01                   + noise model
task = "z"
                                                              │
                                                              ▼
                                                       Parity Check Matrix
                                                       ───────────────────
                                                       H: (24 × 286)
                                                       priors: (286,)
                                                       obs_flip: (286,)
                                                              │
                                                              ▼
Syndrome Samples           ◄──────────────────────────  BP Decoder Input
────────────────                                        ────────────────
.npz file                                               H + syndrome
syndromes: (N, 24)                                           │
observables: (N,)                                            ▼
                                                       Decoder Output
                                                       ──────────────
                                                       predicted_obs: (N,)
                                                              │
                                                              ▼
                                                       Evaluation
                                                       ──────────
                                                       accuracy
                                                       logical_error_rate
```

## Key Concepts Explained

### 1. Circuit → DEM: Why do we need this?

**Problem:** The circuit describes quantum operations, but we need to know:
- Which errors can occur?
- Which detectors do they trigger?
- Do they cause logical errors?

**Solution:** The DEM extracts this information automatically.

**Example:**
```
Circuit says: "Apply CNOT gate with 1% depolarizing noise"
DEM says:     "This can cause 3 types of errors:
               - X error on control (triggers D0, D1)
               - Z error on target (triggers D2, D3)
               - Y error on both (triggers D0, D1, D2, D3, L0)"
```

### 2. DEM → H Matrix: Why convert to matrix form?

**Problem:** DEM is a list of error instructions. BP decoder needs matrix operations.

**Solution:** Convert to parity check matrix H.

**Example:**
```
DEM:
  error(0.01) D0 D1
  error(0.01) D1 D2
  error(0.01) D0 D2 L0

H Matrix:
     E0  E1  E2
D0 [ 1   0   1 ]
D1 [ 1   1   0 ]
D2 [ 0   1   1 ]

obs_flip: [0, 0, 1]  ← Only E2 flips observable
```

### 3. Circuit → Syndromes: What are we sampling?

**Problem:** We need training/test data for the decoder.

**Solution:** Run the circuit many times, record outcomes.

**What happens in one shot:**
1. Initialize qubits in |0⟩ state
2. Apply gates (with noise)
3. Measure syndromes (detectors fire if errors occurred)
4. Measure logical qubit (did it flip?)

**Example:**
```
Shot 1: Syndrome = [0,1,1,0,...], Observable = 0
Shot 2: Syndrome = [1,0,0,1,...], Observable = 0
Shot 3: Syndrome = [0,0,1,1,...], Observable = 1  ← Logical error!
...
```

### 4. H + Syndrome → Decoder: How does BP work?

**Input:**
- H matrix (which errors trigger which detectors)
- Syndrome (which detectors actually fired)
- Priors (how likely is each error?)

**Process:**
1. Start with prior beliefs about errors
2. Update beliefs based on which detectors fired
3. Pass messages between detectors and errors
4. Iterate until beliefs converge
5. Choose most likely error pattern

**Output:**
- Predicted error pattern
- Predicted observable flip

### 5. Predicted vs Actual: How do we evaluate?

**Success:** Predicted observable == Actual observable
- Decoder correctly identified the error pattern
- Logical qubit is protected

**Failure:** Predicted observable != Actual observable
- Decoder made a mistake
- Logical error occurred

**Metric:** Logical error rate = P(failure)

## File Formats

### Circuit File (.stim)
```
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
...
R 0 1 2 3
DEPOLARIZE1(0.01) 0 1 2 3
CX 0 1
DEPOLARIZE2(0.01) 0 1
...
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
```

### DEM File (.dem)
```
error(0.01) D0 D1
error(0.01) D1 D2
error(0.01) D0 D2 L0
...
```

### Syndrome Database (.npz)
```python
{
  'syndromes': array([[0, 1, 1, 0, ...],  # Shot 0
                      [1, 0, 0, 1, ...],  # Shot 1
                      ...]),
  'observables': array([0, 0, 1, ...]),
  'metadata': '{"distance": 3, "rounds": 3, ...}'
}
```

## Complete Example

### Input Parameters
```python
distance = 3
rounds = 3
p = 0.01
task = "z"
num_shots = 1000
```

### Generated Files
```
datasets/noisy_circuits/
├── sc_d3_r3_p0010_z.stim      # Circuit (2.5 KB)
├── sc_d3_r3_p0010_z.dem       # DEM (15 KB)
└── sc_d3_r3_p0010_z.npz       # Syndromes (3 KB)
```

### Data Shapes
```
Circuit:     ~100 operations
DEM:         24 detectors, 286 error mechanisms
H matrix:    (24, 286)
Syndromes:   (1000, 24)
Observables: (1000,)
```

### Expected Results
```
Detection rate:      ~3-5%
Logical error rate:  ~0.5-1% (without decoder)
                     ~0.1-0.3% (with good decoder)
```

## Running the Pipeline

### Command Line
```bash
# Generate everything at once
uv run generate-noisy-circuits \
  --distance 3 \
  --p 0.01 \
  --rounds 3 5 7 \
  --task z \
  --generate-dem \
  --generate-syndromes 1000
```

### Python API
```python
from bpdecoderplus.circuit import generate_circuit
from bpdecoderplus.dem import extract_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import sample_syndromes

# Generate circuit
circuit = generate_circuit(distance=3, rounds=3, p=0.01, task="z")

# Extract DEM and build H
dem = extract_dem(circuit)
H, priors, obs_flip = build_parity_check_matrix(dem)

# Sample syndromes
syndromes, observables = sample_syndromes(circuit, num_shots=1000)

# Now ready for decoder!
```

## Next Steps

1. **Implement BP Decoder** (Issue #3)
   - Use H matrix and priors
   - Implement message passing
   - Decode syndromes to error patterns

2. **Evaluate Decoder**
   - Compare predicted vs actual observables
   - Compute logical error rate
   - Compare with other decoders (MWPM, Neural)

3. **Optimize Performance**
   - Tune BP hyperparameters (damping, iterations)
   - Try BP+OSD for better accuracy
   - Scale to larger codes (d=5, d=7)
