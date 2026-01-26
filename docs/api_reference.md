## API Reference

This reference documents the public API for BPDecoderPlus.

---

## Detector Error Model (DEM)

Functions for working with Stim detector error models.

### DEM Extraction

- `extract_dem(circuit, decompose_errors=True) -> stim.DetectorErrorModel`
  Extract DEM from a Stim circuit.

- `load_dem(path) -> stim.DetectorErrorModel`
  Load a DEM from file.

- `save_dem(dem, path)`
  Save a DEM to file.

### Parity Check Matrix

- `build_parity_check_matrix(dem, split_by_separator=True, merge_hyperedges=True) -> (H, priors, obs_flip)`
  Build parity check matrix from DEM.
  - `H`: Binary matrix of shape `(n_detectors, n_errors)`
  - `priors`: Error probabilities of shape `(n_errors,)`
  - `obs_flip`: Observable flip indicators of shape `(n_errors,)`

- `build_decoding_uai(H, priors, syndrome) -> str`
  Build UAI model string for MAP decoding from parity check matrix and syndrome.

---

## Batch Decoders

High-performance batch decoding for multiple syndromes.

### BatchBPDecoder

```python
from bpdecoderplus.batch_bp import BatchBPDecoder

decoder = BatchBPDecoder(H, priors, device='cpu')
marginals = decoder.decode(syndromes, max_iter=30, damping=0.2)
```

- `BatchBPDecoder(H, priors, device='cpu')`
  Initialize batch BP decoder with parity check matrix and priors.

- `decode(syndromes, max_iter=100, damping=0.2) -> torch.Tensor`
  Decode batch of syndromes, returns marginal probabilities.

### BatchOSDDecoder

```python
from bpdecoderplus.batch_osd import BatchOSDDecoder

osd_decoder = BatchOSDDecoder(H, device='cpu')
solution = osd_decoder.solve(syndrome, marginals, osd_order=10)
```

- `BatchOSDDecoder(H, device='cpu')`
  Initialize OSD decoder with parity check matrix.

- `solve(syndrome, marginals, osd_order=10) -> np.ndarray`
  Find error pattern using Ordered Statistics Decoding.

---

## PyTorch BP (Low-level API)

Low-level belief propagation implementation for factor graphs.

### UAI Parsing

- `read_model_file(path, factor_eltype=torch.float64) -> UAIModel`
  Parse a UAI `.uai` model file.

- `read_model_from_string(content, factor_eltype=torch.float64) -> UAIModel`
  Parse a UAI model from an in-memory string.

- `read_evidence_file(path) -> Dict[int, int]`
  Parse a UAI `.evid` file and return evidence as 1-based indices.

### Data Structures

- `Factor(vars: List[int], values: torch.Tensor)`
  Container for a factor scope and its tensor.

- `UAIModel(nvars: int, cards: List[int], factors: List[Factor])`
  Holds all model metadata for BP.

### Belief Propagation

- `BeliefPropagation(uai_model: UAIModel)`
  Builds factor graph adjacency for BP.

- `initial_state(bp: BeliefPropagation) -> BPState`
  Initialize messages to uniform vectors.

- `collect_message(bp, state, normalize=True)`
  Update factor-to-variable messages in place.

- `process_message(bp, state, normalize=True, damping=0.2)`
  Update variable-to-factor messages in place.

- `belief_propagate(bp, max_iter=100, tol=1e-6, damping=0.2, normalize=True)`
  Run the full BP loop and return `(BPState, BPInfo)`.

- `compute_marginals(state, bp) -> Dict[int, torch.Tensor]`
  Compute marginal distributions after convergence.

- `apply_evidence(bp, evidence: Dict[int, int]) -> BeliefPropagation`
  Return a new BP object with evidence applied to factor tensors.

---

## Syndrome Database

Functions for generating and loading syndrome datasets.

- `sample_syndromes(circuit, num_shots, include_observables=True) -> (syndromes, observables)`
  Sample syndromes from a circuit.

- `save_syndrome_database(syndromes, observables, path, metadata=None)`
  Save syndrome database to `.npz` file.

- `load_syndrome_database(path) -> (syndromes, observables, metadata)`
  Load syndrome database from `.npz` file.

---

## Circuit Generation

Functions for generating surface code circuits.

- `generate_circuit(distance, rounds, p, task='z') -> stim.Circuit`
  Generate a rotated surface code memory circuit.

- `generate_filename(distance, rounds, p, task) -> str`
  Generate standardized filename for circuit parameters.
