## PyTorch BP API Reference

This reference documents the public API exported from `bpdecoderplus.pytorch_bp`.

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
