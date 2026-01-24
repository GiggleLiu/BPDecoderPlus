## Tropical Tensor Network API Reference

Public APIs exported from `tropical_in_new/src`.

### UAI Parsing

- `read_model_file(path, factor_eltype=torch.float64) -> UAIModel`
  Parse a UAI `.uai` model file.

- `read_model_from_string(content, factor_eltype=torch.float64) -> UAIModel`
  Parse a UAI model from an in-memory string.

### Data Structures

- `Factor(vars: Tuple[int, ...], values: torch.Tensor)`
  Container for a factor scope and its tensor.

- `UAIModel(nvars: int, cards: List[int], factors: List[Factor])`
  Holds all model metadata for MPE.

### Primitives

- `safe_log(tensor: torch.Tensor) -> torch.Tensor`
  Convert potentials to log-domain; maps zeros to `-inf`.

- `tropical_einsum(a, b, index_map, track_argmax=True)`
  Binary contraction in max-plus semiring; returns `(values, backpointer)`.

- `argmax_trace(backpointer, assignment) -> Dict[int, int]`
  Decode assignments for eliminated variables from backpointer metadata.

### Network + Contraction

- `build_network(factors: Iterable[Factor]) -> list[TensorNode]`
  Convert factors into log-domain tensors with scopes.

- `choose_order(nodes, heuristic="omeco") -> list[int]`
  Select variable elimination order using `omeco`.

- `build_contraction_tree(order, nodes) -> ContractionTree`
  Construct a contraction plan from order and nodes.

- `contract_tree(tree, einsum_fn, track_argmax=True) -> TreeNode`
  Execute contractions and return the root node with backpointers.

### MPE

- `mpe_tropical(model, evidence=None, order=None)`
  Return `(assignment_dict, score, info)` where `score` is log-domain.

- `recover_mpe_assignment(root) -> Dict[int, int]`
  Recover assignments from a contraction tree root.
