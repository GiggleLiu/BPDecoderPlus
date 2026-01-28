"""Tropical tensor network tools for MPE (independent package)."""

from .contraction import (
    build_contraction_tree,
    choose_order,
    contract_omeco_tree,
    contract_tree,
    get_omeco_tree,
)
from .mpe import mpe_tropical, mpe_tropical_approximate, recover_mpe_assignment
from .network import TensorNode, build_network
from .primitives import safe_log
from .tropical_einsum import (
    Backpointer,
    argmax_trace,
    match_rule,
    tropical_einsum,
    tropical_reduce_max,
)
from .utils import (
    Factor,
    UAIModel,
    build_tropical_factors,
    read_evidence_file,
    read_model_file,
    read_model_from_string,
)

# Approximate contraction methods
from .approximate import (
    TropicalMPS,
    TropicalMPO,
    boundary_contract,
    tropical_svd_approx,
    truncate_mps,
    ApproximateBackpointer,
    BoundaryContractionResult,
)
from .sweep import (
    sweep_contract,
    multi_direction_sweep,
    adaptive_sweep_contract,
    SweepDirection,
    SweepContractionResult,
    estimate_required_chi,
)

__all__ = [
    # Core types
    "Backpointer",
    "Factor",
    "TensorNode",
    "UAIModel",
    # Exact contraction
    "argmax_trace",
    "build_contraction_tree",
    "build_network",
    "build_tropical_factors",
    "choose_order",
    "contract_omeco_tree",
    "contract_tree",
    "get_omeco_tree",
    "match_rule",
    "mpe_tropical",
    "read_evidence_file",
    "read_model_file",
    "read_model_from_string",
    "recover_mpe_assignment",
    "safe_log",
    "tropical_einsum",
    "tropical_reduce_max",
    # Approximate contraction
    "TropicalMPS",
    "TropicalMPO",
    "boundary_contract",
    "tropical_svd_approx",
    "truncate_mps",
    "ApproximateBackpointer",
    "BoundaryContractionResult",
    "sweep_contract",
    "multi_direction_sweep",
    "adaptive_sweep_contract",
    "SweepDirection",
    "SweepContractionResult",
    "estimate_required_chi",
    "mpe_tropical_approximate",
]
