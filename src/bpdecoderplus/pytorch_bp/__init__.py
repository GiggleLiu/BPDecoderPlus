"""
PyTorch Belief Propagation (BP) submodule for approximate inference.
"""

from .uai_parser import (
    read_model_file,
    read_model_from_string,
    read_evidence_file,
    UAIModel,
    Factor
)

from .belief_propagation import (
    BeliefPropagation,
    BPState,
    BPInfo,
    initial_state,
    collect_message,
    process_message,
    belief_propagate,
    compute_marginals,
    apply_evidence
)

__all__ = [
    # UAI parsing
    'read_model_file',
    'read_model_from_string',
    'read_evidence_file',
    'UAIModel',
    'Factor',
    # Belief Propagation
    'BeliefPropagation',
    'BPState',
    'BPInfo',
    'initial_state',
    'collect_message',
    'process_message',
    'belief_propagate',
    'compute_marginals',
    'apply_evidence',
]
