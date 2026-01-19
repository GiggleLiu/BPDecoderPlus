"""
Belief Propagation (BP) algorithm implementation using PyTorch.
"""

from typing import List, Dict, Tuple, Optional
import torch
from copy import deepcopy

from .uai_parser import UAIModel, Factor


class BeliefPropagation:
    """Belief Propagation object for factor graphs."""
    
    def __init__(self, uai_model: UAIModel):
        """
        Construct BP object from UAI model.
        
        Args:
            uai_model: Parsed UAI model
        """
        self.nvars = uai_model.nvars
        self.factors = uai_model.factors
        self.cards = uai_model.cards
        
        # Build mapping: t2v (factor -> variables), v2t (variable -> factors)
        self.t2v = [list(factor.vars) for factor in self.factors]
        self.v2t = self._build_v2t()
    
    def _build_v2t(self) -> List[List[int]]:
        """Build variable-to-factors mapping."""
        v2t = [[] for _ in range(self.nvars)]
        for factor_idx, vars_list in enumerate(self.t2v):
            for var in vars_list:
                if 0 < var <= self.nvars:  # Ensure valid index
                    v2t[var - 1].append(factor_idx)  # Convert to 0-based
        return v2t
    
    def num_tensors(self) -> int:
        """Return number of factors (tensors)."""
        return len(self.t2v)
    
    def num_variables(self) -> int:
        """Return number of variables."""
        return self.nvars


class BPState:
    """BP state storing messages."""
    
    def __init__(self, message_in: List[List[torch.Tensor]], 
                 message_out: List[List[torch.Tensor]]):
        """
        Args:
            message_in: Incoming messages from factors to variables
            message_out: Outgoing messages from variables to factors
        """
        self.message_in = message_in
        self.message_out = message_out


class BPInfo:
    """BP convergence information."""
    
    def __init__(self, converged: bool, iterations: int):
        self.converged = converged
        self.iterations = iterations
    
    def __repr__(self):
        status = "converged" if self.converged else "not converged"
        return f"BPInfo({status}, iterations={self.iterations})"


def initial_state(bp: BeliefPropagation) -> BPState:
    """
    Initialize BP message state with all ones vectors.
    
    Args:
        bp: BeliefPropagation object
    
    Returns:
        BPState with initialized messages
    """
    message_in = []
    message_out = []
    
    for var_idx in range(bp.nvars):
        var_messages_in = []
        var_messages_out = []
        
        for factor_idx in bp.v2t[var_idx]:
            card = bp.cards[var_idx]
            msg = torch.ones(card, dtype=torch.float64)
            var_messages_in.append(msg.clone())
            var_messages_out.append(msg.clone())
        
        message_in.append(var_messages_in)
        message_out.append(var_messages_out)
    
    return BPState(deepcopy(message_in), message_out)


def _compute_factor_to_var_message(
    factor_tensor: torch.Tensor,
    incoming_messages: List[torch.Tensor],
    target_var_idx: int
) -> torch.Tensor:
    """
    Compute factor to variable message using tensor contraction.
    
    μ_{f→x}(x) = Σ_{other vars} [φ_f(...) * Π_{y≠x} μ_{y→f}]
    
    Args:
        factor_tensor: Factor tensor with shape (d1, d2, ..., dn)
        incoming_messages: List of incoming messages, one for each variable in factor
        target_var_idx: Index of target variable (0-based) in factor's variable list
    
    Returns:
        Output message vector with shape (d_target,)
    """
    ndims = len(incoming_messages)

    if ndims == 1:
        return factor_tensor.clone()

    # Multiply factor tensor by incoming messages (excluding target) and sum out dims.
    result = factor_tensor
    for dim in range(ndims):
        if dim == target_var_idx:
            continue
        msg = incoming_messages[dim]
        shape = [1] * ndims
        shape[dim] = msg.shape[0]
        result = result * msg.view(*shape)

    # Sum over all dimensions except target
    sum_dims = [dim for dim in range(ndims) if dim != target_var_idx]
    if sum_dims:
        result = result.sum(dim=tuple(sum_dims))
    return result


def collect_message(bp: BeliefPropagation, state: BPState, normalize: bool = True) -> None:
    """
    Collect and update messages from factors to variables.
    
    μ_{f→x}(x) = Σ[φ_f(...) * Π μ_{y→f}]
    
    Args:
        bp: BeliefPropagation object
        state: BPState (modified in place)
        normalize: Whether to normalize messages
    """
    for factor_idx, factor in enumerate(bp.factors):
        # Get incoming messages from variables to this factor
        incoming_messages = []
        for var in factor.vars:
            var_idx_0based = var - 1
            # Find position of this factor in v2t[var_idx_0based]
            factor_pos = bp.v2t[var_idx_0based].index(factor_idx)
            incoming_messages.append(state.message_out[var_idx_0based][factor_pos])
        
        # Compute outgoing message to each variable
        for var_pos, var in enumerate(factor.vars):
            var_idx_0based = var - 1
            # Compute message from factor to variable
            outgoing_msg = _compute_factor_to_var_message(
                factor.values,
                incoming_messages,
                var_pos
            )
            
            # Normalize
            if normalize:
                msg_sum = outgoing_msg.sum()
                if msg_sum > 0:
                    outgoing_msg = outgoing_msg / msg_sum
            
            # Update message_in
            factor_pos = bp.v2t[var_idx_0based].index(factor_idx)
            state.message_in[var_idx_0based][factor_pos] = outgoing_msg


def process_message(
    bp: BeliefPropagation,
    state: BPState,
    normalize: bool = True,
    damping: float = 0.2,
) -> None:
    r"""
    Process and update messages from variables to factors.

    μ_{x→f}(x) = Π_{g∈ne(x)\setminus f} μ_{g→x}(x)

    Args:
        bp: BeliefPropagation object
        state: BPState (modified in place)
        normalize: Whether to normalize messages
        damping: Damping factor for message update
    """
    for var_idx_0based in range(bp.nvars):
        for factor_pos, factor_idx in enumerate(bp.v2t[var_idx_0based]):
            # Compute product of all incoming messages except from current factor
            product = torch.ones(bp.cards[var_idx_0based], dtype=torch.float64)
            
            for other_factor_pos, other_factor_idx in enumerate(bp.v2t[var_idx_0based]):
                if other_factor_pos != factor_pos:
                    product = product * state.message_in[var_idx_0based][other_factor_pos]
            
            # Normalize
            if normalize:
                msg_sum = product.sum()
                if msg_sum > 0:
                    product = product / msg_sum
            
            # Damping update
            old_message = state.message_out[var_idx_0based][factor_pos].clone()
            state.message_out[var_idx_0based][factor_pos] = (
                damping * old_message + (1 - damping) * product
            )


def _check_convergence(message_new: List[List[torch.Tensor]], 
                      message_old: List[List[torch.Tensor]], 
                      tol: float = 1e-6) -> bool:
    """
    Check if messages have converged.
    
    Args:
        message_new: Current iteration messages
        message_old: Previous iteration messages
        tol: Convergence tolerance
    
    Returns:
        True if converged, False otherwise
    """
    for var_msgs_new, var_msgs_old in zip(message_new, message_old):
        for msg_new, msg_old in zip(var_msgs_new, var_msgs_old):
            # Compute L1 distance
            diff = torch.abs(msg_new - msg_old).sum()
            if diff > tol:
                return False
    return True


def belief_propagate(bp: BeliefPropagation, 
                    max_iter: int = 100, 
                    tol: float = 1e-6, 
                    damping: float = 0.2,
                    normalize: bool = True) -> Tuple[BPState, BPInfo]:
    """
    Run Belief Propagation algorithm main loop.
    
    Args:
        bp: BeliefPropagation object
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        damping: Damping factor
        normalize: Whether to normalize messages
    
    Returns:
        Tuple of (BPState, BPInfo)
    """
    state = initial_state(bp)
    
    for iteration in range(max_iter):
        # Save previous messages for convergence check
        prev_message_in = deepcopy(state.message_in)
        
        # Update messages
        collect_message(bp, state, normalize=normalize)
        process_message(bp, state, normalize=normalize, damping=damping)
        
        # Check convergence
        if _check_convergence(state.message_in, prev_message_in, tol=tol):
            return state, BPInfo(converged=True, iterations=iteration + 1)
    
    return state, BPInfo(converged=False, iterations=max_iter)


def compute_marginals(state: BPState, bp: BeliefPropagation) -> Dict[int, torch.Tensor]:
    """
    Compute marginal probabilities from converged BP state.
    
    b(x) = (1/Z) * Π_{f∈ne(x)} μ_{f→x}(x)
    
    Args:
        state: Converged BPState
        bp: BeliefPropagation object
    
    Returns:
        Dictionary mapping variable index (1-based) to marginal probability distribution
    """
    marginals = {}
    
    for var_idx_0based in range(bp.nvars):
        # Product of all incoming messages
        product = torch.ones(bp.cards[var_idx_0based], dtype=torch.float64)
        
        for msg in state.message_in[var_idx_0based]:
            product = product * msg
        
        # Normalize to get probability distribution
        msg_sum = product.sum()
        if msg_sum > 0:
            product = product / msg_sum
        
        marginals[var_idx_0based + 1] = product  # Convert to 1-based indexing
    
    return marginals


def apply_evidence(bp: BeliefPropagation, evidence: Dict[int, int]) -> BeliefPropagation:
    """
    Apply evidence constraints to BP object.
    
    Modifies factor tensors to zero out non-evidence assignments.
    
    Args:
        bp: Original BeliefPropagation object
        evidence: Dictionary mapping variable index (1-based) to value (0-based)
    
    Returns:
        New BeliefPropagation object with evidence applied
    """
    # Create new factors with evidence constraints
    new_factors = []
    
    for factor in bp.factors:
        # Create mask for evidence constraints
        factor_tensor = factor.values.clone()
        
        # Apply evidence constraints
        for var_pos, var in enumerate(factor.vars):
            if var in evidence:
                evid_value = evidence[var]
                # Create slice that zeros out non-evidence values
                slices = [slice(None)] * len(factor.vars)
                slices[var_pos] = evid_value
                
                # Zero out all non-evidence assignments
                mask = torch.ones_like(factor_tensor)
                for i in range(factor_tensor.shape[var_pos]):
                    if i != evid_value:
                        slices_mask = slices.copy()
                        slices_mask[var_pos] = i
                        mask[tuple(slices_mask)] = 0
                
                factor_tensor = factor_tensor * mask
        
        new_factors.append(Factor(factor.vars, factor_tensor))
    
    # Create new UAIModel with modified factors
    from .uai_parser import UAIModel
    new_uai = UAIModel(bp.nvars, bp.cards, new_factors)
    
    return BeliefPropagation(new_uai)
