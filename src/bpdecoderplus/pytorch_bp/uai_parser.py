"""
UAI file format parser for Belief Propagation.
"""

from typing import List, Dict
import torch


class Factor:
    """Factor class representing a factor in the factor graph."""
    
    def __init__(self, vars: List[int], values: torch.Tensor):
        """
        Args:
            vars: List of variable indices (1-based)
            values: Tensor of factor values with shape matching variable cardinalities
        """
        self.vars = tuple(vars)
        self.values = values
    
    def __repr__(self):
        return f"Factor(vars={self.vars}, shape={self.values.shape})"


class UAIModel:
    """UAI model class containing variables, cardinalities, and factors."""
    
    def __init__(self, nvars: int, cards: List[int], factors: List[Factor]):
        """
        Args:
            nvars: Number of variables
            cards: List of cardinalities for each variable
            factors: List of factors
        """
        self.nvars = nvars
        self.cards = cards
        self.factors = factors
    
    def __repr__(self):
        return f"UAIModel(nvars={self.nvars}, nfactors={len(self.factors)})"


def read_model_file(filepath: str, factor_eltype=torch.float64) -> UAIModel:
    """
    Parse UAI format model file.
    
    Args:
        filepath: Path to .uai file
        factor_eltype: Data type for factor values (default: torch.float64)
    
    Returns:
        UAIModel object
    """
    with open(filepath, 'r') as f:
        content = f.read()
    return read_model_from_string(content, factor_eltype=factor_eltype)


def read_model_from_string(content: str, factor_eltype=torch.float64) -> UAIModel:
    """
    Parse UAI model from string.
    
    Args:
        content: UAI file content as string
        factor_eltype: Data type for factor values
    
    Returns:
        UAIModel object
    """
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Parse header
    network_type = lines[0]  # MARKOV or BAYES
    if network_type not in ("MARKOV", "BAYES"):
        raise ValueError(
            f"Unsupported UAI network type: {network_type!r}. Expected 'MARKOV' or 'BAYES'."
        )
    nvars = int(lines[1])
    cards = [int(x) for x in lines[2].split()]
    ntables = int(lines[3])
    
    # Parse factor scopes
    scopes = []
    for i in range(ntables):
        parts = lines[4 + i].split()
        scope_size = int(parts[0])
        if len(parts) - 1 != scope_size:
            raise ValueError(
                f"Scope size mismatch on line {4 + i}: "
                f"declared {scope_size}, found {len(parts) - 1} variables."
            )
        scope = [int(x) + 1 for x in parts[1:]]  # Convert to 1-based
        scopes.append(scope)
    
    # Parse factor tables
    idx = 4 + ntables
    tokens: List[str] = []
    while idx < len(lines):
        tokens.extend(lines[idx].split())
        idx += 1
    cursor = 0

    factors: List[Factor] = []
    for scope in scopes:
        if cursor >= len(tokens):
            raise ValueError("Unexpected end of UAI factor table data.")
        nelements = int(tokens[cursor])
        cursor += 1
        values = torch.tensor(
            [float(x) for x in tokens[cursor:cursor + nelements]],
            dtype=factor_eltype
        )
        cursor += nelements

        # Reshape according to cardinalities in original scope order
        shape = tuple([cards[v - 1] for v in scope])
        values = values.reshape(shape)
        factors.append(Factor(scope, values))

    return UAIModel(nvars, cards, factors)


def read_evidence_file(filepath: str) -> Dict[int, int]:
    """
    Parse evidence file (.evid format).
    
    Args:
        filepath: Path to .evid file
    
    Returns:
        Dictionary mapping variable index (1-based) to observed value (0-based)
    """
    if not filepath:
        return {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return {}
    
    # Parse last line
    last_line = lines[-1].strip()
    parts = [int(x) for x in last_line.split()]
    
    nobsvars = parts[0]
    evidence = {}
    
    for i in range(nobsvars):
        var_idx = parts[1 + 2*i] + 1  # Convert to 1-based
        var_value = parts[2 + 2*i]
        evidence[var_idx] = var_value
    
    return evidence
