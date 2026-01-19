import itertools
import torch


def exact_marginals(model, evidence=None):
    evidence = evidence or {}
    assignments = list(itertools.product(*[range(c) for c in model.cards]))
    weights = []
    for assignment in assignments:
        if any(assignment[var_idx - 1] != val for var_idx, val in evidence.items()):
            weights.append(0.0)
            continue
        weight = 1.0
        for factor in model.factors:
            idx = tuple(assignment[v - 1] for v in factor.vars)
            weight *= float(factor.values[idx])
        weights.append(weight)
    total = sum(weights)
    marginals = {}
    for var_idx, card in enumerate(model.cards):
        values = []
        for value in range(card):
            mass = 0.0
            for assignment, weight in zip(assignments, weights):
                if assignment[var_idx] == value:
                    mass += weight
            values.append(mass / total if total > 0 else 0.0)
        marginals[var_idx + 1] = torch.tensor(values, dtype=torch.float64)
    return marginals
