from pytorch_bp import (
    read_model_file,
    read_evidence_file,
    BeliefPropagation,
    belief_propagate,
    compute_marginals,
    apply_evidence,
)


def main():
    model = read_model_file("examples/simple_model.uai")
    evidence = read_evidence_file("examples/simple_model.evid")
    bp = apply_evidence(BeliefPropagation(model), evidence)
    state, info = belief_propagate(bp, max_iter=50, tol=1e-8, damping=0.1)
    print(info)

    marginals = compute_marginals(state, bp)
    for var_idx, marginal in marginals.items():
        print(f"Variable {var_idx} marginal: {marginal}")


if __name__ == "__main__":
    main()
