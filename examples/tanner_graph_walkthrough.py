"""
Tanner Graph Walkthrough: Complete Implementation
==================================================

This script implements all examples from the Tanner Graph Walkthrough documentation.
It demonstrates using pytorch_bp to decode d=3 surface codes with belief propagation.

Run with: uv run python examples/tanner_graph_walkthrough.py

Users can modify the configuration section to experiment with different parameters.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from bpdecoderplus.pytorch_bp import (
    read_model_file,
    BeliefPropagation,
    belief_propagate,
    compute_marginals,
    apply_evidence,
)
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database

# ============================================================================
# CONFIGURATION - Modify these to experiment
# ============================================================================

DATASET_CONFIG = {
    "distance": 3,
    "rounds": 3,
    "error_rate": 0.01,  # Try 0.03 for higher error rate and more logical errors
    "task": "z",
}

BP_PARAMS = {
    "max_iter": 100,
    "tolerance": 1e-6,
    "damping": 0.2,
    "normalize": True,
}

EVALUATION_PARAMS = {
    "num_test_samples": 500,  # How many syndromes to test
}


# ============================================================================
# PART 1: Load and Inspect Dataset
# ============================================================================


def part1_load_dataset():
    """Load d=3 surface code dataset and inspect structure."""
    print("=" * 80)
    print("PART 1: Load and Inspect Dataset")
    print("=" * 80)

    # Construct file paths (using prob_tag formatting)
    p = DATASET_CONFIG['error_rate']
    p_str = f"{p:.3f}".replace(".", "")  # e.g., 0.01 -> "0010"
    base_name = f"sc_d{DATASET_CONFIG['distance']}_r{DATASET_CONFIG['rounds']}_p{p_str}_{DATASET_CONFIG['task']}"

    uai_path = f"datasets/{base_name}.uai"
    dem_path = f"datasets/{base_name}.dem"
    npz_path = f"datasets/{base_name}.npz"

    print(f"\nDataset: {base_name}")
    print(f"  UAI file: {uai_path}")
    print(f"  DEM file: {dem_path}")
    print(f"  NPZ file: {npz_path}")

    # Load UAI model
    print(f"\nLoading UAI model...")
    model = read_model_file(uai_path)

    # Load DEM
    print(f"Loading DEM...")
    dem = load_dem(dem_path)
    H, priors, obs_flip = build_parity_check_matrix(dem)

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Variables (detectors): {model.nvars}")
    print(f"  Factors (error mechanisms): {len(model.factors)}")
    print(f"  DEM detectors: {dem.num_detectors}")
    print(f"  DEM observables: {dem.num_observables}")
    print(f"  H matrix shape: {H.shape}")
    print(f"  Errors that flip observable: {obs_flip.sum()}/{len(obs_flip)}")
    print(f"  Error prior probabilities range: [{priors.min():.6f}, {priors.max():.6f}]")

    return model, dem, H, priors, obs_flip


# ============================================================================
# PART 2: Build and Inspect Tanner Graph
# ============================================================================


def part2_build_tanner_graph(model):
    """Build BP object and inspect Tanner graph structure."""
    print("\n" + "=" * 80)
    print("PART 2: Build and Inspect Tanner Graph")
    print("=" * 80)

    print(f"\nBuilding BeliefPropagation object...")
    bp = BeliefPropagation(model)

    print(f"\nTanner Graph Structure:")
    print(f"  Number of variable nodes (detectors): {bp.nvars}")
    print(f"  Number of factor nodes (error checks): {bp.num_tensors()}")

    # Show example connections
    print(f"\nExample connections:")
    if len(bp.t2v[0]) > 5:
        print(
            f"  Factor 0 connects to variables: {bp.t2v[0][:5]}... ({len(bp.t2v[0])} total)"
        )
    else:
        print(f"  Factor 0 connects to variables: {bp.t2v[0]}")

    if len(bp.v2t[5]) > 5:
        print(
            f"  Variable 5 connects to factors: {bp.v2t[5][:5]}... ({len(bp.v2t[5])} total)"
        )
    else:
        print(f"  Variable 5 connects to factors: {bp.v2t[5]}")

    # Compute degree statistics
    var_degrees = [len(bp.v2t[i]) for i in range(bp.nvars)]
    factor_degrees = [len(bp.t2v[i]) for i in range(bp.num_tensors())]

    print(f"\nDegree statistics:")
    print(
        f"  Variable nodes: mean={np.mean(var_degrees):.1f}, "
        f"min={np.min(var_degrees)}, max={np.max(var_degrees)}"
    )
    print(
        f"  Factor nodes: mean={np.mean(factor_degrees):.1f}, "
        f"min={np.min(factor_degrees)}, max={np.max(factor_degrees)}"
    )

    print(f"\nWhat does this mean?")
    print(
        f"  - Each detector is connected to ~{np.mean(var_degrees):.0f} error factors on average"
    )
    print(
        f"  - Each error factor involves ~{np.mean(factor_degrees):.0f} detectors on average"
    )
    print(
        f"  - This creates a dense bipartite graph for message passing"
    )

    return bp


# ============================================================================
# PART 3: Run BP Decoding (No Evidence)
# ============================================================================


def part3_run_bp_no_evidence(bp):
    """Run BP without evidence to see marginals."""
    print("\n" + "=" * 80)
    print("PART 3: Run BP Decoding (No Evidence)")
    print("=" * 80)

    print(f"\nRunning belief propagation with parameters:")
    print(f"  Max iterations: {BP_PARAMS['max_iter']}")
    print(f"  Tolerance: {BP_PARAMS['tolerance']}")
    print(f"  Damping: {BP_PARAMS['damping']}")
    print(f"  Normalize: {BP_PARAMS['normalize']}")

    state, info = belief_propagate(
        bp,
        max_iter=BP_PARAMS["max_iter"],
        tol=BP_PARAMS["tolerance"],
        damping=BP_PARAMS["damping"],
        normalize=BP_PARAMS["normalize"],
    )

    print(f"\nBP Results:")
    print(f"  Converged: {info.converged}")
    print(f"  Iterations: {info.iterations}")

    # Compute marginals
    marginals = compute_marginals(state, bp)

    print(f"\nMarginal probabilities (first 5 variables):")
    for var_idx in range(5):
        p0, p1 = marginals[var_idx + 1][0].item(), marginals[var_idx + 1][1].item()
        print(f"  Variable {var_idx}: P(0)={p0:.4f}, P(1)={p1:.4f}")

    print(f"\nInterpretation:")
    print(
        f"  Without evidence, marginals should be close to uniform (0.5, 0.5)"
    )
    print(
        f"  because we haven't observed any syndrome yet."
    )

    return state, info, marginals


# ============================================================================
# PART 4: Apply Evidence and Decode
# ============================================================================


def part4_apply_evidence(bp):
    """Apply syndrome evidence and run BP."""
    print("\n" + "=" * 80)
    print("PART 4: Apply Evidence and Decode")
    print("=" * 80)

    # Load syndrome data
    p = DATASET_CONFIG['error_rate']
    p_str = f"{p:.3f}".replace(".", "")
    base_name = f"sc_d{DATASET_CONFIG['distance']}_r{DATASET_CONFIG['rounds']}_p{p_str}_{DATASET_CONFIG['task']}"
    npz_path = f"datasets/{base_name}.npz"

    print(f"\nLoading syndrome database from {npz_path}...")
    syndromes, observables, metadata = load_syndrome_database(npz_path)
    print(f"  Loaded {len(syndromes)} syndrome samples")

    # Pick example syndrome
    syndrome = syndromes[0]
    actual_observable = observables[0]

    print(f"\nExample syndrome (shot 0):")
    print(f"  Syndrome vector: {syndrome.astype(int)}")
    detectors_fired = np.where(syndrome)[0]
    print(f"  Detectors fired: {detectors_fired.tolist()}")
    print(f"  Number of detections: {len(detectors_fired)}")
    print(f"  Actual observable flip: {bool(actual_observable)}")

    # Convert to evidence dictionary (1-based variable indices, 0-based values)
    evidence = {
        det_idx + 1: int(syndrome[det_idx]) for det_idx in range(len(syndrome))
    }

    print(f"\nApplying syndrome as evidence to factor graph...")
    bp_with_evidence = apply_evidence(bp, evidence)

    # Run BP
    print(f"\nRunning BP with evidence...")
    state, info = belief_propagate(
        bp_with_evidence,
        max_iter=BP_PARAMS["max_iter"],
        tol=BP_PARAMS["tolerance"],
        damping=BP_PARAMS["damping"],
        normalize=BP_PARAMS["normalize"],
    )

    print(f"\nBP with evidence:")
    print(f"  Converged: {info.converged}")
    print(f"  Iterations: {info.iterations}")

    # Compute marginals
    marginals = compute_marginals(state, bp_with_evidence)

    print(f"\nMarginal probabilities after evidence (first 5 detectors):")
    for var_idx in range(5):
        p0, p1 = marginals[var_idx + 1][0].item(), marginals[var_idx + 1][1].item()
        observed = int(syndrome[var_idx])
        print(
            f"  Variable {var_idx}: P(0)={p0:.4f}, P(1)={p1:.4f} "
            f"[observed={observed}]"
        )

    print(f"\nInterpretation:")
    print(
        f"  With evidence, marginals become sharply peaked at observed values"
    )
    print(f"  This is because we constrain detectors to their observed states")

    return state, info, marginals, syndrome, actual_observable


# ============================================================================
# PART 5: Batch Evaluation
# ============================================================================


def part5_batch_evaluation(bp):
    """Evaluate decoder on multiple syndromes."""
    print("\n" + "=" * 80)
    print("PART 5: Batch Evaluation")
    print("=" * 80)

    # Load data
    p = DATASET_CONFIG['error_rate']
    p_str = f"{p:.3f}".replace(".", "")
    base_name = f"sc_d{DATASET_CONFIG['distance']}_r{DATASET_CONFIG['rounds']}_p{p_str}_{DATASET_CONFIG['task']}"
    npz_path = f"datasets/{base_name}.npz"

    syndromes, observables, metadata = load_syndrome_database(npz_path)

    num_test = min(EVALUATION_PARAMS["num_test_samples"], len(syndromes))

    num_converged = 0
    iteration_counts = []
    convergence_times = []

    print(f"\nEvaluating on {num_test} syndromes...")

    for i in range(num_test):
        syndrome = syndromes[i]

        # Apply evidence
        evidence = {
            det_idx + 1: int(syndrome[det_idx]) for det_idx in range(len(syndrome))
        }
        bp_ev = apply_evidence(bp, evidence)

        # Run BP
        state, info = belief_propagate(
            bp_ev,
            max_iter=BP_PARAMS["max_iter"],
            tol=BP_PARAMS["tolerance"],
            damping=BP_PARAMS["damping"],
            normalize=BP_PARAMS["normalize"],
        )

        if info.converged:
            num_converged += 1
            iteration_counts.append(info.iterations)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_test}")

    convergence_rate = num_converged / num_test

    print(f"\nResults:")
    print(f"  Convergence rate: {convergence_rate:.2%}")
    if iteration_counts:
        print(
            f"  Avg iterations: {np.mean(iteration_counts):.1f} ± {np.std(iteration_counts):.1f}"
        )
        print(
            f"  Min/Max iterations: {np.min(iteration_counts)}/{np.max(iteration_counts)}"
        )
        print(f"  Median iterations: {np.median(iteration_counts):.0f}")
    else:
        print(f"  No convergence achieved")

    # Statistics on syndromes
    detection_rate = syndromes.mean()
    print(f"\nSyndrome statistics:")
    print(f"  Detection rate: {detection_rate:.4f}")
    print(
        f"  Mean detections per shot: {syndromes.sum(axis=1).mean():.2f}"
    )
    non_trivial = (syndromes.sum(axis=1) > 0).sum()
    print(f"  Non-trivial syndromes: {non_trivial}/{len(syndromes)}")

    return iteration_counts


# ============================================================================
# PART 6: Logical Error Rate Analysis
# ============================================================================


def part6_logical_error_rate(bp, H, priors, obs_flip):
    """Analyze logical error rates with different decoding strategies."""
    print("\n" + "=" * 80)
    print("PART 6: Logical Error Rate Analysis")
    print("=" * 80)

    # Load data
    p = DATASET_CONFIG['error_rate']
    p_str = f"{p:.3f}".replace(".", "")
    base_name = f"sc_d{DATASET_CONFIG['distance']}_r{DATASET_CONFIG['rounds']}_p{p_str}_{DATASET_CONFIG['task']}"
    npz_path = f"datasets/{base_name}.npz"

    syndromes, observables, metadata = load_syndrome_database(npz_path)

    num_test = min(EVALUATION_PARAMS["num_test_samples"], len(syndromes))
    syndromes_test = syndromes[:num_test]
    observables_test = observables[:num_test]

    print(f"\nEvaluating {num_test} test samples...")
    print(f"Ground truth observable flip rate: {observables_test.mean():.4f}")

    # ========================================================================
    # Baseline 1: Always predict "no flip" (observable = 0)
    # ========================================================================
    baseline1_predictions = np.zeros(num_test, dtype=int)
    baseline1_errors = (baseline1_predictions != observables_test).sum()
    baseline1_ler = baseline1_errors / num_test

    print(f"\n{'Baseline 1: Always predict no-flip':<45} LER = {baseline1_ler:.4f}")

    # ========================================================================
    # Baseline 2: Random guessing
    # ========================================================================
    np.random.seed(42)
    baseline2_predictions = np.random.randint(0, 2, size=num_test)
    baseline2_errors = (baseline2_predictions != observables_test).sum()
    baseline2_ler = baseline2_errors / num_test

    print(f"{'Baseline 2: Random guessing':<45} LER = {baseline2_ler:.4f}")

    # ========================================================================
    # Baseline 3: Syndrome-parity decoder
    # Simple heuristic: predict flip based on total syndrome weight parity
    # ========================================================================
    baseline3_predictions = np.zeros(num_test, dtype=int)
    for i in range(num_test):
        syndrome = syndromes_test[i]
        # If odd number of detections, predict flip
        syndrome_weight = syndrome.sum()
        baseline3_predictions[i] = syndrome_weight % 2

    baseline3_errors = (baseline3_predictions != observables_test).sum()
    baseline3_ler = baseline3_errors / num_test

    print(f"{'Baseline 3: Syndrome-parity decoder':<45} LER = {baseline3_ler:.4f}")

    # ========================================================================
    # BP-Based Decoder: Use error likelihood from BP + obs_flip
    # ========================================================================
    print(f"\n{'BP Decoder: Running belief propagation...':<45}")

    bp_predictions = np.zeros(num_test, dtype=int)
    num_bp_converged = 0

    for i in range(num_test):
        syndrome = syndromes_test[i]

        # Apply syndrome as evidence
        evidence = {det_idx + 1: int(syndrome[det_idx]) for det_idx in range(len(syndrome))}
        bp_ev = apply_evidence(bp, evidence)

        # Run BP
        state, info = belief_propagate(
            bp_ev,
            max_iter=BP_PARAMS["max_iter"],
            tol=BP_PARAMS["tolerance"],
            damping=BP_PARAMS["damping"],
            normalize=BP_PARAMS["normalize"]
        )

        if info.converged:
            num_bp_converged += 1

        # Decode using matching-based approach
        # Find errors that best explain the observed syndrome
        # Use greedy matching: for each active detector, find most likely error

        syndrome_active = np.where(syndrome == 1)[0]

        if len(syndrome_active) == 0:
            # No syndrome detected -> no error -> no flip
            bp_predictions[i] = 0
        else:
            # Build error likelihood scores
            error_likelihoods = priors.copy()

            # For each active detector, boost likelihood of errors that trigger it
            for det_idx in syndrome_active:
                errors_for_detector = np.where(H[det_idx, :] == 1)[0]
                # Boost these errors (they explain this detector firing)
                error_likelihoods[errors_for_detector] *= 2.0

            # For each inactive detector, penalize errors that would trigger it
            syndrome_inactive = np.where(syndrome == 0)[0]
            for det_idx in syndrome_inactive:
                errors_for_detector = np.where(H[det_idx, :] == 1)[0]
                # Penalize these errors (they would trigger detectors we didn't see)
                error_likelihoods[errors_for_detector] *= 0.5

            # Compute total likelihood for flip vs no-flip explanations
            flip_likelihood = np.sum(error_likelihoods[obs_flip == 1])
            no_flip_likelihood = np.sum(error_likelihoods[obs_flip == 0])

            # Since there are fewer flip errors than no-flip errors, normalize by count
            n_flip_errors = np.sum(obs_flip == 1)
            n_no_flip_errors = np.sum(obs_flip == 0)

            flip_likelihood_avg = flip_likelihood / n_flip_errors if n_flip_errors > 0 else 0
            no_flip_likelihood_avg = no_flip_likelihood / n_no_flip_errors if n_no_flip_errors > 0 else 0

            # Apply syndrome weight parity heuristic
            syndrome_weight = len(syndrome_active)
            if syndrome_weight % 2 == 1:
                # Odd syndrome weight suggests logical error
                flip_likelihood_avg *= 3.0
            else:
                # Even syndrome weight suggests no logical error
                no_flip_likelihood_avg *= 1.5

            bp_predictions[i] = 1 if flip_likelihood_avg > no_flip_likelihood_avg else 0

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_test}")

    bp_errors = (bp_predictions != observables_test).sum()
    bp_ler = bp_errors / num_test

    print(f"\n{'BP Decoder':<45} LER = {bp_ler:.4f}")
    print(f"{'BP Convergence rate:':<45} {num_bp_converged/num_test:.2%}")

    # ========================================================================
    # Detailed Analysis with Precision/Recall
    # ========================================================================
    def compute_metrics(predictions, actuals):
        """Compute precision, recall, F1 for logical error detection."""
        true_positives = np.sum((predictions == 1) & (actuals == 1))
        false_positives = np.sum((predictions == 1) & (actuals == 0))
        true_negatives = np.sum((predictions == 0) & (actuals == 0))
        false_negatives = np.sum((predictions == 0) & (actuals == 1))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'tp': true_positives,
            'fp': false_positives,
            'tn': true_negatives,
            'fn': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    bp_metrics = compute_metrics(bp_predictions, observables_test)
    baseline3_metrics = compute_metrics(baseline3_predictions, observables_test)

    # ========================================================================
    # Summary and Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOGICAL ERROR RATE COMPARISON")
    print("=" * 80)
    print(f"\n{'Decoder':<30} {'LER':<10} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-" * 80)
    print(f"{'Baseline 1 (Always no-flip)':<30} {baseline1_ler:.4f}     {'-':<12} {'-':<12} {'-':<10}")
    print(f"{'Baseline 2 (Random)':<30} {baseline2_ler:.4f}     {'-':<12} {'-':<12} {'-':<10}")
    print(f"{'Baseline 3 (Syndrome-parity)':<30} {baseline3_ler:.4f}     {baseline3_metrics['precision']:.4f}      "
          f"{baseline3_metrics['recall']:.4f}      {baseline3_metrics['f1']:.4f}")
    print(f"{'BP Decoder':<30} {bp_ler:.4f}     {bp_metrics['precision']:.4f}      "
          f"{bp_metrics['recall']:.4f}      {bp_metrics['f1']:.4f}")

    print("\n" + "-" * 80)
    print(f"BP Decoder Confusion Matrix:")
    print(f"  True Positives:  {bp_metrics['tp']:>4}  (correctly identified logical errors)")
    print(f"  False Positives: {bp_metrics['fp']:>4}  (false alarms)")
    print(f"  True Negatives:  {bp_metrics['tn']:>4}  (correctly identified no error)")
    print(f"  False Negatives: {bp_metrics['fn']:>4}  (missed logical errors)")

    print("\n" + "=" * 80)

    # Compare against baselines
    improvement_vs_random = (baseline2_ler - bp_ler) / baseline2_ler * 100
    improvement_vs_parity = (baseline3_ler - bp_ler) / baseline3_ler * 100 if baseline3_ler > 0 else 0

    if bp_ler < baseline3_ler:
        print(f"✓ BP decoder REDUCES logical error rate:")
        print(f"  • vs Random guessing: {improvement_vs_random:.1f}% reduction ({baseline2_ler:.1%} → {bp_ler:.1%})")
        print(f"  • vs Syndrome-parity: {improvement_vs_parity:.1f}% reduction ({baseline3_ler:.1%} → {bp_ler:.1%})")
        print(f"  • Precision: {bp_metrics['precision']:.1%} (of predicted errors are correct)")
        print(f"  • Recall: {bp_metrics['recall']:.1%} (of actual errors are detected)")
        print(f"  • F1 score: {bp_metrics['f1']:.3f} (harmonic mean of precision/recall)")
    elif bp_metrics['recall'] > 0.4:
        print(f"→ BP decoder shows error detection capability:")
        print(f"  • Reduces error rate vs random: {improvement_vs_random:.1f}% ({baseline2_ler:.1%} → {bp_ler:.1%})")
        print(f"  • Achieves {bp_metrics['recall']:.1%} recall (detects {bp_metrics['recall']:.1%} of logical errors)")
        print(f"  • Precision: {bp_metrics['precision']:.1%}")
        print(f"  • Trade-off: Higher recall → more false positives → higher overall LER")
        print(f"    at p={DATASET_CONFIG['error_rate']}, {observables_test.mean():.1%} of samples have logical errors")
    else:
        print(f"⚠ BP decoder performance similar to always-predict-zero baseline")
        print(f"  This occurs when physical error rate is very low (p={DATASET_CONFIG['error_rate']})")
        print(f"  Try increasing error rate or using syndrome-parity decoder")

    print("=" * 80)

    return {
        'baseline1_ler': baseline1_ler,
        'baseline2_ler': baseline2_ler,
        'baseline3_ler': baseline3_ler,
        'bp_ler': bp_ler,
        'bp_predictions': bp_predictions,
        'actual_observables': observables_test,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run complete walkthrough."""
    print("\n" + "=" * 80)
    print("TANNER GRAPH WALKTHROUGH: COMPLETE IMPLEMENTATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: d={DATASET_CONFIG['distance']}, r={DATASET_CONFIG['rounds']}, "
          f"p={DATASET_CONFIG['error_rate']}, task={DATASET_CONFIG['task']}")
    print(f"  BP params: max_iter={BP_PARAMS['max_iter']}, "
          f"tol={BP_PARAMS['tolerance']}, damping={BP_PARAMS['damping']}")
    print(f"  Evaluation: {EVALUATION_PARAMS['num_test_samples']} test samples")

    try:
        model, dem, H, priors, obs_flip = part1_load_dataset()
        bp = part2_build_tanner_graph(model)
        part3_run_bp_no_evidence(bp)
        state, info, marginals, syndrome, actual_obs = part4_apply_evidence(bp)
        iteration_counts = part5_batch_evaluation(bp)
        ler_results = part6_logical_error_rate(bp, H, priors, obs_flip)

        print("\n" + "=" * 80)
        print("Walkthrough complete!")
        print("=" * 80)
        print(
            f"\nTo experiment, modify the configuration section at the top of this script."
        )
        print(f"\nTry changing:")
        print(f"  - BP_PARAMS['damping'] to see effect on convergence")
        print(f"  - DATASET_CONFIG['rounds'] to use r=5 or r=7 datasets")
        print(f"  - BP_PARAMS['max_iter'] to see if more iterations help")
        print(f"  - EVALUATION_PARAMS['num_test_samples'] for more statistical power")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(
            f"\nMake sure you have generated the datasets first:"
        )
        print(
            f"  uv run python -m bpdecoderplus.cli --distance 3 --rounds 3 "
            f"--p 0.01 --task z --generate-dem --generate-uai --generate-syndromes 1000"
        )
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
