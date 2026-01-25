"""
Generate Tanner Graph Visualizations
=====================================

This script generates all visualization figures for the Tanner Graph Walkthrough documentation.

Run with: uv run python examples/generate_tanner_visualizations.py

Output: docs/images/tanner_graph/*.png

Requirements:
    - matplotlib
    - networkx
    - seaborn (optional, for better heatmaps)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import networkx as nx
except ImportError:
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found. Heatmaps will use basic matplotlib.")

from bpdecoderplus.pytorch_bp import (
    read_model_file,
    BeliefPropagation,
    belief_propagate,
    apply_evidence,
)
from bpdecoderplus.dem import load_dem, build_parity_check_matrix
from bpdecoderplus.syndrome import load_syndrome_database

# Output directory
OUTPUT_DIR = Path("docs/images/tanner_graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_networkx_graph(bp):
    """Build networkx graph from BeliefPropagation object."""
    G = nx.Graph()

    # Variable nodes (detectors): 0 to nvars-1
    detector_nodes = list(range(bp.nvars))
    G.add_nodes_from(detector_nodes, bipartite=0, node_type="detector")

    # Factor nodes: offset by 100 to distinguish
    factor_offset = 100
    factor_nodes = [factor_offset + i for i in range(len(bp.factors))]
    G.add_nodes_from(factor_nodes, bipartite=1, node_type="factor")

    # Add edges
    for factor_idx, factor in enumerate(bp.factors):
        factor_node = factor_offset + factor_idx
        for var in factor.vars:
            var_idx = var - 1  # Convert to 0-based
            G.add_edge(var_idx, factor_node)

    return G, detector_nodes, factor_nodes


def visualize_full_tanner_graph(bp):
    """Generate full Tanner graph visualization."""
    print("Generating full Tanner graph visualization...")

    G, detector_nodes, factor_nodes = build_networkx_graph(bp)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Layout
    pos = nx.bipartite_layout(G, detector_nodes)

    # Node sizes based on degree
    detector_sizes = [G.degree(n) * 20 for n in detector_nodes]
    factor_sizes = [G.degree(n) * 5 for n in factor_nodes]

    # Draw
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=detector_nodes,
        node_color="lightblue",
        node_size=detector_sizes,
        label="Detectors",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=factor_nodes,
        node_color="lightcoral",
        node_size=factor_sizes,
        label="Error Factors",
        ax=ax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax)

    plt.title(
        f"Tanner Graph: d=3, r=3 Surface Code\n"
        f"{bp.nvars} detectors, {len(bp.factors)} error factors",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=12, loc="upper right")
    plt.axis("off")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "tanner_graph_full.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_subgraph(bp, center_detector=5, k_hop=1):
    """Visualize detector neighborhood subgraph."""
    print(f"Generating subgraph visualization (detector {center_detector}, {k_hop}-hop)...")

    G, detector_nodes, factor_nodes = build_networkx_graph(bp)

    # Get k-hop neighborhood
    subgraph_nodes = {center_detector}
    current_frontier = {center_detector}

    for _ in range(k_hop):
        next_frontier = set()
        for node in current_frontier:
            next_frontier.update(G.neighbors(node))
        subgraph_nodes.update(next_frontier)
        current_frontier = next_frontier

    # Extract subgraph
    subG = G.subgraph(subgraph_nodes)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate node lists
    sub_detectors = [n for n in subgraph_nodes if n < 100]
    sub_factors = [n for n in subgraph_nodes if n >= 100]

    # Layout
    pos = nx.spring_layout(subG, k=0.5, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_nodes(
        subG, pos, nodelist=sub_detectors, node_color="lightblue", node_size=600, ax=ax
    )
    nx.draw_networkx_nodes(
        subG, pos, nodelist=sub_factors, node_color="lightcoral", node_size=400, ax=ax
    )

    # Highlight center
    nx.draw_networkx_nodes(
        subG, pos, nodelist=[center_detector], node_color="darkblue", node_size=800, ax=ax
    )

    nx.draw_networkx_edges(subG, pos, alpha=0.5, width=2, ax=ax)

    # Labels
    labels = {}
    for n in sub_detectors:
        labels[n] = f"D{n}"
    for n in sub_factors:
        labels[n] = f"F{n-100}"
    nx.draw_networkx_labels(subG, pos, labels, font_size=10, ax=ax)

    plt.title(
        f"Tanner Subgraph: Detector {center_detector} and {k_hop}-hop neighborhood",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "tanner_graph_subgraph.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_degree_distribution(bp):
    """Generate degree distribution histogram."""
    print("Generating degree distribution...")

    G, detector_nodes, factor_nodes = build_networkx_graph(bp)

    var_degrees = [G.degree(n) for n in detector_nodes]
    factor_degrees = [G.degree(n) for n in factor_nodes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Detector degrees
    ax1.hist(var_degrees, bins=20, alpha=0.7, color="lightblue", edgecolor="black")
    ax1.set_xlabel("Degree", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(
        f"Detector Node Degrees\n(mean={np.mean(var_degrees):.1f})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(alpha=0.3)

    # Factor degrees
    ax2.hist(factor_degrees, bins=30, alpha=0.7, color="lightcoral", edgecolor="black")
    ax2.set_xlabel("Degree", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(
        f"Factor Node Degrees\n(mean={np.mean(factor_degrees):.1f})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "degree_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_adjacency_matrix(H):
    """Generate H matrix heatmap."""
    print("Generating adjacency matrix heatmap...")

    fig, ax = plt.subplots(figsize=(14, 6))

    if HAS_SEABORN:
        sns.heatmap(H, cmap="Blues", cbar=True, ax=ax, xticklabels=False, yticklabels=False)
    else:
        im = ax.imshow(H, cmap="Blues", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax)

    ax.set_xlabel("Error Mechanisms", fontsize=12)
    ax.set_ylabel("Detectors", fontsize=12)
    ax.set_title(
        f"Parity Check Matrix H ({H.shape[0]} × {H.shape[1]})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "adjacency_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_parameter_comparison(bp):
    """Generate parameter comparison plots (damping)."""
    print("Generating parameter comparison plots...")

    # Load syndrome
    try:
        syndromes, observables, _ = load_syndrome_database("datasets/sc_d3_r3_p0010_z.npz")
    except FileNotFoundError:
        print("  ⚠ Warning: Syndrome database not found, skipping parameter comparison")
        return

    syndrome = syndromes[0]
    evidence = {det_idx + 1: int(syndrome[det_idx]) for det_idx in range(len(syndrome))}
    bp_ev = apply_evidence(bp, evidence)

    damping_values = [0.0, 0.1, 0.3, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, damping in zip(axes, damping_values):
        state, info = belief_propagate(
            bp_ev, max_iter=100, tol=1e-6, damping=damping, normalize=True
        )

        # Create simple visualization showing convergence info
        ax.text(
            0.5,
            0.6,
            f"Damping = {damping}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.4,
            f"Converged: {info.converged}\nIterations: {info.iterations}",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )

        # Color background based on convergence
        if info.converged:
            ax.set_facecolor("#e8f5e8")  # Light green
        else:
            ax.set_facecolor("#f5e8e8")  # Light red

        ax.set_title(f"Damping = {damping}", fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "BP Convergence with Different Damping Values",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = OUTPUT_DIR / "damping_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_convergence_analysis(bp):
    """Generate convergence analysis plot."""
    print("Generating convergence analysis...")

    try:
        syndromes, observables, _ = load_syndrome_database("datasets/sc_d3_r3_p0010_z.npz")
    except FileNotFoundError:
        print("  ⚠ Warning: Syndrome database not found, skipping convergence analysis")
        return

    # Test multiple syndromes with different damping values
    damping_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    num_test = min(50, len(syndromes))

    results = {d: [] for d in damping_values}

    for damping in damping_values:
        print(f"  Testing damping={damping}...")
        for i in range(num_test):
            syndrome = syndromes[i]
            evidence = {
                det_idx + 1: int(syndrome[det_idx]) for det_idx in range(len(syndrome))
            }
            bp_ev = apply_evidence(bp, evidence)

            state, info = belief_propagate(
                bp_ev, max_iter=100, tol=1e-6, damping=damping, normalize=True
            )

            if info.converged:
                results[damping].append(info.iterations)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    data_to_plot = []
    labels = []

    for i, damping in enumerate(damping_values):
        if results[damping]:
            positions.append(i)
            data_to_plot.append(results[damping])
            labels.append(f"{damping}")

    bp_plot = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

    # Color boxes
    for patch in bp_plot["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_xlabel("Damping Value", fontsize=12)
    ax.set_ylabel("Iterations to Convergence", fontsize=12)
    ax.set_title(
        "Convergence Speed vs Damping Factor\n(Lower is faster)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "convergence_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("Generating Tanner Graph Visualizations")
    print("=" * 70)

    # Check if dataset exists
    try:
        print("\nLoading dataset...")
        model = read_model_file("datasets/sc_d3_r3_p0010_z.uai")
        dem = load_dem("datasets/sc_d3_r3_p0010_z.dem")
        H, priors, obs_flip = build_parity_check_matrix(dem)
        bp = BeliefPropagation(model)
        print(f"  ✓ Loaded d=3, r=3 surface code dataset")
        print(f"    Variables: {bp.nvars}, Factors: {len(bp.factors)}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease generate the dataset first:")
        print(
            "  uv run python -m bpdecoderplus.cli --distance 3 --rounds 3 "
            "--p 0.01 --task z --generate-dem --generate-uai --generate-syndromes 1000"
        )
        return 1

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerating visualizations...\n")

    # Generate all visualizations
    try:
        visualize_full_tanner_graph(bp)
        visualize_subgraph(bp, center_detector=5, k_hop=1)
        visualize_degree_distribution(bp)
        visualize_adjacency_matrix(H)
        visualize_parameter_comparison(bp)
        visualize_convergence_analysis(bp)

        print("\n" + "=" * 70)
        print(f"✓ All visualizations saved to {OUTPUT_DIR}/")
        print("=" * 70)
        print("\nGenerated files:")
        for file in sorted(OUTPUT_DIR.glob("*.png")):
            print(f"  - {file.name}")

        return 0

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
