#!/usr/bin/env python3
"""
Visualization scripts for BPDecoderPlus benchmark results.

This script generates publication-quality plots from benchmark data.

Usage:
    python python/visualize.py [--data-dir benchmark/data] [--output-dir results/plots]
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'IP': '#1f77b4',      # Blue
    'BP': '#ff7f0e',      # Orange
    'BPOSD': '#2ca02c',   # Green
    'Matching': '#d62728', # Red
}

MARKERS = {
    'IP': 'o',
    'BP': 's',
    'BPOSD': '^',
    'Matching': 'D',
}


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_threshold(data_dir, output_dir, decoder_name='ip'):
    """
    Plot threshold curves (logical error rate vs physical error rate).

    This is the standard plot for evaluating decoder performance.
    """
    filename = f"{decoder_name}_decoder_results.json"
    filepath = data_dir / filename

    if not filepath.exists():
        print(f"Warning: {filepath} not found, skipping threshold plot for {decoder_name}")
        return

    data = load_json(filepath)

    fig, ax = plt.subplots(figsize=(8, 6))

    distances = data['distances']
    error_rates = data['error_rates']

    for d in distances:
        d_str = str(d)
        if d_str not in data['data']:
            continue

        lers = []
        stds = []
        ps = []

        for p in error_rates:
            p_str = str(p)
            if p_str in data['data'][d_str]:
                ps.append(p)
                lers.append(data['data'][d_str][p_str]['logical_error_rate'])
                stds.append(data['data'][d_str][p_str]['std_error'])

        ax.errorbar(ps, lers, yerr=stds, marker='o', label=f'd={d}',
                   capsize=3, capthick=1, markersize=6)

    ax.set_xlabel('Physical Error Rate (p)')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title(f'{decoder_name.upper()} Decoder Threshold Curve')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(error_rates) * 1.05)

    output_file = output_dir / f'threshold_{decoder_name}.png'
    fig.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close(fig)


def plot_decoder_comparison(data_dir, output_dir):
    """
    Compare multiple decoders on the same plot.
    """
    decoders = ['ip', 'bp', 'bposd']
    decoder_data = {}

    for dec in decoders:
        filepath = data_dir / f"{dec}_decoder_results.json"
        if filepath.exists():
            decoder_data[dec.upper()] = load_json(filepath)

    if not decoder_data:
        print("No decoder data found for comparison plot")
        return

    # Get common distances
    first_data = list(decoder_data.values())[0]
    distances = first_data['distances']

    # Create one subplot per distance
    n_distances = len(distances)
    fig, axes = plt.subplots(1, n_distances, figsize=(5 * n_distances, 5))
    if n_distances == 1:
        axes = [axes]

    for idx, d in enumerate(distances):
        ax = axes[idx]
        d_str = str(d)

        for dec_name, data in decoder_data.items():
            if d_str not in data['data']:
                continue

            error_rates = data['error_rates']
            lers = []
            stds = []
            ps = []

            for p in error_rates:
                p_str = str(p)
                if p_str in data['data'][d_str]:
                    ps.append(p)
                    lers.append(data['data'][d_str][p_str]['logical_error_rate'])
                    stds.append(data['data'][d_str][p_str]['std_error'])

            ax.errorbar(ps, lers, yerr=stds,
                       marker=MARKERS.get(dec_name, 'o'),
                       color=COLORS.get(dec_name, 'gray'),
                       label=dec_name, capsize=2, markersize=5)

        ax.set_xlabel('Physical Error Rate')
        ax.set_ylabel('Logical Error Rate')
        ax.set_title(f'Distance d={d}')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Decoder Comparison', fontsize=16)
    fig.tight_layout()

    output_file = output_dir / 'decoder_comparison.png'
    fig.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close(fig)


def plot_timing_comparison(data_dir, output_dir):
    """
    Plot decoder timing comparison.
    """
    filepath = data_dir / "timing_results.json"
    if not filepath.exists():
        print("Timing results not found, skipping timing plot")
        return

    data = load_json(filepath)

    fig, ax = plt.subplots(figsize=(8, 6))

    for dec_name, dec_data in data['data'].items():
        distances = []
        times = []
        stds = []

        for d_str, timing in sorted(dec_data.items(), key=lambda x: int(x[0])):
            distances.append(int(d_str))
            times.append(timing['mean_time_ms'])
            stds.append(timing['std_time_ms'])

        ax.errorbar(distances, times, yerr=stds,
                   marker=MARKERS.get(dec_name, 'o'),
                   color=COLORS.get(dec_name, 'gray'),
                   label=dec_name, capsize=3, markersize=8)

    ax.set_xlabel('Code Distance')
    ax.set_ylabel('Decoding Time (ms)')
    ax.set_title('Decoder Timing Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    output_file = output_dir / 'timing_comparison.png'
    fig.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close(fig)


def plot_atom_loss(data_dir, output_dir):
    """
    Plot logical error rate vs atom loss rate.
    """
    filepath = data_dir / "atom_loss_results.json"
    if not filepath.exists():
        print("Atom loss results not found, skipping atom loss plot")
        return

    data = load_json(filepath)

    fig, ax = plt.subplots(figsize=(8, 6))

    for d_str, d_data in data['data'].items():
        loss_rates = []
        lers = []
        stds = []

        for p_loss_str, result in sorted(d_data.items(), key=lambda x: float(x[0])):
            loss_rates.append(float(p_loss_str))
            lers.append(result['logical_error_rate'])
            stds.append(result['std_error'])

        ax.errorbar(loss_rates, lers, yerr=stds,
                   marker='o', label=f'd={d_str}', capsize=3, markersize=6)

    ax.set_xlabel('Atom Loss Rate')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title(f'Effect of Atom Loss (p_error={data["p_error"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = output_dir / 'atom_loss.png'
    fig.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close(fig)


def plot_scalability(data_dir, output_dir):
    """
    Plot decoder scalability (time and accuracy vs code size).
    """
    filepath = data_dir / "scalability_results.json"
    if not filepath.exists():
        print("Scalability results not found, skipping scalability plot")
        return

    data = load_json(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for dec_name, dec_data in data['data'].items():
        n_qubits = []
        times = []
        lers = []

        for d_str, result in sorted(dec_data.items(), key=lambda x: int(x[0])):
            n_qubits.append(result['n_qubits'])
            times.append(result['avg_time_ms'])
            lers.append(result['logical_error_rate'])

        ax1.plot(n_qubits, times, marker='o', label=dec_name)
        ax2.plot(n_qubits, lers, marker='s', label=dec_name)

    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Decoding Time (ms)')
    ax1.set_title('Decoding Time vs Code Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Logical Error Rate')
    ax2.set_title(f'Accuracy vs Code Size (p={data["p"]})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    fig.tight_layout()

    output_file = output_dir / 'scalability.png'
    fig.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize BPDecoderPlus benchmark results')
    parser.add_argument('--data-dir', type=str, default='benchmark/data',
                       help='Directory containing benchmark data')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                       help='Directory for output plots')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("BPDecoderPlus Visualization")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Generate all plots
    print("\n>>> Generating threshold plots...")
    for decoder in ['ip', 'bp', 'bposd']:
        plot_threshold(data_dir, output_dir, decoder)

    print("\n>>> Generating decoder comparison plot...")
    plot_decoder_comparison(data_dir, output_dir)

    print("\n>>> Generating timing comparison plot...")
    plot_timing_comparison(data_dir, output_dir)

    print("\n>>> Generating atom loss plot...")
    plot_atom_loss(data_dir, output_dir)

    print("\n>>> Generating scalability plot...")
    plot_scalability(data_dir, output_dir)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


def visualize_circuit_stim(distance=3, rounds=1, noise=0.01, output_dir=None,
                          diagram_type='timeline-svg'):
    """
    Visualize a surface code circuit with error model using Stim.

    Stim provides high-quality visualizations for quantum error correction circuits
    including timeline diagrams, detector slice diagrams, and more.

    Args:
        distance: Surface code distance (default: 3)
        rounds: Number of syndrome extraction rounds (default: 1)
        noise: Physical error probability (default: 0.01)
        output_dir: Directory to save outputs (default: results/plots)
        diagram_type: Type of diagram to generate. Options:
            - 'timeline-svg': Timeline diagram showing circuit operations
            - 'detslice-svg': Detector slice diagram showing stabilizers
            - 'timeline-text': ASCII timeline diagram
            - 'matchgraph-svg': Matching graph visualization
            - 'timeslice-svg': Time-slice diagram
            - 'detslice-with-ops-svg': Detector slices with operations

    Returns:
        Tuple of (circuit, diagram_content)
    """
    try:
        import stim
    except ImportError:
        print("Error: stim is not installed. Install with: pip install stim")
        return None, None

    output_dir = Path(output_dir) if output_dir else Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a surface code circuit with noise using Stim's built-in generator
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )

    print(f"Generated surface code circuit:")
    print(f"  Distance: {distance}")
    print(f"  Rounds: {rounds}")
    print(f"  Noise: {noise}")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Operations: {circuit.num_operations}")
    print(f"  Detectors: {circuit.num_detectors}")
    print(f"  Observables: {circuit.num_observables}")

    # Generate the requested diagram
    diagram_content = None
    output_file = None

    if diagram_type == 'timeline-svg':
        diagram_content = circuit.diagram('timeline-svg')
        output_file = output_dir / f'circuit_d{distance}_r{rounds}_timeline.svg'
        with open(output_file, 'w') as f:
            f.write(str(diagram_content))
        print(f"Saved timeline diagram: {output_file}")

    elif diagram_type == 'detslice-svg':
        diagram_content = circuit.diagram('detslice-svg')
        output_file = output_dir / f'circuit_d{distance}_r{rounds}_detslice.svg'
        with open(output_file, 'w') as f:
            f.write(str(diagram_content))
        print(f"Saved detector slice diagram: {output_file}")

    elif diagram_type == 'timeline-text':
        diagram_content = circuit.diagram('timeline-text')
        output_file = output_dir / f'circuit_d{distance}_r{rounds}_timeline.txt'
        with open(output_file, 'w') as f:
            f.write(str(diagram_content))
        print(f"Saved ASCII timeline: {output_file}")
        print("\nASCII Timeline:")
        print(str(diagram_content))

    elif diagram_type == 'matchgraph-svg':
        # Get the detector error model first
        dem = circuit.detector_error_model(decompose_errors=True)
        diagram_content = dem.diagram('matchgraph-svg')
        output_file = output_dir / f'circuit_d{distance}_r{rounds}_matchgraph.svg'
        with open(output_file, 'w') as f:
            f.write(str(diagram_content))
        print(f"Saved matching graph: {output_file}")

    elif diagram_type == 'timeslice-svg':
        diagram_content = circuit.diagram('timeslice-svg')
        output_file = output_dir / f'circuit_d{distance}_r{rounds}_timeslice.svg'
        with open(output_file, 'w') as f:
            f.write(str(diagram_content))
        print(f"Saved time-slice diagram: {output_file}")

    elif diagram_type == 'detslice-with-ops-svg':
        diagram_content = circuit.diagram('detslice-with-ops-svg')
        output_file = output_dir / f'circuit_d{distance}_r{rounds}_detslice_ops.svg'
        with open(output_file, 'w') as f:
            f.write(str(diagram_content))
        print(f"Saved detector slice with ops: {output_file}")

    else:
        print(f"Unknown diagram type: {diagram_type}")
        print("Available types: timeline-svg, detslice-svg, timeline-text, "
              "matchgraph-svg, timeslice-svg, detslice-with-ops-svg")

    # Also save the circuit definition
    circuit_file = output_dir / f'circuit_d{distance}_r{rounds}.stim'
    with open(circuit_file, 'w') as f:
        f.write(str(circuit))
    print(f"Saved circuit definition: {circuit_file}")

    # Save the detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    dem_file = output_dir / f'circuit_d{distance}_r{rounds}.dem'
    with open(dem_file, 'w') as f:
        f.write(str(dem))
    print(f"Saved detector error model: {dem_file}")

    return circuit, diagram_content


def visualize_circuit_with_samples(distance=3, rounds=1, noise=0.01, n_samples=1,
                                   output_dir=None, seed=None):
    """
    Visualize a surface code circuit with sampled error instances.

    This function samples actual errors from the noise model and creates
    visualizations showing where errors occurred and which detectors fired.

    Args:
        distance: Surface code distance
        rounds: Number of syndrome extraction rounds
        noise: Physical error probability
        n_samples: Number of error samples to visualize
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility

    Returns:
        Tuple of (circuit, samples)
    """
    try:
        import stim
    except ImportError:
        print("Error: stim is not installed. Install with: pip install stim")
        return None, None

    output_dir = Path(output_dir) if output_dir else Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate circuit
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )

    # Create sampler
    sampler = circuit.compile_detector_sampler(seed=seed)

    # Sample detection events
    detection_events, observable_flips = sampler.sample(
        shots=n_samples,
        separate_observables=True
    )

    print(f"\nSampled {n_samples} error instance(s):")
    print(f"  Detection event shape: {detection_events.shape}")
    print(f"  Observable flip shape: {observable_flips.shape}")

    # Create visualization of detection events
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Detection event heatmap
    ax1 = axes[0]
    if n_samples > 1:
        im = ax1.imshow(detection_events.T, aspect='auto', cmap='Reds')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Detector')
        ax1.set_title(f'Detection Events (d={distance}, r={rounds}, p={noise})')
        plt.colorbar(im, ax=ax1, label='Triggered')
    else:
        # Single sample - show as bar plot
        triggered = np.where(detection_events[0])[0]
        ax1.bar(range(len(detection_events[0])), detection_events[0], color='red', alpha=0.7)
        ax1.set_xlabel('Detector Index')
        ax1.set_ylabel('Triggered (0/1)')
        ax1.set_title(f'Detection Events (d={distance}, r={rounds}, p={noise})\n'
                     f'{len(triggered)} detectors triggered')

    # Right: Summary statistics
    ax2 = axes[1]
    n_triggered_per_sample = detection_events.sum(axis=1)
    ax2.hist(n_triggered_per_sample, bins=max(1, min(20, int(n_triggered_per_sample.max()) + 1)),
             color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(n_triggered_per_sample.mean(), color='red', linestyle='--',
               label=f'Mean: {n_triggered_per_sample.mean():.1f}')
    ax2.set_xlabel('Number of Triggered Detectors')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Detection Events')
    ax2.legend()

    # Add observable flip info
    n_logical_errors = observable_flips.sum()
    logical_error_rate = n_logical_errors / n_samples
    info_text = (f"Noise Model:\n"
                f"  p = {noise}\n"
                f"  distance = {distance}\n"
                f"  rounds = {rounds}\n\n"
                f"Results:\n"
                f"  samples = {n_samples}\n"
                f"  logical errors = {n_logical_errors}\n"
                f"  LER = {logical_error_rate:.4f}")
    ax2.text(0.98, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    output_file = output_dir / f'circuit_d{distance}_r{rounds}_samples.png'
    fig.savefig(output_file)
    print(f"Saved sample visualization: {output_file}")
    plt.close(fig)

    return circuit, (detection_events, observable_flips)


def visualize_circuit_command():
    """
    Command-line interface for circuit visualization using Stim.
    """
    parser = argparse.ArgumentParser(
        description='Visualize surface code circuit with error model using Stim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py circuit --distance 5 --diagram timeline-svg
  python visualize.py circuit --distance 3 --rounds 3 --diagram detslice-svg
  python visualize.py circuit --distance 5 --samples 1000 --noise 0.01
        """)
    parser.add_argument('--distance', '-d', type=int, default=3,
                       help='Surface code distance (default: 3)')
    parser.add_argument('--rounds', '-r', type=int, default=1,
                       help='Number of syndrome extraction rounds (default: 1)')
    parser.add_argument('--noise', '-p', type=float, default=0.01,
                       help='Physical error probability (default: 0.01)')
    parser.add_argument('--output-dir', '-o', type=str, default='results/plots',
                       help='Output directory (default: results/plots)')
    parser.add_argument('--diagram', type=str, default='timeline-svg',
                       choices=['timeline-svg', 'detslice-svg', 'timeline-text',
                               'matchgraph-svg', 'timeslice-svg', 'detslice-with-ops-svg', 'all'],
                       help='Type of diagram to generate (default: timeline-svg)')
    parser.add_argument('--samples', '-n', type=int, default=0,
                       help='Number of error samples to visualize (0 = skip sampling)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for sampling')
    args = parser.parse_args()

    print("="*60)
    print("Surface Code Circuit Visualization (using Stim)")
    print("="*60)

    # Generate diagram(s)
    if args.diagram == 'all':
        for dtype in ['timeline-svg', 'detslice-svg', 'matchgraph-svg', 'timeslice-svg']:
            print(f"\n>>> Generating {dtype}...")
            visualize_circuit_stim(
                distance=args.distance,
                rounds=args.rounds,
                noise=args.noise,
                output_dir=args.output_dir,
                diagram_type=dtype
            )
    else:
        visualize_circuit_stim(
            distance=args.distance,
            rounds=args.rounds,
            noise=args.noise,
            output_dir=args.output_dir,
            diagram_type=args.diagram
        )

    # Generate sample visualization if requested
    if args.samples > 0:
        print(f"\n>>> Sampling {args.samples} error instances...")
        visualize_circuit_with_samples(
            distance=args.distance,
            rounds=args.rounds,
            noise=args.noise,
            n_samples=args.samples,
            output_dir=args.output_dir,
            seed=args.seed
        )

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'circuit':
        sys.argv.pop(1)  # Remove 'circuit' from args
        visualize_circuit_command()
    else:
        main()
