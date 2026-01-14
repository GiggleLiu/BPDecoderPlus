#!/usr/bin/env python3
"""
Generate circuit and detector graph diagrams using Stim.

Creates visualizations for documentation:
- Circuit diagrams (timeline, detector slice)
- Detector error model graphs
- Matching graphs

Usage:
    python python/generate_diagrams.py
"""

import os
import stim

OUTPUT_DIR = "benchmark/circuit_data/diagrams"


def generate_circuit_diagrams(distance: int = 3, rounds: int = 2, p_error: float = 0.001):
    """Generate various circuit diagram types."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate circuit
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_error,
        before_measure_flip_probability=p_error,
        after_reset_flip_probability=p_error,
    )

    dem = circuit.detector_error_model(decompose_errors=True)

    print(f"Generating diagrams for d={distance}, r={rounds}")
    print(f"  Circuit has {circuit.num_qubits} qubits")
    print(f"  DEM has {dem.num_detectors} detectors, {dem.num_observables} observables")

    # 1. Timeline diagram (shows gates over time)
    print("  Generating timeline diagram...")
    timeline_svg = circuit.diagram("timeline-svg")
    with open(os.path.join(OUTPUT_DIR, "circuit_timeline.svg"), "w") as f:
        f.write(str(timeline_svg))

    # 2. Detector slice diagram (shows detector structure)
    print("  Generating detector slice diagram...")
    try:
        det_slice_svg = circuit.diagram("detslice-svg")
        with open(os.path.join(OUTPUT_DIR, "detector_slice.svg"), "w") as f:
            f.write(str(det_slice_svg))
    except Exception as e:
        print(f"    Warning: Could not generate detector slice: {e}")

    # 3. Detector error model graph (matchgraph)
    print("  Generating matching graph diagram...")
    try:
        matchgraph_svg = dem.diagram("matchgraph-svg")
        with open(os.path.join(OUTPUT_DIR, "matching_graph.svg"), "w") as f:
            f.write(str(matchgraph_svg))
    except Exception as e:
        print(f"    Warning: Could not generate matching graph: {e}")

    # 4. 3D matchgraph (shows space-time structure)
    print("  Generating 3D matching graph...")
    try:
        matchgraph3d_svg = dem.diagram("matchgraph-3d-svg")
        with open(os.path.join(OUTPUT_DIR, "matching_graph_3d.svg"), "w") as f:
            f.write(str(matchgraph3d_svg))
    except Exception as e:
        print(f"    Warning: Could not generate 3D matching graph: {e}")

    # 5. Text-based representations for README
    print("  Generating text diagrams...")

    # DEM text representation
    with open(os.path.join(OUTPUT_DIR, "dem_structure.txt"), "w") as f:
        f.write(f"Detector Error Model for d={distance}, r={rounds}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of detectors: {dem.num_detectors}\n")
        f.write(f"Number of observables: {dem.num_observables}\n")
        f.write(f"Number of error mechanisms: {dem.num_errors}\n\n")
        f.write("First 30 error mechanisms:\n")
        f.write("-" * 50 + "\n")
        for i, instruction in enumerate(dem):
            if i >= 30:
                f.write("...\n")
                break
            f.write(str(instruction) + "\n")

    print(f"  Diagrams saved to {OUTPUT_DIR}/")
    return circuit, dem


def generate_ascii_surface_code(distance: int = 3):
    """Generate ASCII art of surface code layout."""

    lines = []
    lines.append(f"Rotated Surface Code (d={distance})")
    lines.append("")

    # Generate grid
    # Data qubits: O, X-stabilizers: X, Z-stabilizers: Z
    for row in range(2 * distance + 1):
        line = ""
        for col in range(2 * distance + 1):
            if row % 2 == 1 and col % 2 == 1:
                # Data qubit positions
                line += "◯ "
            elif row % 2 == 0 and col % 2 == 0:
                # Stabilizer positions
                if (row // 2 + col // 2) % 2 == 0:
                    if 0 < row < 2 * distance and 0 < col < 2 * distance:
                        line += "X "
                    else:
                        line += "  "
                else:
                    if 0 < row < 2 * distance and 0 < col < 2 * distance:
                        line += "Z "
                    else:
                        line += "  "
            else:
                line += "  "
        lines.append(line)

    lines.append("")
    lines.append("Legend: ◯ = Data qubit, X = X-stabilizer, Z = Z-stabilizer")

    return "\n".join(lines)


def generate_detection_event_ascii():
    """Generate ASCII explanation of detection events."""

    diagram = """
Detection Events: Syndrome Differencing
========================================

Round 1        Round 2        Round 3        Detection Events
─────────      ─────────      ─────────      ─────────────────

  0   0          0   0          0   1        Round 1: 0 0 0 0
    0              1              0          Round 2: 0 1 0 0  (changed!)
  0   0          0   0          0   0        Round 3: 0 0 0 1  (changed!)

Syndrome:      Syndrome:      Syndrome:      Detection Events:
[0,0,0,0]      [0,1,0,0]      [0,0,0,1]      R1: [0,0,0,0]
                                             R2: [0,1,0,0] ⊕ [0,0,0,0] = [0,1,0,0]
                                             R3: [0,0,0,1] ⊕ [0,1,0,0] = [0,1,0,1]

Why use detection events?
─────────────────────────
• Measurement errors cause random syndrome flips
• Detection events localize changes in space-time
• Decoders work on the 3D detection event graph
"""
    return diagram


def generate_dem_explanation_ascii():
    """Generate ASCII explanation of DEM structure."""

    diagram = """
Detector Error Model (DEM) Structure
====================================

Physical Error          Affected Detectors       DEM Representation
──────────────          ──────────────────       ──────────────────

   ┌───┐
   │ X │ on qubit 3     Triggers D0, D1          error(p) D0 D1
   └───┘                (adjacent stabilizers)

   ┌───┐
   │ Z │ on qubit 5     Triggers D2, D3, L0      error(p) D2 D3 L0
   └───┘                (+ logical error!)

   ┌───┐
   │ M │ error          Triggers D4              error(p) D4
   └───┘                (measurement flip)


DEM as a Tanner Graph
─────────────────────

    Detectors (checks)           Error mechanisms (variables)

        D0 ─────────────────── e1 (X error on q3)
        │                     /
        D1 ─────────────────/

        D2 ─────────────────── e2 (Z error on q5)
        │                     /│
        D3 ─────────────────/  │
        │                      │
        L0 ────────────────────┘  (logical observable)


Decoding = Finding minimum weight error set that explains detectors
"""
    return diagram


def main():
    print("=" * 60)
    print("Generating Circuit Diagrams")
    print("=" * 60)

    # Generate Stim diagrams
    circuit, dem = generate_circuit_diagrams(distance=3, rounds=2)

    # Generate ASCII diagrams
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nGenerating ASCII diagrams...")

    with open(os.path.join(OUTPUT_DIR, "surface_code_layout.txt"), "w") as f:
        f.write(generate_ascii_surface_code(3))

    with open(os.path.join(OUTPUT_DIR, "detection_events_explained.txt"), "w") as f:
        f.write(generate_detection_event_ascii())

    with open(os.path.join(OUTPUT_DIR, "dem_explained.txt"), "w") as f:
        f.write(generate_dem_explanation_ascii())

    print(f"\nAll diagrams saved to {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    main()
