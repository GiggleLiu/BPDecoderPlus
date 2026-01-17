#!/usr/bin/env python3
"""
Generate noisy surface-code circuits in Stim format (pure Python).

Outputs stim circuits for rotated surface-code memory experiments at distance d=3
with circuit-level depolarizing noise p=0.01 by default. Detection events and a
logical observable are included via Stim's built-in generator.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, List

import stim


def parse_rounds(values: Iterable[int]) -> List[int]:
    unique = sorted({int(v) for v in values if int(v) > 0})
    if not unique:
        raise ValueError("At least one positive integer round count is required.")
    return unique


def prob_tag(p: float) -> str:
    # p=0.01 -> "p001"
    return f"p{p:.3f}".replace(".", "")


def generate_circuit(distance: int, rounds: int, p: float, task: str) -> stim.Circuit:
    """
    Build a rotated memory surface-code circuit with depolarizing noise.

    task: "z" or "x" logical memory experiment (Stim generator names end with _z / _x).
    Noise is applied to Clifford gates, reset/measure flips, and data between rounds.
    """
    if task not in {"x", "z"}:
        raise ValueError("task must be 'x' or 'z'")
    name = f"surface_code:rotated_memory_{task}"
    return stim.Circuit.generated(
        name,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def run_smoke_test(circuit: stim.Circuit, shots: int = 4) -> None:
    """
    Quick structural check: compile detector sampler and draw a few samples.
    """
    sampler = circuit.compile_detector_sampler()
    sampler.sample(shots)  # Raises if the circuit is invalid.


def write_circuit(circuit: stim.Circuit, path: pathlib.Path) -> None:
    path.write_text(str(circuit))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate noisy surface-code Stim circuits (rotated memory).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("datasets/noisy_circuits"),
        help="Output directory for .stim circuits",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=3,
        help="Surface-code distance",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        nargs="+",
        type=int,
        default=[3, 5, 7],
        help="List of measurement rounds to generate",
    )
    parser.add_argument(
        "-p",
        "--p",
        type=float,
        default=0.01,
        help="Depolarizing error rate",
    )
    parser.add_argument(
        "--task",
        choices=["x", "z"],
        default="z",
        help="Memory experiment orientation",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip compiling and sampling for quick validation",
    )
    args = parser.parse_args()

    if not 0.0 < args.p < 1.0:
        raise ValueError("p must be in (0, 1).")
    rounds_list = parse_rounds(args.rounds)

    args.output.mkdir(parents=True, exist_ok=True)

    for r in rounds_list:
        circuit = generate_circuit(args.distance, r, args.p, args.task)
        if not args.no_smoke_test:
            run_smoke_test(circuit)
        filename = f"sc_d{args.distance}_r{r}_{prob_tag(args.p)}_{args.task}.stim"
        write_circuit(circuit, args.output / filename)
        print(f"Wrote {args.output / filename}")

    print("Done.")


if __name__ == "__main__":
    main()
