"""
Command-line interface for generating noisy surface-code circuits and syndrome databases.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from bpdecoderplus.circuit import (
    generate_circuit,
    generate_filename,
    parse_rounds,
    run_smoke_test,
    write_circuit,
)
from bpdecoderplus.dem import generate_dem_from_circuit
from bpdecoderplus.syndrome import generate_syndrome_database_from_circuit


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
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
    parser.add_argument(
        "--generate-syndromes",
        type=int,
        metavar="NUM_SHOTS",
        help="Generate syndrome database with specified number of shots",
    )
    parser.add_argument(
        "--generate-dem",
        action="store_true",
        help="Generate detector error model (.dem file)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate error rate
    if not 0.0 < args.p < 1.0:
        print(f"Error: p must be in (0, 1), got {args.p}", file=sys.stderr)
        return 1

    # Parse and validate rounds
    try:
        rounds_list = parse_rounds(args.rounds)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate circuits
    for r in rounds_list:
        circuit = generate_circuit(args.distance, r, args.p, args.task)

        if not args.no_smoke_test:
            try:
                run_smoke_test(circuit)
            except Exception as e:
                print(f"Error: Smoke test failed for r={r}: {e}", file=sys.stderr)
                return 1

        filename = generate_filename(args.distance, r, args.p, args.task)
        output_path = args.output / filename
        write_circuit(circuit, output_path)
        print(f"Wrote {output_path}")

        # Generate DEM if requested
        if args.generate_dem:
            dem_path = generate_dem_from_circuit(output_path)
            print(f"Wrote {dem_path}")

        # Generate syndrome database if requested
        if args.generate_syndromes:
            syndrome_path = generate_syndrome_database_from_circuit(
                output_path, args.generate_syndromes
            )
            print(f"Wrote {syndrome_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
