"""
BPDecoderPlus: Noisy circuit generation for BP decoding of surface codes.

This package provides tools for generating noisy surface-code circuits
in Stim format for belief propagation (BP) decoding demonstrations.
"""

from bpdecoderplus.circuit import (
    generate_circuit,
    parse_rounds,
    prob_tag,
    run_smoke_test,
    write_circuit,
)
from bpdecoderplus.batch_osd import BatchOSDDecoder

__version__ = "0.1.0"
__all__ = [
    "generate_circuit",
    "parse_rounds",
    "prob_tag",
    "run_smoke_test",
    "write_circuit",
    "BatchOSDDecoder",
]
