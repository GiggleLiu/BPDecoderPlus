# Tropical Tensor Network for MPE

This folder contains an independent implementation of tropical tensor network
contraction for Most Probable Explanation (MPE). It uses `omeco` for contraction
order optimization and does not depend on the `bpdecoderplus` package; all code
lives under `tropical_in_new/src`.

`omeco` provides high-quality contraction order heuristics (greedy and
simulated annealing). Install it alongside Torch to run the examples and tests.

## Structure

```
tropical_in_new/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── primitives.py
│   ├── network.py
│   ├── contraction.py
│   ├── mpe.py
│   └── utils.py
├── tests/
│   ├── test_primitives.py
│   ├── test_contraction.py
│   ├── test_mpe.py
│   └── test_utils.py
├── examples/
│   └── asia_network/
│       ├── main.py
│       └── model.uai
└── docs/
    ├── mathematical_description.md
    ├── api_reference.md
    └── usage_guide.md
```

## Quick Start

```bash
pip install -r tropical_in_new/requirements.txt
python tropical_in_new/examples/asia_network/main.py
```

## Notes on omeco

`omeco` is a Rust-backed Python package. If a prebuilt wheel is not available
for your Python version, you will need a Rust toolchain with `cargo` on PATH to
build it from source. See the omeco repository for details:
https://github.com/GiggleLiu/omeco
