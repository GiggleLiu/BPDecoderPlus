# Tropical Tensor Network for MPE

This folder contains an independent implementation of tropical tensor network
contraction for Most Probable Explanation (MPE). It does not depend on the
`bpdecoderplus` package; all code lives under `tropical_in_new/src`.

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
│   └── test_mpe.py
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
python tropical_in_new/examples/asia_network/main.py
```
