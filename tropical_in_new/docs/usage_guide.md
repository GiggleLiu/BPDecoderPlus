## Tropical Tensor Network Usage

This guide shows how to parse a UAI model and compute MPE using the
tropical tensor network implementation.

### Quick Start

```python
from src import mpe_tropical, read_model_file

model = read_model_file("tropical_in_new/examples/asia_network/model.uai")
assignment, score, info = mpe_tropical(model)
print(assignment, score, info)
```

### Evidence

```python
from src import mpe_tropical, read_model_file

model = read_model_file("tropical_in_new/examples/asia_network/model.uai")
evidence = {1: 0}  # variable index is 1-based
assignment, score, info = mpe_tropical(model, evidence=evidence)
```

### Running the Example

```bash
python tropical_in_new/examples/asia_network/main.py
```
