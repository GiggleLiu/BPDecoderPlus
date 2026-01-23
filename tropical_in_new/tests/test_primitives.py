import torch

from src.primitives import IndexMap, argmax_trace, safe_log, tropical_einsum


def test_safe_log_zero_to_neg_inf():
    values = torch.tensor([1.0, 0.0], dtype=torch.float64)
    logged = safe_log(values)
    assert logged[0].item() == 0.0
    assert torch.isneginf(logged[1])


def test_tropical_einsum_binary_reduce():
    # a(x) with x in {0,1}, b(x,y) with y in {0,1}
    a = torch.tensor([0.2, 0.8])
    b = torch.tensor([[0.1, 0.5], [0.7, 0.3]])
    index_map = IndexMap(a_vars=(1,), b_vars=(1, 2), out_vars=(2,), elim_vars=(1,))
    values, backpointer = tropical_einsum(a, b, index_map, track_argmax=True)

    expected = torch.tensor([
        max(0.2 + 0.1, 0.8 + 0.7),
        max(0.2 + 0.5, 0.8 + 0.3),
    ])
    assert torch.allclose(values, expected)
    recovered = argmax_trace(backpointer, {2: 0})
    assert recovered[1] in (0, 1)
