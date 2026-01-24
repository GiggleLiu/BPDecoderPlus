import pytest
import torch

from tropical_in_new.src.primitives import (
    IndexMap,
    argmax_trace,
    safe_log,
    tropical_einsum,
    tropical_reduce_max,
)


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


def test_tropical_einsum_no_elim():
    a = torch.tensor([0.1, 0.2])
    b = torch.tensor([0.3, 0.4])
    index_map = IndexMap(a_vars=(1,), b_vars=(2,), out_vars=(1, 2), elim_vars=())
    values, backpointer = tropical_einsum(a, b, index_map)
    assert values.shape == (2, 2)
    assert backpointer is None


def test_tropical_einsum_out_vars_mismatch():
    a = torch.tensor([0.1, 0.2])
    b = torch.tensor([0.3, 0.4])
    index_map = IndexMap(a_vars=(1,), b_vars=(2,), out_vars=(2, 1), elim_vars=())
    with pytest.raises(ValueError, match="out_vars does not match"):
        tropical_einsum(a, b, index_map)


def test_tropical_reduce_max_no_elim():
    t = torch.tensor([1.0, 2.0])
    values, bp = tropical_reduce_max(t, (1,), [])
    assert torch.equal(values, t)
    assert bp is None


def test_tropical_reduce_max_missing_var():
    t = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="not present in vars"):
        tropical_reduce_max(t, (1,), (99,))


def test_tropical_reduce_max_no_track():
    t = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    values, bp = tropical_reduce_max(t, (1, 2), (2,), track_argmax=False)
    assert values.shape == (2,)
    assert bp is None


def test_argmax_trace_no_elim_vars():
    from tropical_in_new.src.primitives import Backpointer
    bp = Backpointer(elim_vars=(), elim_shape=(), out_vars=(), argmax_flat=torch.tensor(0))
    result = argmax_trace(bp, {})
    assert result == {}


def test_argmax_trace_missing_key():
    from tropical_in_new.src.primitives import Backpointer
    bp = Backpointer(
        elim_vars=(2,), elim_shape=(3,), out_vars=(1,),
        argmax_flat=torch.tensor([0, 1, 2])
    )
    with pytest.raises(KeyError, match="Missing assignment"):
        argmax_trace(bp, {})


def test_argmax_trace_scalar_backpointer():
    """Test argmax_trace when out_vars is empty (scalar result)."""
    from tropical_in_new.src.primitives import Backpointer
    bp = Backpointer(
        elim_vars=(1,), elim_shape=(3,), out_vars=(),
        argmax_flat=torch.tensor(2)
    )
    result = argmax_trace(bp, {})
    assert result == {1: 2}
