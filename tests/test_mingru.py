import pytest
import torch

import mingru
from tests.helpers import scriptable


def test_mingrucell():

    rnn = mingru.MinGRUCell(input_size=1, hidden_size=5)

    x = torch.randn(2, 5, 1)
    h = rnn.init_hidden_state(x)
    assert h.shape == (2, 1, 5)

    out_seq = []
    h_seq = h

    for i in range(x.shape[1]):
        out, h = rnn(x[:, i : i + 1], h)  # NOQA
        assert h.shape == (2, 1, 5)
        assert out.shape == (2, 1, 5)
        out_seq.append(out)
        h_seq = h
    out_seq = torch.cat(out_seq, 1)

    out_par, h_par = rnn(x, rnn.init_hidden_state(x))
    assert out_par.shape == (2, 5, 5)
    assert torch.allclose(out_seq, out_par)
    assert torch.allclose(h_seq, h_par)


@pytest.mark.parametrize("bias", [True, False])
def test_mingrucell_scripting(bias):
    rnn = mingru.MinGRUCell(input_size=2, hidden_size=5, bias=bias)
    scriptable.assert_scriptable(rnn, is_conv=False)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_mingru(bias, residual):

    rnn = mingru.MinGRU(
        input_size=1,
        hidden_sizes=[3, 5],
        bias=bias,
        residual=residual,
    )

    x = torch.randn(2, 5, 1)
    h = rnn.init_hidden_state(x)
    assert h[0].shape == (2, 1, 3)
    assert h[1].shape == (2, 1, 5)

    out_seq = []
    for i in range(x.shape[1]):
        out, h = rnn(x[:, i : i + 1], h)  # NOQA
        assert h[0].shape == (2, 1, 3)
        assert h[1].shape == (2, 1, 5)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, h = rnn(x, rnn.init_hidden_state(x))
    assert out_par.shape == (2, 5, 5)
    assert torch.allclose(out_seq, out_par, atol=1e-4)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_mingru_scriptable(bias, residual):
    rnn = mingru.MinGRU(
        input_size=2,
        hidden_sizes=[3, 5],
        bias=bias,
        residual=residual,
    )

    scriptable.assert_scriptable(rnn, is_conv=False)
