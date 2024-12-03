import pytest
import torch

import mingru
from tests.helpers import scriptable


def test_mingruconv2dcell():

    rnn = mingru.MinConv2dGRUCell(
        input_size=3,
        hidden_size=5,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    x = torch.randn(2, 5, 3, 32, 32)
    h_seq = rnn.init_hidden_state(x)
    assert isinstance(h_seq, list)
    assert len(h_seq) == 1
    assert h_seq[0].shape == (2, 1, 5, 32, 32)

    out_seq = []
    for i in range(x.shape[1]):
        out, h_seq = rnn(x[:, i : i + 1], h_seq)  # NOQA
        assert out.shape == (2, 1, 5, 32, 32)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, _ = rnn(x, rnn.init_hidden_state(x))
    assert out_par.shape == (2, 5, 5, 32, 32)
    assert torch.allclose(out_seq, out_par)


def test_mingruconv2dcell_downsample():
    rnn = mingru.MinConv2dGRUCell(
        input_size=3,
        hidden_size=5,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    x = torch.randn(2, 5, 3, 32, 32)
    h = rnn.init_hidden_state(x)
    assert isinstance(h, list)
    assert len(h) == 1
    assert h[0].shape == (2, 1, 5, 16, 16)

    out_seq = []
    for i in range(x.shape[1]):
        out, h = rnn(x[:, i : i + 1], h)  # NOQA
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, _ = rnn(x, rnn.init_hidden_state(x))
    assert torch.allclose(out_seq, out_par)


def test_mingrucell_scripting():
    rnn = mingru.MinConv2dGRUCell(
        input_size=3,
        hidden_size=5,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    scriptable.assert_scriptable(rnn, is_conv=True)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_mingruconv2d(bias, residual):

    rnn = mingru.MinConv2dGRU(
        input_size=1,
        hidden_sizes=[3, 5],
        kernel_sizes=[3, 3],
        paddings=[1, 1],
        strides=1,
        bias=bias,
        residual=residual,
    )

    assert rnn.layer_sizes == (1, 3, 5)
    assert rnn.num_layers == 2

    x = torch.randn(2, 5, 1, 32, 32)
    h = rnn.init_hidden_state(x)
    assert h[0].shape == (2, 1, 3, 32, 32)
    assert h[1].shape == (2, 1, 5, 32, 32)

    out_seq = []
    for i in range(x.shape[1]):
        out, h = rnn(x[:, i : i + 1], h)  # NOQA
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, h = rnn(x, rnn.init_hidden_state(x))

    assert torch.allclose(out_seq, out_par, atol=1e-4)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_mingru_scripting(bias, residual):
    rnn = mingru.MinConv2dGRU(
        input_size=2,
        hidden_sizes=[3, 5],
        kernel_sizes=[3, 3],
        paddings=[1, 1],
        strides=1,
        bias=bias,
        residual=residual,
    )

    scriptable.assert_scriptable(rnn, is_conv=True)
