import pytest
import torch

import mingru
from tests.helpers import scriptable


def test_bidirectional_cell():
    bidir = mingru.Bidirectional(mingru.MinGRUCell(input_size=1, hidden_size=5))

    x = torch.randn(2, 5, 1)
    h = bidir.init_hidden_state(x)
    assert h[0].shape == (2, 1, 5)
    assert h[1].shape == (2, 1, 5)

    out, h = bidir(x, h=h)
    assert len(out) == 2
    assert len(h) == 2
    assert out[0].shape == (2, 5, 5)
    assert out[1].shape == (2, 5, 5)
    assert h[0].shape == (2, 1, 5)
    assert h[1].shape == (2, 1, 5)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(h[0], h[1])


def test_bidirectional_stacked():
    bidir = mingru.Bidirectional(mingru.MinGRU(input_size=1, hidden_sizes=[3, 5]))

    x = torch.randn(2, 5, 1)
    h = bidir.init_hidden_state(x)
    assert h[0][0].shape == (2, 1, 3)
    assert h[0][1].shape == (2, 1, 5)
    assert h[1][0].shape == (2, 1, 3)
    assert h[1][1].shape == (2, 1, 5)

    out, h = bidir(x, h=h)
    assert len(out) == 2
    assert len(h) == 2
    assert out[0].shape == (2, 5, 5)
    assert out[1].shape == (2, 5, 5)
    assert h[0][0].shape == (2, 1, 3)
    assert h[0][1].shape == (2, 1, 5)
    assert h[1][0].shape == (2, 1, 3)
    assert h[1][1].shape == (2, 1, 5)


def test_bidirectional_scriptable():
    rnn = mingru.Bidirectional(mingru.MinGRUCell(input_size=1, hidden_size=5))
    scriptable.assert_scriptable(rnn, is_conv=False, is_bidir=True)

    rnn = mingru.Bidirectional(mingru.MinGRU(input_size=1, hidden_sizes=[3, 5]))
    scriptable.assert_scriptable(rnn, is_conv=False, is_bidir=True)
