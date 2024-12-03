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
    assert out.shape == (2, 5, 5 * 2)
    assert h[0].shape == (2, 1, 5)
    assert h[1].shape == (2, 1, 5)
    assert not torch.allclose(out[..., :5], out[..., 5:])
    assert not torch.allclose(h[0], h[1])


def test_bidirectional_stacked():
    bidir = mingru.Bidirectional(mingru.MinGRU(input_size=1, hidden_sizes=[3, 5]))

    x = torch.randn(4, 5, 1)
    h = bidir.init_hidden_state(x)
    assert len(h) == (4)
    assert h[0].shape == (4, 1, 3)
    assert h[1].shape == (4, 1, 5)
    assert h[2].shape == (4, 1, 3)
    assert h[3].shape == (4, 1, 5)

    out, h = bidir(x, h=h)
    assert out.shape == (4, 5, 5 * 2)
    assert h[0].shape == (4, 1, 3)
    assert h[1].shape == (4, 1, 5)
    assert h[2].shape == (4, 1, 3)
    assert h[3].shape == (4, 1, 5)


def test_bidirectional_scriptable():
    rnn = mingru.Bidirectional(mingru.MinGRUCell(input_size=1, hidden_size=5))
    scriptable.assert_scriptable(rnn, is_conv=False)

    rnn = mingru.Bidirectional(mingru.MinGRU(input_size=1, hidden_sizes=[3, 5]))
    scriptable.assert_scriptable(rnn, is_conv=False)
