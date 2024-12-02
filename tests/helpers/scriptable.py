import torch
import io


def assert_same_outputs(actual, expected):
    if isinstance(expected, tuple):
        # for non-cells
        rnn_out, rnn_h = expected
        scripted_out, scripted_h = actual

        assert torch.allclose(scripted_out, rnn_out, atol=1e-4)
        assert isinstance(scripted_h, (list, tuple))
        for i in range(len(rnn_h)):
            assert torch.allclose(scripted_h[i], rnn_h[i], atol=1e-4)
    else:
        # for cells
        assert torch.allclose(actual, expected, atol=1e-4)


def assert_scriptable(rnn: torch.nn.Module, is_conv: bool):

    if is_conv:
        x = torch.randn(1, 10, rnn.layer_sizes[0], 32, 32)
    else:
        x = torch.randn(1, 128, rnn.layer_sizes[0])

    h = rnn.init_hidden_state(x)
    scripted = torch.jit.script(rnn)
    h_scripted = rnn.init_hidden_state(x)

    scripted_out = scripted(x, h_scripted)
    rnn_out = rnn(x, h)

    assert_same_outputs(scripted_out, rnn_out)

    scripted_out = scripted(x)
    rnn_out = rnn(x)

    assert_same_outputs(scripted_out, rnn_out)

    # Save load
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)

    buffer.seek(0)
    loaded = torch.jit.load(buffer, map_location=torch.device("cpu"))
    loaded_out = loaded(x)
    rnn_out = rnn(x)
    assert_same_outputs(loaded_out, rnn_out)
