"""PyTorch (convolutional) MinLSTM reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import torch
import torch.nn.functional as F

from mingru.scan import parallel_scan_log

def g(x: torch.Tensor) -> torch.Tensor:
    """Activation function for hidden state
    Ensures that g(h) is non-negative and hence avoids
    complex numbers in log-space.
    """
    out = torch.empty_like(x)
    mask = x >= 0
    out[mask] = x[mask] + 0.5
    out[~mask] = torch.sigmoid(x[~mask])
    return out

def log_g(x: torch.Tensor) -> torch.Tensor:
    """Log-activation function for hidden state
    Ensures that g(h) is non-negative and hence avoids
    complex numbers in log-space.
    """
    out = torch.empty_like(x)
    mask = x >= 0
    out[mask] = (x[mask] + 0.5).log()
    out[~mask] = -F.softplus(-x[~mask])
    return out

def _minlstm_parallel(
    h: torch.Tensor,
    c: torch.Tensor,
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    output_gate: torch.Tensor,
    cell_state: torch.Tensor,
):
    """Parallel MinLSTM forward

    This function works for any number of spatial dimensions,
    which is indicated by `*` below.

    Params:
        h: (B,1,hidden_dims,*) previous hidden state
        c: (B,1,hidden_dims,*) previous cell state
        input_gate: (B,S,hidden_dims,*) input gate outputs
        forget_gate: (B,S,hidden_dims,*) forget gate outputs
        output_gate: (B,S,hidden_dims,*) output gate outputs
        cell_state: (B,S,hidden_dims,*) cell state outputs

    Returns:
        h: (B,S,hidden_dims,*) hidden states
        c: (B,S,hidden_dims,*) cell states
    """

    log_input_gate = -F.softplus(-input_gate)  # log(i)
    log_forget_gate = -F.softplus(-forget_gate)  # log(f)
    log_output_gate = -F.softplus(-output_gate)  # log(o)
    log_cell_state = log_g(cell_state)

    log_c_0 = c.log()
    log_c = parallel_scan_log(
        log_forget_gate,
        torch.cat((log_c_0, log_input_gate + log_cell_state), dim=1),
    )
    c = log_c[:, 1:].exp()  # tail

    #h = torch.sigmoid(output_gate) * g(c)
    h = log_output_gate * g(c)
    return h, c

def _minlstm_sequential(
    h: torch.Tensor,
    c: torch.Tensor,
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    output_gate: torch.Tensor,
    cell_state: torch.Tensor,
):
    """Sequential MinLSTM forward.

    This function works for any number of spatial dimensions,
    which is indicated by `*` below.

    Params:
        h: (B,1,hidden_dims,*) previous hidden state
        c: (B,1,hidden_dims,*) previous cell state
        input_gate: (B,1,hidden_dims,*) input gate outputs
        forget_gate: (B,1,hidden_dims,*) forget gate outputs
        output_gate: (B,1,hidden_dims,*) output gate outputs
        cell_state: (B,1,hidden_dims,*) cell state outputs

    Returns:
        h: (B,1,hidden_dims,*) next hidden state
        c: (B,1,hidden_dims,*) next cell state
    """

    i = torch.sigmoid(input_gate)
    f = torch.sigmoid(forget_gate)
    o = torch.sigmoid(output_gate)
    c_tilde = g(cell_state)

    c_t = f * c + i * c_tilde
    h_t = o * g(c_t)
    return h_t, c_t

def minlstm_gate_hidden(
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    output_gate: torch.Tensor,
    cell_state: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
):
    """Evaluate the (convolutional) MinLSTM

    This method is the main entry point to evaluate the MinLSTM. It
    works for both convolutional and non-convolutional MinLSTMs.

    The code chooses sequential and parallel forward
    depending on the size of the sequence dimension S.

    Params:
        input_gate: (B,1,hidden_dims,*) input gate outputs
        forget_gate: (B,1,hidden_dims,*) forget gate outputs
        output_gate: (B,1,hidden_dims,*) output gate outputs
        cell_state: (B,1,hidden_dims,*) cell state outputs
        h: (B,1,hidden_dims,*) previous hidden state
        c: (B,1,hidden_dims,*) previous cell state

    Returns:
        h: (B,S,hidden_dims,*) next hidden states
        c: (B,S,hidden_dims,*) next cell states
    """

    if input_gate.shape[1] == 1:
        return _minlstm_sequential(h, c, input_gate, forget_gate, output_gate, cell_state)
    else:
        return _minlstm_parallel(h, c, input_gate, forget_gate, output_gate, cell_state)

__all__ = ["minlstm_gate_hidden", "g", "log_g"]