"""PyTorch (convolutional) MinLSTM reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""
import sys
import torch
import torch.nn.functional as F

from .scan import parallel_scan_log
from torch.nn import Linear


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

    This function takes gate and hidden outputs directly,
    as MinLSTM forward is equal for convolutional/standard
    MinLSTM from this point on.

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
    diff = F.softplus(-forget_gate) - F.softplus(-input_gate)
    log_f = -F.softplus(diff)
    log_i = -F.softplus(-diff)
    log_c_0 = c.log()

    log_tilde_c = torch.tanh(cell_state).log()
    c_next = parallel_scan_log(
        log_f, 
        torch.cat([log_c_0, log_i + log_tilde_c], dim=1))
    c_next = c_next[:, 1:]  # tail
    
    h_next = torch.sigmoid(output_gate) * torch.tanh(c_next)
    
    return h_next, c_next

def _minlstm_sequential(
    h: torch.Tensor,
    c: torch.Tensor,
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    output_gate: torch.Tensor,
    cell_state: torch.Tensor,
):
    """Sequential MinLSTM forward.

    This function takes gate and hidden outputs directly,
    as MinLSTM forward is equal for convolutional/standard
    MinLSTM from this point on.

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
    i_t = torch.sigmoid(input_gate)
    f_t = torch.sigmoid(forget_gate)
    o_t = torch.sigmoid(output_gate)
    c_tilde = torch.tanh(cell_state)
    
    c_next = f_t * c + i_t * c_tilde
    h_next = o_t * torch.tanh(c_next)
    
    return h_next, c_next

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
        input_gate: (B,S,hidden_dims,*) input gate outputs
        forget_gate: (B,S,hidden_dims,*) forget gate outputs
        output_gate: (B,S,hidden_dims,*) output gate outputs
        cell_state: (B,S,hidden_dims,*) cell state outputs
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
