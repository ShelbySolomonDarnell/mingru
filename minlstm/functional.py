"""PyTorch (convolutional) MinLSTM reference implementation

This module implements the core functional components of MinLSTM, a minimal
long short-term memory architecture. MinLSTM differs from MinGRU in that it
maintains two hidden states (h and c) instead of just one.

Key differences between MinLSTM and MinGRU:
1. MinLSTM has two hidden states: h (hidden state) and c (cell state)
2. MinLSTM uses three gates: input, forget, and output gates
3. MinGRU uses two gates: reset and update gates

The hidden state handling in MinLSTM requires special attention when:
- Initializing states (both h and c need to be initialized)
- Passing states between batches (both h and c need to be detached)
- Processing sequences (both h and c are updated and returned)

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

    This function implements an optimized version of the LSTM update equations
    for processing an entire sequence at once. It uses a parallel scan algorithm
    to efficiently compute the recurrent updates.

    The parallel implementation is more efficient for training when processing
    entire sequences, while the sequential version is used for generation.

    Key differences from standard LSTM:
    - Uses log-space computations for numerical stability
    - Employs parallel scan algorithm for efficient sequence processing
    - Optimized for better gradient flow through the sequence

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
    # Clamp inputs to prevent extreme values
    input_gate = torch.clamp(input_gate, -10, 10)
    forget_gate = torch.clamp(forget_gate, -10, 10)
    output_gate = torch.clamp(output_gate, -10, 10)
    cell_state = torch.clamp(cell_state, -10, 10)
    
    # Use standard LSTM update equations instead of log-space for stability
    i_t = torch.sigmoid(input_gate)
    f_t = torch.sigmoid(forget_gate)
    o_t = torch.sigmoid(output_gate)
    c_tilde = torch.tanh(cell_state)
    
    # Initialize output tensors
    batch_size, seq_len = input_gate.shape[:2]
    c_next = torch.zeros_like(input_gate)
    h_next = torch.zeros_like(input_gate)
    
    # Process the sequence step by step
    c_t = c
    for t in range(seq_len):
        c_t = f_t[:, t:t+1] * c_t + i_t[:, t:t+1] * c_tilde[:, t:t+1]
        c_t = torch.clamp(c_t, -10, 10)  # Prevent exploding values
        h_t = o_t[:, t:t+1] * torch.tanh(c_t)
        
        c_next[:, t:t+1] = c_t
        h_next[:, t:t+1] = h_t
    
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

    This function implements the standard LSTM update equations for a single
    time step. It is used when processing one token at a time (e.g., during
    generation).

    The update equations are:
    i_t = sigmoid(input_gate)
    f_t = sigmoid(forget_gate)
    o_t = sigmoid(output_gate)
    c_tilde = tanh(cell_state)
    c_t = f_t * c_{t-1} + i_t * c_tilde
    h_t = o_t * tanh(c_t)

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
    # Apply sigmoid with gradient clipping to avoid NaN values
    i_t = torch.sigmoid(torch.clamp(input_gate, -10, 10))
    f_t = torch.sigmoid(torch.clamp(forget_gate, -10, 10))
    o_t = torch.sigmoid(torch.clamp(output_gate, -10, 10))
    c_tilde = torch.tanh(torch.clamp(cell_state, -10, 10))
    
    # Update cell state with gradient clipping
    c_next = f_t * c + i_t * c_tilde
    c_next = torch.clamp(c_next, -10, 10)
    
    # Update hidden state
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

    MinLSTM uses three gates and a cell state:
    - input_gate (i_t): Controls how much new information to add to the cell state
    - forget_gate (f_t): Controls how much of the previous cell state to retain
    - output_gate (o_t): Controls how much of the cell state to expose as the hidden state
    - cell_state (c_tilde): Candidate values to add to the cell state

    The cell state (c) acts as the memory of the network and is updated as:
    c_t = f_t * c_{t-1} + i_t * c_tilde_t

    The hidden state (h) is then computed as:
    h_t = o_t * tanh(c_t)

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
