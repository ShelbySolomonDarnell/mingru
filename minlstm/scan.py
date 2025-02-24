"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import torch
import torch.nn.functional as F


def parallel_scan_log(log_a: torch.Tensor, log_b: torch.Tensor):
    """Parallel scan in log-space.

    Efficiently computes
        x_t = a_t*x_{t-1} + b_t
    for non-negative numbers.

    Params:
        log_a: (B,T,*) log-coefficients for timestep 1..T
        log_b: (B,T+1,*) log-values of b including x_0

    Returns:
        x: (B,T+1,*) sequence values computed in parallel.

    Based on:
        Efficient Parallelization of a Ubiquitous Sequential Computation
        Franz A. Heinsen, 2023, https://arxiv.org/pdf/2311.06281
    """
    a_star = F.pad(
        torch.cumsum(log_a, dim=1),
        [0] * (log_a.ndim - 2) * 2 + [1, 0],
    )
    x0_plus_b_star = torch.logcumsumexp(log_b - a_star, dim=1)
    log_x = a_star + x0_plus_b_star
    return torch.exp(log_x)


__all__ = ["parallel_scan_log"]
