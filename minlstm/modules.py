"""PyTorch (convolutional) MinLSTM reference implementation

Christoph Heind, 2024
https://github.com/cheind/minlstm
"""

import abc
import sys
from typing import Final

import torch
import numpy as np

from . import functional as mF
from torch.nn import Linear 
from torch.nn.functional import pad

class MinLSTMBase(torch.nn.Module, metaclass=abc.ABCMeta):
    """Common base interface for all MinLSTM implementations."""

    @abc.abstractmethod
    @torch.jit.export
    def init_hidden_state(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize a 'zero' hidden state."""

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        h: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Evaluate the MinLSTM."""
        
    @torch.jit.export
    def forward_with_separate_states(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor],
        c: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass with separate h and c states for TorchScript compatibility.
        
        This method is a wrapper around forward() that takes h and c separately
        and returns them separately, avoiding TorchScript type issues.
        
        Args:
            x: Input tensor
            h: Hidden state list
            c: Cell state list
            
        Returns:
            Tuple of (output, next_h, next_c)
        """
        output, (next_h, next_c) = self.forward(x, (h, c))
        return output, next_h, next_c


class MinLSTMCell(MinLSTMBase):
    """A minimal long short-term memory recurrent unit cell."""

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize MinLSTM cell.

        Params:
            input_size: number of expected features in input
            hidden_size: number of features in hidden state
            bias: If false, no bias weights will be allocated
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        self.to_gate_hidden = torch.nn.Linear(
            input_size,
            hidden_size * 2,  # Compute gate/hidden outputs in tandem
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate the MinLSTM

        Params:
            x: (B,S,input_size) input features
            h: [(B,1,hidden_size),] optional previous hidden state
                features

        Returns:
            out: (B,S,hidden_sizes) outputs of the last layer
            h': [(B,1,hidden_size),] next hidden state,
                corresponding to last sequence element of `out`.
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)

        gate, hidden = self.to_gate_hidden(x).chunk(2, dim=2)

        out = mF.mingru_gate_hidden(gate, hidden, h[0])
        return out, [out[:, -1:]]

    def init_hidden_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns a 'zero' hidden state."""
        return [
            mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1])),
        ]


class MinLSTM(MinLSTMBase):
    """A multi-layer minimal long short-term memory (MinLSTM)."""

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]
    residual: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        *,
        bias: bool = True,
        norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize MinLSTM cell.

        Params:
            input_size: number of expected features in input
            hidden_sizes: list of number of features in each stacked hidden
                state
            bias: If false, no bias weights will be allocated in linear layers
            norm: If true, adds layer normalization to inputs of each layer.
            dropout: If > 0, dropout will be applied to each layer input,
                except for last layer.
            residual: If true, residual connections will be added between each
                layer. If the input/output sizes are different, linear
                adjustment layers will be added.
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """

        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # Convert hidden_sizes to list if it's a tuple
        if isinstance(hidden_sizes, tuple):
            hidden_sizes = list(hidden_sizes)
        self.layer_sizes = tuple([input_size] + hidden_sizes)
        self.num_layers = len(hidden_sizes)
        self.dropout = max(min(dropout, 1.0), 0.0)
        self.residual = residual
        self.norm = norm

        layers = []
        factory_kwargs = {"device": device, "dtype": dtype}
        gen = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        for lidx, (ind, outd) in enumerate(gen):
            # Gate and hidden linear features
            mdict = {}

            if norm:
                mdict["norm"] = torch.nn.LayerNorm(ind)
            else:
                mdict["norm"] = torch.nn.Identity()

            # Combined linear features for input gate, forget gate, output gate, and cell state
            mdict["gate_hidden"] = torch.nn.Linear(
                ind, outd * 4, bias=bias, **factory_kwargs
            )

            # Residual alignment layer if features size mismatch
            if residual and ind != outd:
                mdict["res_align"] = torch.nn.Linear(
                    ind, outd, bias=False, **factory_kwargs
                )
            else:
                mdict["res_align"] = torch.nn.Identity()

            # Dropout for outputs except for last
            if dropout > 0.0 and lidx < (self.num_layers - 1):
                mdict["dropout"] = torch.nn.Dropout(p=dropout)
            else:
                mdict["dropout"] = torch.nn.Identity()

            layers.append(torch.nn.ModuleDict(mdict))
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        h: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Evaluate the MinLSTM.

        Params:
            x: (B,S,input_size) input features
            h: optional tuple of (h_states, c_states) where each is a list of tensors
               with shape (B,1,hidden_sizes[i]) containing previous hidden states.

        Returns:
            out: (B,S,hidden_sizes[-1]) outputs of the last layer
            h': tuple of (h_states, c_states) where each is a list of tensors
                with shape (B,1,hidden_sizes[i]) containing next hidden states per layer.
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h_states, c_states = self.init_hidden_state(x)
        else:
            h_states, c_states = h

        # input to next layer
        inp = x
        next_h_states = []
        next_c_states = []

        # hidden states across layers
        for lidx, layer in enumerate(self.layers):
            # Get previous hidden and cell states
            h_prev = h_states[lidx] if lidx < len(h_states) else None
            c_prev = c_states[lidx] if lidx < len(c_states) else None
            
            # Initialize if not available
            if h_prev is None or c_prev is None:
                h_prev = mF.g(inp.new_zeros(inp.shape[0], 1, self.layer_sizes[lidx+1]))
                c_prev = inp.new_zeros(inp.shape[0], 1, self.layer_sizes[lidx+1])
                
            # Split into input gate, forget gate, output gate, and cell state
            gates_and_cell = layer.gate_hidden(layer.norm(inp))
            input_gate, forget_gate, output_gate, cell_state = gates_and_cell.chunk(4, dim=2)
            
            # Apply layer normalization to gates for stability
            input_gate = torch.clamp(input_gate, -10, 10)
            forget_gate = torch.clamp(forget_gate, -10, 10)
            output_gate = torch.clamp(output_gate, -10, 10)
            cell_state = torch.clamp(cell_state, -10, 10)
            
            # Process the cell state
            out, c_next = mF.minlstm_gate_hidden(
                input_gate, 
                forget_gate, 
                output_gate, 
                cell_state, 
                h_prev,  # hidden state
                c_prev   # cell state
            )
            next_h_states.append(out[:, -1:])
            next_c_states.append(c_next[:, -1:])

            # Add skip connection
            if self.residual:
                out = out + layer.res_align(inp)

            out = layer.dropout(out)
            inp = out

        return out, (next_h_states, next_c_states)

    def init_hidden_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns a tuple of lists of 'zero' hidden and cell states for each layer."""

        batch_size = x.shape[0]
        h_states = []
        c_states = []
        
        for hidden_size in self.layer_sizes[1:]:
            # Hidden state (h)
            h_states.append(torch.zeros(batch_size, 1, hidden_size, device=x.device, dtype=x.dtype))
            # Cell state (c)
            c_states.append(torch.zeros(batch_size, 1, hidden_size, device=x.device, dtype=x.dtype))
            
        return h_states, c_states


class MinConv2dLSTMCell(MinLSTMBase):
    """A minimal convolutional long short-term memory cell."""

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize convolutional MinGRU cell.

        Params:
            input_size: number of expected features in input
            hidden_size: number of features in hidden state
            kernel_size: kernel size of convolutional layer
            stride: stride in convolutional layer
            padding: padding in convolutional layer
            bias: If false, no bias weights will be allocated
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """

        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "bias": bias,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        }

        self.to_gate_hidden = torch.nn.Conv2d(
            input_size,
            hidden_size * 2,
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate the convolutional MinGRU

        Params:
            x: (B,S,input_size,H,W) input features
            h: [(B,1,hidden_size,H',W'),] optional previous
                hidden state features

        Returns:
            out: (B,S,hidden_sizes,H',W') outputs of the last layer
            h': [(B,1,hidden_size,H',W'),] next hidden state, corresponding
                to last element of `out`.
        """

        assert (
            x.ndim == 5 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size,H,W)"

        if h is None:
            h = self.init_hidden_state(x)

        B, S = x.shape[:2]
        gate, hidden = (
            self.to_gate_hidden(x.flatten(0, 1))
            .unflatten(
                0,
                (B, S),
            )
            .chunk(2, dim=2)
        )

        out = mF.mingru_gate_hidden(gate, hidden, h[0])
        return out, [out[:, -1:]]

    def init_hidden_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, S = x.shape[:2]
        with torch.no_grad():
            H, W = (
                self.to_gate_hidden(
                    x[:1, :1].flatten(0, 1),
                )
                .unflatten(0, (1, 1))
                .shape[3:]
            )
            return [
                mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1], H, W)),
            ]


class MinConv2dLSTM(MinLSTMBase):
    """A multi-layer minimal convolutional gated recurrent unit (MinGRU)."""

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]
    residual: Final[bool]
    dropout: Final[float]

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        kernel_sizes: int | list[int],
        *,
        strides: int | list[int] = 1,
        paddings: int | list[int] = 0,
        dropout: float = 0.0,
        residual: bool = False,
        bias: bool = True,
        norm: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize convolutional MinLSTM cell.

        Params:
            input_size: number of expected features in input
            hidden_sizes: list containing the number of
                output features per hidden layer.
            kernel_sizes: kernel sizes per convolutional layer
            strides: strides per convolutional layer
            paddings: paddings per convolutional layer
            residual: If true, skip connections between each layer
                are added. If spatials or feature dimensions mismatch,
                necessary alignment convolutions are added.
            bias: If false, no bias weights will be allocated
            norm: If true, applies group normalization to inputs of each layer.
                The number of groups is maximized as long as more than
                4 channels per group are left. See
                https://arxiv.org/pdf/1803.08494
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(hidden_sizes)

        if isinstance(strides, int):
            strides = [strides] * len(hidden_sizes)

        if isinstance(paddings, int):
            paddings = [paddings] * len(hidden_sizes)

        self.layer_sizes = tuple([input_size] + hidden_sizes)
        self.num_layers = len(hidden_sizes)
        self.dropout = max(min(dropout, 1.0), 0.0)
        self.residual = residual

        factory_kwargs = {"device": device, "dtype": dtype}
        layers = []

        gen = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        for lidx, (ind, outd) in enumerate(gen):
            # Gate and hidden linear features
            mdict = {}

            if norm:
                groups = [64, 32, 16, 8, 4, 1]
                states = [(ind % g == 0) and (ind // g) >= min(4, ind) for g in groups]
                choice = np.where(states)[0][0]
                mdict["norm"] = torch.nn.GroupNorm(groups[choice], ind)
            else:
                mdict["norm"] = torch.nn.Identity()

            # Combined linear features for gate and hidden
            mdict["gate_hidden"] = torch.nn.Conv2d(
                ind,
                outd * 2,
                bias=bias,
                kernel_size=kernel_sizes[lidx],
                stride=strides[lidx],
                padding=paddings[lidx],
                **factory_kwargs,
            )

            # Residual alignment layer if features size or
            # spatial dims mismatch
            if residual:
                with torch.no_grad():
                    x = torch.randn(1, ind, 16, 16)
                    y = mdict["gate_hidden"](x)
                if ind != outd or x.shape[2:] != y.shape[2:]:
                    mdict["res_align"] = torch.nn.Conv2d(
                        ind,
                        outd,
                        bias=False,
                        kernel_size=kernel_sizes[lidx],
                        stride=strides[lidx],
                        padding=paddings[lidx],
                        **factory_kwargs,
                    )
                else:
                    mdict["res_align"] = torch.nn.Identity()

            # Dropout for outputs except for last
            if dropout > 0.0 and lidx < (self.num_layers - 1):
                mdict["dropout"] = torch.nn.Dropout(p=dropout)
            else:
                mdict["dropout"] = torch.nn.Identity()

            layers.append(torch.nn.ModuleDict(mdict))
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate the MinGRU.

        Params:
            x: (B,S,input_size,H,W) input of first layer
            h: optional previous/initial hidden states per layer.
                If not given a 'zero' initial state is allocated.
                Shape per layer $i$ is given by (B,1,hidden_size[i],H',W'),
                where W' and H' are determined by the convolution settings.

        Returns:
            out: (B,S,hidden_dims,H',W') outputs of the last layer
            h': list of next hidden states per layer. Shape for layer $i$
                is given by (B,1,hidden_size[i],H',W'), where spatial
                dimensions are determined by convolutional settings.
        """
        assert (
            x.ndim == 5 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)

        B, S = x.shape[:2]
        inp = x
        next_hidden = []

        # hidden states across layermingru.s
        for lidx, layer in enumerate(self.layers):
            h_prev = h[lidx]

            gate, hidden = (
                layer.gate_hidden(layer.norm(inp.flatten(0, 1)))
                .unflatten(0, (B, S))
                .chunk(2, dim=2)
            )

            out = mF.mingru_gate_hidden(gate, hidden, h_prev)
            next_hidden.append(out[:, -1:])

            # Add skip connection
            if self.residual:
                out = out + layer.res_align(inp.flatten(0, 1)).unflatten(0, (B, S))

            out = layer.dropout(out)
            inp = out

        return out, next_hidden

    def init_hidden_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        hs = []
        B = x.shape[0]
        # We dynamically determine the required hidden shapes, to avoid
        # fiddling with spatial dimension computation. This just uses
        # the first sequence element from the first batch todo so, and hence
        # should not lead to major performance impact.

        # Cannot make the following a reusable function because
        # nn.Modules are not accepted as parameters in scripting...
        with torch.no_grad():
            for layer in self.layers:
                y, _ = (
                    layer.gate_hidden(x[:1, :1].flatten(0, 1))
                    .unflatten(
                        0,
                        (1, 1),
                    )
                    .chunk(2, dim=2)
                )
                h = mF.g(y.new_zeros(B, 1, y.shape[2], y.shape[3], y.shape[4]))
                hs.append(h)
                x = y
        return hs


class Bidirectional(MinLSTMBase):
    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]

    def __init__(self, rnn: MinLSTMBase):
        super().__init__()
        self.rnn = rnn
        self.layer_sizes = rnn.layer_sizes
        self.num_layers = rnn.num_layers

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate the Bidirectional GRU."""

        if h is None:
            h = self.init_hidden_state(x)

        h_fwd, h_bwd = h[: self.num_layers], h[self.num_layers :]

        out_fwd, h_fwd = self.rnn(x, h=h_fwd)
        out_bwd, h_bwd = self.rnn(torch.flip(x, dims=(1,)), h=h_bwd)

        return torch.cat((out_fwd, out_bwd), 2), h_fwd + h_bwd

    def init_hidden_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Initialize bidirectional hidden state"""
        h_fwd = self.rnn.init_hidden_state(x)
        h_bwd = self.rnn.init_hidden_state(x)
        return h_fwd + h_bwd


__all__ = ["MinLSTMCell", "MinLSTM", "MinConv2dLSTMCell", "MinConv2dLSTM", "Bidirectional"]
# This is a duplicate class definition at the end of the file - removing it
