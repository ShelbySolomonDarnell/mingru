"""PyTorch (convolutional) MinLSTM reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import abc
from typing import Final

import torch
import numpy as np

from . import functional as mF

class MinLSTMBase(torch.nn.Module, metaclass=abc.ABCMeta):
    """Common base interface for all MinLSTM implementations."""

    @abc.abstractmethod
    @torch.jit.export
    def init_hidden_state(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize 'zero' hidden and cell states."""

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
        c: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evaluate the MinLSTM."""

class MinLSTMCell(MinLSTMBase):
    """A minimal long short-term memory cell."""

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

        self.to_gates = torch.nn.Linear(
            input_size,
            hidden_size * 4,  # Compute input, forget, output, and cell outputs in tandem
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
        c: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evaluate the MinLSTM

        Params:
            x: (B,S,input_size) input features
            h: [(B,1,hidden_size),] optional previous hidden state
            c: [(B,1,hidden_size),] optional previous cell state

        Returns:
            out: (B,S,hidden_sizes) outputs of the last layer
            h': [(B,1,hidden_size),] next hidden state,
                corresponding to last sequence element of `out`.
            c': [(B,1,hidden_size),] next cell state,
                corresponding to last sequence element of `out`.
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)
        if c is None:
            c = self.init_hidden_state(x)

        gates = self.to_gates(x)
        input_gate, forget_gate, output_gate, cell_state = gates.chunk(4, dim=2)

        out, c_out = mF.minlstm_gate_hidden(input_gate, forget_gate, output_gate, cell_state, h[0], c[0])
        return out, [out[:, -1:]], [c_out[:, -1:]]

    def init_hidden_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns 'zero' hidden and cell states."""
        return [
            mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1])),
        ], [
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

            # Combined linear features for gates and cell state
            mdict["gates"] = torch.nn.Linear(
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
        h: list[torch.Tensor] | None = None,
        c: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evaluate the MinLSTM.

        Params:
            x: (B,S,input_size) input features
            h: optional list of tensors with shape (B,1,hidden_sizes[i])
                containing previous hidden states.
            c: optional list of tensors with shape (B,1,hidden_sizes[i])
                containing previous cell states.

        Returns:
            out: (B,S,hidden_sizes[-1]) outputs of the last layer
            h': list of tensors with shape (B,1,hidden_sizes[i]) containing
                next hidden states per layer.
            c': list of tensors with shape (B,1,hidden_sizes[i]) containing
                next cell states per layer.
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)
        if c is None:
            c = self.init_hidden_state(x)

        # input to next layer
        inp = x
        next_hidden = []
        next_cell = []

        # hidden states across layers
        for lidx, layer in enumerate(self.layers):
            h_prev = h[lidx]
            c_prev = c[lidx]
            gates = layer.gates(layer.norm(inp))
            input_gate, forget_gate, output_gate, cell_state = gates.chunk(4, dim=2)
            out, c_out = mF.minlstm_gate_hidden(input_gate, forget_gate, output_gate, cell_state, h_prev, c_prev)
            next_hidden.append(out[:, -1:])
            next_cell.append(c_out[:, -1:])

            # Add skip connection
            if self.residual:
                out = out + layer.res_align(inp)

            out = layer.dropout(out)
            inp = out

        return out, next_hidden, next_cell

    def init_hidden_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns a list of 'zero' hidden and cell states for each layer."""

        return [
            mF.g(x.new_zeros(x.shape[0], 1, hidden_size))
            for hidden_size in self.layer_sizes[1:]
        ], [
            mF.g(x.new_zeros(x.shape[0], 1, hidden_size))
            for hidden_size in self.layer_sizes[1:]
        ]

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
        """Initialize convolutional MinLSTM cell.

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

        self.to_gates = torch.nn.Conv2d(
            input_size,
            hidden_size * 4,
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
        c: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evaluate the convolutional MinLSTM

        Params:
            x: (B,S,input_size,H,W) input features
            h: [(B,1,hidden_size,H',W'),] optional previous
                hidden state features
            c: [(B,1,hidden_size,H',W'),] optional previous
                cell state features

        Returns:
            out: (B,S,hidden_sizes,H',W') outputs of the last layer
            h': [(B,1,hidden_size,H',W'),] next hidden state, corresponding
                to last element of `out`.
            c': [(B,1,hidden_size,H',W'),] next cell state, corresponding
                to last element of `out`.
        """

        assert (
            x.ndim == 5 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size,H,W)"

        if h is None:
            h = self.init_hidden_state(x)
        if c is None:
            c = self.init_hidden_state(x)

        B, S = x.shape[:2]
        gates = (
            self.to_gates(x.flatten(0, 1))
            .unflatten(
                0,
                (B, S),
            )
            .chunk(4, dim=2)
        )
        input_gate, forget_gate, output_gate, cell_state = gates

        out, c_out = mF.minlstm_gate_hidden(input_gate, forget_gate, output_gate, cell_state, h[0], c[0])
        return out, [out[:, -1:]], [c_out[:, -1:]]

    def init_hidden_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        B, S = x.shape[:2]
        with torch.no_grad():
            H, W = (
                self.to_gates(
                    x[:1, :1].flatten(0, 1),
                )
                .unflatten(0, (1, 1))
                .shape[3:]
            )
            return [
                mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1], H, W)),
            ], [
                mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1], H, W)),
            ]

class MinConv2dLSTM(MinLSTMBase):
    """A multi-layer minimal convolutional long short-term memory (MinLSTM)."""

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

            # Combined linear features for gates and cell state
            mdict["gates"] = torch.nn.Conv2d(
                ind,
                outd * 4,
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
                    y = mdict["gates"](x)
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
        c: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evaluate the MinLSTM.

        Params:
            x: (B,S,input_size,H,W) input of first layer
            h: optional previous/initial hidden states per layer.
                If not given a 'zero' initial state is allocated.
                Shape per layer $i$ is given by (B,1,hidden_size[i],H',W'),
                where W' and H' are determined by the convolution settings.
            c: optional previous/initial cell states per layer.
                If not given a 'zero' initial state is allocated.
                Shape per layer $i$ is given by (B,1,hidden_size[i],H',W'),
                where W' and H' are determined by the convolution settings.

        Returns:
            out: (B,S,hidden_dims,H',W') outputs of the last layer
            h': list of next hidden states per layer. Shape for layer $i$
                is given by (B,1,hidden_size[i],H',W'), where spatial
                dimensions are determined by convolutional settings.
            c': list of next cell states per layer. Shape for layer $i$
                is given by (B,1,hidden_size[i],H',W'), where spatial
                dimensions are determined by convolutional settings.
        """
        assert (
            x.ndim == 5 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)
        if c is None:
            c = self.init_hidden_state(x)

        B, S = x.shape[:2]
        inp = x
        next_hidden = []
        next_cell = []

        # hidden states across layers
        for lidx, layer in enumerate(self.layers):
            h_prev = h[lidx]
            c_prev = c[lidx]

            gates = (
                layer.gates(layer.norm(inp.flatten(0, 1)))
                .unflatten(0, (B, S))
                .chunk(4, dim=2)
            )
            input_gate, forget_gate, output_gate, cell_state = gates

            out, c_out = mF.minlstm_gate_hidden(input_gate, forget_gate, output_gate, cell_state, h_prev, c_prev)
            next_hidden.append(out[:, -1:])
            next_cell.append(c_out[:, -1:])

            # Add skip connection
            if self.residual:
                out = out + layer.res_align(inp.flatten(0, 1)).unflatten(0, (B, S))

            out = layer.dropout(out)
            inp = out

        return out, next_hidden, next_cell

    def init_hidden_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hs = []
        cs = []
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
                    layer.gates(x[:1, :1].flatten(0, 1))
                    .unflatten(
                        0,
                        (1, 1),
                    )
                    .chunk(4, dim=2)
                )
                h = mF.g(y.new_zeros(B, 1, y.shape[2], y.shape[3], y.shape[4]))
                c = mF.g(y.new_zeros(B, 1, y.shape[2], y.shape[3], y.shape[4]))
                hs.append(h)
                cs.append(c)
                x = y
        return hs, cs

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
        c: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Evaluate the Bidirectional LSTM."""

        if h is None:
            h = self.init_hidden_state(x)
        if c is None:
            c = self.init_hidden_state(x)

        h_fwd, h_bwd = h[: self.num_layers], h[self.num_layers :]
        c_fwd, c_bwd = c[: self.num_layers], c[self.num_layers :]

        out_fwd, h_fwd, c_fwd = self.rnn(x, h=h_fwd, c=c_fwd)
        out_bwd, h_bwd, c_bwd = self.rnn(torch.flip(x, dims=(1,)), h=h_bwd, c=c_bwd)

        return torch.cat((out_fwd, out_bwd), 2), h_fwd + h_bwd, c_fwd + c_bwd

    def init_hidden_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize bidirectional hidden and cell states"""
        h_fwd, c_fwd = self.rnn.init_hidden_state(x)
        h_bwd, c_bwd = self.rnn.init_hidden_state(x)
        return h_fwd + h_bwd, c_fwd + c_bwd

__all__ = ["MinLSTMCell", "MinLSTM", "MinConv2dLSTMCell", "MinConv2dLSTM", "Bidirectional"]