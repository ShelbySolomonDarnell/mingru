"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

import mingru


# Taken from
# https://github.com/MinhZou/selective-copying-mamba/blob/main/data_generator.py
def torch_copying_data(
    L, M, A, variable=False, batch_shape=(), one_hot=False, reverse=False
):
    """
    Generate a dataset for a sequence copying task.
    This code is adopted from the copying.py script in the S4 repository.
    The original code can be found at:
    https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

    Params:
        L (int): Number of padding tokens
        M (int): Number of tokens to memorize
        A (int): Alphabet size
        variable (bool): If True, selective copying task
        batch_shape (tuple): Shape of the batch
        one_hot (bool): If True, convert the input sequence into a one-hot
            encoded tensor
        reverse (bool): If True, reverse the order of the target sequence

    Returns:
        x: generated sequence
        y: target sequence
    """
    tokens = torch.randint(low=1, high=A - 1, size=batch_shape + (M,))
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack(
            [torch.randperm(L + M)[:M] for _ in range(total_batch)],
            0,
        )
        inds = inds.reshape(batch_shape + (M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape + (1,))
    zeros_x = torch.zeros(batch_shape + (M + L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A - 1) * torch.ones(batch_shape + (M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse:
        y_ = y_.flip(-1)
    if one_hot:
        x = F.one_hot(x_, A).float()
    else:
        x = x_
    y = y_
    return x, y


class SelectiveCopyingModel(torch.nn.Module):
    num_memorize: Final[int]

    def __init__(self, cfg: dict):
        super().__init__()
        self.emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dims"])

        self.rnn = mingru.MinGRU(
            input_size=cfg["emb_dims"],
            hidden_sizes=cfg["hidden_sizes"],
            residual=True,
            norm=cfg["norm"],
        )
        self.logits = torch.nn.Linear(
            cfg["hidden_sizes"][-1],
            cfg["vocab_size"],
        )
        self.num_memorize = cfg["num_memorize"]

    def forward(self, x: torch.Tensor):
        out, h = self.rnn(self.emb(x))
        return self.logits(out)[:, -self.num_memorize :]  # NOQA


def train(cfg: dict):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveCopyingModel(cfg).to(dev)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for step in range(cfg["num_steps"]):
        inputs, targets = torch_copying_data(
            cfg["seq_len"],
            cfg["num_memorize"],
            cfg["vocab_size"],
            variable=True,
            batch_shape=(cfg["batch_size"],),
            one_hot=False,
            reverse=False,
        )
        inputs = inputs.to(dev)
        targets = targets.to(dev)
        outputs = model(inputs).permute(0, 2, 1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss = loss.item()
        total = targets.size(0) * targets.size(1)
        correct = (outputs.argmax(1) == targets).sum().item()
        accuracy = 100 * correct / total
        if step % 20 == 0:
            print(
                f'Step [{step+1}/{cfg["num_steps"]}], Loss: {step_loss/cfg["batch_size"]:.4f}, Accuracy: {accuracy:.2f}%'  # NOQA
            )

    return model


@torch.no_grad()
def validate(cfg, model):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs, targets = inputs, targets = torch_copying_data(
        cfg["seq_len"],
        cfg["num_memorize"],
        cfg["vocab_size"],
        variable=True,
        batch_shape=(cfg["batch_size"],),
        one_hot=False,
        reverse=False,
    )
    inputs = inputs.to(dev)
    targets = targets.to(dev)
    outputs = model(inputs).permute(0, 2, 1)
    total = targets.size(0) * targets.size(1)
    correct = (outputs.argmax(1) == targets).sum().item()
    accuracy = 100 * correct / total

    print(f"Validation Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":

    cfg = {
        "seq_len": 64,
        "num_memorize": 4,
        "vocab_size": 6,
        "emb_dims": 4,
        "hidden_sizes": [64, 64, 64],
        "norm": True,
        "batch_size": 64,
        "num_steps": 2000,
        "lr": 1e-3,
    }
    model = train(cfg)

    # Script and eval the model
    scripted = torch.jit.script(model)
    validate(cfg, scripted)
