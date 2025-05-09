import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import deepspeed
import argparse

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)

def get_args():
    parser = argparse.ArgumentParser(
        description='DeepSpeed TP training script'
    )
    # Allow DeepSpeed's launcher to set the local rank
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Add DeepSpeed's built‑in arguments (e.g., --deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--tp_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--train_steps', type=int, default=10,
                        help='number of training steps')
    return parser.parse_args()

def main():
    args = get_args()

    # Instantiate model and move parameters under DeepSpeed control
    model = SimpleModel()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # DeepSpeed initialization: returns engine wrapping model & optimizer
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config_params={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "tensor_parallel": {
                "tp": {
                    "tp_size": args.tp_size,
                    "tp_grain_size": 64
                }
            }
        }
    )

    # Training loop over random data
    for step in range(args.train_steps):
        # Create dummy input and target
        x = torch.randn(1, 8).to(model_engine.device)
        target = torch.randn(1, 8).to(model_engine.device)

        # Forward pass
        output = model_engine(x)

        # Compute simple MSE loss
        loss = nn.functional.mse_loss(output, target)

        # Backward pass (handles gradient accumulation, communication, etc.)
        model_engine.backward(loss)

        # Optimizer step (updates weights, zeroes grads, updates LR scheduler)
        model_engine.step()

        if model_engine.local_rank == 0:
            print(f"[Step {step:2d}] loss = {loss.item():.6f}")

if __name__ == "__main__":
    main()