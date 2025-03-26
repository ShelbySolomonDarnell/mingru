"""Example script for character-level MinGRU

This script demonstrates how to use the character-level MinGRU implementation.
"""

import torch
import logging
from pathlib import Path

from char_mingru.model import CharNLPModel
from char_mingru.dataset import CharTokenDataset
from char_mingru.train import train, generate_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

_logger = logging.getLogger("char_mingru_example")

def main():
    """Run the example."""
    # Configuration
    cfg = {
        "optim": "adamw",
        "num_epochs": 3,
        "batch_size": 64,
        "lr": 0.001,
        "seqlen": 100,
        "emb_size": 128,
        "hidden_sizes": [128, 256, 512],
        "dropout": 0.1,
        "norm": True,
        "wandb": False,
    }
    
    # Path to text file
    textfile = "~/Datasets/tiny-shakespeare/train_coriolanus.csv.90percent"
    textfile = Path(textfile).expanduser()
    
    # Load dataset
    _logger.info(f"Loading dataset from {textfile}")
    train_ds, val_ds = CharTokenDataset.from_textfile(textfile, cfg["seqlen"])
    
    # Update configuration with vocabulary size
    cfg["vocab_size"] = train_ds.vocab_size
    
    _logger.info(f"Vocabulary size: {cfg['vocab_size']}")
    _logger.info(f"Training with configuration: {cfg}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharNLPModel(cfg).to(device)
    
    # Train model
    train(model, train_ds, val_ds, cfg, device)
    
    # Generate text
    sample_text = generate_text(
        model,
        "\n",
        train_ds.idx_to_char,
        train_ds.char_to_idx,
        num_chars=200,
        device=device,
    )
    
    _logger.info(f"Sample text:\n{sample_text}")


if __name__ == "__main__":
    main()
