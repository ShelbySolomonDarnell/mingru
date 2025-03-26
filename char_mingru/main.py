"""Main script for training and using character-level MinGRU models

This script provides command-line functionality for training and using
character-level MinGRU models.
"""

import argparse
import logging
import torch
import json
from pathlib import Path

from .model import CharNLPModel
from .dataset import CharTokenDataset
from .train import train, validate, generate_text
from .utils import save_model, load_model

_logger = logging.getLogger("char_mingru")

def main():
    """Main function for the character-level MinGRU implementation."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Character-level MinGRU")
    subparsers = parser.add_subparsers(dest="cmd")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("textfile", help="Path to text file to train on")
    train_parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    train_parser.add_argument("--optim", type=str, default="adamw", help="Optimizer to use (adamw or sgd)")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--seqlen", type=int, default=100, help="Sequence length")
    train_parser.add_argument("--emb-size", type=int, default=128, help="Embedding size")
    train_parser.add_argument("--hidden-sizes", type=str, default="128,256,512", help="Hidden sizes (comma-separated)")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Sample parser
    sample_parser = subparsers.add_parser("sample", help="Sample from a model")
    sample_parser.add_argument("model", help="Path to model checkpoint")
    sample_parser.add_argument("--prefix", type=str, default="\n", help="Prefix text")
    sample_parser.add_argument("--num-chars", type=int, default=500, help="Number of characters to generate")
    sample_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    sample_parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.cmd == "train":
        # Parse hidden sizes
        hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]
        
        # Create configuration
        cfg = {
            "optim": args.optim,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seqlen": args.seqlen,
            "emb_size": args.emb_size,
            "hidden_sizes": hidden_sizes,
            "dropout": args.dropout,
            "norm": True,
            "wandb": args.wandb,
        }
        
        # Load dataset
        _logger.info(f"Loading dataset from {args.textfile}")
        train_ds, val_ds = CharTokenDataset.from_textfile(args.textfile, args.seqlen)
        
        # Update configuration with vocabulary size
        cfg["vocab_size"] = train_ds.vocab_size
        
        _logger.info(f"Vocabulary size: {cfg['vocab_size']}")
        _logger.info(f"Training with configuration: {cfg}")
        
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CharNLPModel(cfg).to(device)
        
        # Train model
        train(model, train_ds, val_ds, cfg, device)
        
    elif args.cmd == "sample":
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, metadata = load_model(CharNLPModel, args.model, device)
        
        # Extract character mappings
        char_to_idx = metadata.get("char_to_idx", {})
        idx_to_char = metadata.get("idx_to_char", {})
        
        if not char_to_idx or not idx_to_char:
            _logger.error("Model checkpoint does not contain character mappings")
            return
        
        # Convert string indices to integers in idx_to_char
        idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        
        # Generate text
        generated_text = generate_text(
            model,
            args.prefix,
            idx_to_char,
            char_to_idx,
            num_chars=args.num_chars,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        
        print(f"Generated text:\n{generated_text}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
