"""Cross-validation of generative model performance

Christoph Heindl, 2024
https://github.com/cheind/mingru
"""

import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import tiktoken
import torch
from pathlib import Path
import wandb
from examples.nlp import generate_text_mbili

_logger = logging.getLogger("crossval")
handler = RotatingFileHandler("tmp/minrnn.boros.xover.log", maxBytes=512000, backupCount=100)
_logger.addHandler(handler)


def cross_validate_generation(model_path: str, test_file: str, sample_size: int, use_wandb: bool = False):
    """Evaluate model's generative ability using cross-validation.
    
    Args:
        model_path: Path to trained model checkpoint
        test_file: Path to text file for testing
        sample_size: Number of tokens to use as input for generation
        
    Returns:
        List of perplexity values from each sample
    """
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="MinRNN Cross-Validation",
            name=f"Cross-val {Path(model_path).stem} Sample size {sample_size}",
            config={
                "model_path": model_path,
                "test_file": test_file,
                "sample_size": sample_size
            }
        )

    # Load model
    model = torch.jit.load(model_path)
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    
    # Load and tokenize test data
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    
    # Calculate number of chunks based on text length and sample size
    num_chunks = len(tokens) // sample_size
    perplexities = []
    
    for i in range(num_chunks):
        # Select a different chunk each time
        start = i * sample_size
        end = start + sample_size
        if end > len(tokens):
            continue
            
        # Get sample text
        sample_tokens = tokens[start:end]
        prefix = enc.decode(sample_tokens)
        
        # Generate text and get perplexity
        _, perplexity = generate_text_mbili(
            model, 
            dev,
            prefix=prefix,
            num_tokens=sample_size,
            top_k=200
        )
        
        perplexities.append(perplexity.item())
        _logger.info(f"Sample {i+1}/{num_chunks} - Perplexity: {perplexity:.2f}")
        
        if use_wandb:
            wandb.log({
                "sample": i+1,
                "perplexity": perplexity.item(),
                "mean_perplexity": np.mean(perplexities),
                "std_perplexity": np.std(perplexities)
            })
    
    mean_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)
    
    if use_wandb:
        wandb.log({
            "final_mean_perplexity": mean_perplexity,
            "final_std_perplexity": std_perplexity
        })
        wandb.finish()
    
    return perplexities

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model checkpoint file")
    parser.add_argument("testfile", help="Path to text file for testing")
    parser.add_argument("--sample-size", type=int, default=32, 
                       help="Number of tokens to use as input for generation")
    parser.add_argument("--wandb", type=bool, default=False,
                       help="Enable wandb logging")
    args = parser.parse_args()
    
    perplexities = cross_validate_generation(
        args.model,
        args.testfile,
        args.sample_size,
        args.wandb
    )
    
    mean_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)
    
    _logger.info(f"Cross-validation results:")
    _logger.info(f"Mean perplexity: {mean_perplexity:.2f}")
    _logger.info(f"Std. deviation: {std_perplexity:.2f}")
