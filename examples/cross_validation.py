"""Cross-validation of generative model performance

Christoph Heindl, 2024
https://github.com/cheind/mingru
"""

import logging
import numpy as np
import tiktoken
import torch
from pathlib import Path
from examples.nlp import generate_text_mbili

_logger = logging.getLogger("crossval")

def cross_validate_generation(model_path: str, test_file: str, sample_size: int, num_samples: int = 5):
    """Evaluate model's generative ability using cross-validation.
    
    Args:
        model_path: Path to trained model checkpoint
        test_file: Path to text file for testing
        sample_size: Number of tokens to use as input for generation
        num_samples: Number of cross-validation samples to take
        
    Returns:
        List of perplexity values from each sample
    """
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
    
    # Calculate chunk size
    chunk_size = len(tokens) // num_samples
    perplexities = []
    
    for i in range(num_samples):
        # Select a different chunk each time
        start = i * chunk_size
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
        _logger.info(f"Sample {i+1}/{num_samples} - Perplexity: {perplexity:.2f}")
    
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
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of cross-validation samples to take")
    
    args = parser.parse_args()
    
    perplexities = cross_validate_generation(
        args.model,
        args.testfile,
        args.sample_size,
        args.num_samples
    )
    
    mean_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)
    
    _logger.info(f"Cross-validation results:")
    _logger.info(f"Mean perplexity: {mean_perplexity:.2f}")
    _logger.info(f"Std. deviation: {std_perplexity:.2f}")
