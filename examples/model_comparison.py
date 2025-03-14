"""Model comparison for MinRNN architectures

Christoph Heindl, 2024
https://github.com/cheind/mingru
"""

import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import tiktoken
import torch
from pathlib import Path
import sys

# Try to import matplotlib, provide helpful error if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Visualization features will be disabled.")
    print("To install matplotlib, run: pip install matplotlib")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging features will be disabled.")
    print("To install wandb, run: pip install wandb")
from examples.nlp import generate_text_mbili, NLPModel
from examples.cross_validation import cross_validate_generation
import os

_logger = logging.getLogger("model_comparison")
handler = RotatingFileHandler("tmp/minrnn.boros.comparison.log", maxBytes=512000, backupCount=100)
_logger.addHandler(handler)

def load_model_from_checkpoint(model_path):
    """Load a model from a checkpoint file without using TorchScript.
    
    This is a fallback method when TorchScript models fail to load.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        # Extract model configuration from filename
        model_name = Path(model_path).stem
        
        # Determine architecture
        arch = "minLSTM" if "minLSTM" in model_name else "minGRU"
        
        # Extract hidden sizes
        hidden_sizes = [256, 512, 1024]  # Default
        if "_hidden" in model_name:
            try:
                hidden_part = model_name.split("_hidden")[1]
                sizes_str = hidden_part.split(".")[0]
                hidden_sizes = [int(s) for s in sizes_str.split("_")]
            except:
                pass
        
        # Create model config
        cfg = {
            "arch": arch,
            "vocab_size": 50257,  # GPT-2 vocabulary size
            "emb_size": 512,
            "hidden_sizes": hidden_sizes,
            "dropout": 0.1,
            "norm": True
        }
        
        # Create model
        model = NLPModel(cfg)
        
        # Try to load state dict
        try:
            # First try loading as TorchScript
            scripted_model = torch.jit.load(model_path)
            # Extract state dict from scripted model
            state_dict = {}
            for name, param in scripted_model.named_parameters():
                state_dict[name] = param.detach()
            model.load_state_dict(state_dict)
            return model
        except:
            # If that fails, try loading as regular checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            return model
    except Exception as e:
        _logger.error(f"Failed to load model {model_path}: {str(e)}")
        return None

def evaluate_model_directly(model_path, test_file, sample_size=256):
    """Evaluate a model directly without using TorchScript.
    
    Args:
        model_path: Path to model checkpoint
        test_file: Path to test file
        sample_size: Number of tokens to generate
        
    Returns:
        Tuple of (mean_perplexity, std_perplexity) or (inf, 0) if evaluation fails
    """
    model = load_model_from_checkpoint(model_path)
    if model is None:
        return float('inf'), 0
    
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    
    # Load test data
    test_file_path = Path(test_file).expanduser()
    with open(test_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Take a few samples from the text
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    
    # Take up to 5 samples
    num_samples = min(5, len(tokens) // sample_size)
    perplexities = []
    
    for i in range(num_samples):
        start = i * sample_size
        sample_tokens = tokens[start:start+sample_size]
        prefix = enc.decode(sample_tokens[:min(32, len(sample_tokens))])  # Use shorter prefix
        
        try:
            _, perplexity = generate_text_mbili(
                model, dev, prefix, min(64, sample_size), top_k=200
            )
            perplexities.append(perplexity.item())
        except Exception as e:
            _logger.error(f"Error in direct evaluation: {str(e)}")
            continue
    
    if not perplexities:
        return float('inf'), 0
    
    return np.mean(perplexities), np.std(perplexities)

def compare_models(model_paths, test_file, sample_size, use_wandb=False):
    """Compare multiple models on the same test data.
    
    Args:
        model_paths: List of paths to trained model checkpoints
        test_file: Path to text file for testing
        sample_size: Number of tokens to use as input for generation
        use_wandb: Whether to log results to wandb
        
    Returns:
        DataFrame with model comparison results
    """
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="MinRNN Model Comparison",
            name=f"Model Comparison - Sample size {sample_size}",
            config={
                "model_paths": model_paths,
                "test_file": test_file,
                "sample_size": sample_size
            }
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb logging requested but wandb is not installed")
    
    results = []
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        _logger.info(f"Evaluating model: {model_name}")
        
        # First try cross-validation
        try:
            perplexities = cross_validate_generation(
                model_path,
                test_file,
                sample_size,
                False  # Don't use wandb for individual models
            )
            
            # Check if we got valid results
            if len(perplexities) == 0 or np.isinf(perplexities).any():
                _logger.warning(f"Cross-validation failed for {model_name}, trying direct evaluation")
                mean_perplexity, std_perplexity = evaluate_model_directly(model_path, test_file, sample_size)
                if np.isinf(mean_perplexity):
                    _logger.error(f"Failed to evaluate model {model_name} - skipping")
                    continue
                min_perplexity = max_perplexity = mean_perplexity  # We don't have min/max in direct evaluation
            else:
                # Calculate statistics from cross-validation
                mean_perplexity = np.mean(perplexities)
                std_perplexity = np.std(perplexities)
                min_perplexity = np.min(perplexities)
                max_perplexity = np.max(perplexities)
        except Exception as e:
            _logger.warning(f"Error in cross-validation for {model_name}: {str(e)}, trying direct evaluation")
            mean_perplexity, std_perplexity = evaluate_model_directly(model_path, test_file, sample_size)
            if np.isinf(mean_perplexity):
                _logger.error(f"Failed to evaluate model {model_name} - skipping")
                continue
            min_perplexity = max_perplexity = mean_perplexity  # We don't have min/max in direct evaluation
        
        # Extract model architecture from filename
        arch = "MinLSTM" if "minLSTM" in model_name else "MinGRU"
        
        # Extract hidden sizes from filename if available
        hidden_sizes = "unknown"
        if "_hidden" in model_name:
            try:
                hidden_part = model_name.split("_hidden")[1]
                hidden_sizes = hidden_part.split(".")[0]
            except:
                pass
        
        results.append({
            "model_name": model_name,
            "architecture": arch,
            "hidden_sizes": hidden_sizes,
            "mean_perplexity": mean_perplexity,
            "std_perplexity": std_perplexity,
            "min_perplexity": min_perplexity,
            "max_perplexity": max_perplexity
        })
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "model": model_name,
                "architecture": arch,
                "mean_perplexity": mean_perplexity,
                "std_perplexity": std_perplexity
            })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Sort by mean perplexity (lower is better)
    df = df.sort_values("mean_perplexity")
    
    # Print results
    _logger.info("\nModel Comparison Results:")
    _logger.info(df.to_string())
    
    # Create visualization if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plot_comparison(df)
    
    if use_wandb and WANDB_AVAILABLE:
        # Log the table
        wandb.log({"comparison_table": wandb.Table(dataframe=df)})
        
        # Log the plot if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            fig = plot_comparison(df, show=False)
            wandb.log({"comparison_plot": wandb.Image(fig)})
            plt.close(fig)
        
        wandb.finish()
    
    return df

def plot_comparison(df, show=True):
    """Create a bar chart comparing model perplexities.
    
    Args:
        df: DataFrame with model comparison results
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plot: matplotlib is not installed")
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.bar(
        df["model_name"], 
        df["mean_perplexity"],
        yerr=df["std_perplexity"],
        capsize=5,
        alpha=0.7
    )
    
    # Color bars by architecture
    for i, arch in enumerate(df["architecture"]):
        bars[i].set_color("blue" if arch == "MinLSTM" else "orange")
    
    # Add labels and title
    ax.set_xlabel("Model")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Model Comparison - Mean Perplexity with Standard Deviation")
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="blue", label="MinLSTM"),
        Patch(facecolor="orange", label="MinGRU")
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                       help="Paths to model checkpoint files")
    parser.add_argument("--testfile", default="~/Datasets/tiny-shakespeare/train_coriolanus.csv.10percent",
                       help="Path to text file for testing")
    parser.add_argument("--sample-size", type=int, default=256, 
                       help="Number of tokens to use as input for generation")
    parser.add_argument("--wandb", type=bool, default=False,
                       help="Enable wandb logging")
    parser.add_argument("--direct", action="store_true",
                       help="Use direct evaluation instead of cross-validation")
    args = parser.parse_args()
    
    # Expand the user home directory in the testfile path
    testfile = Path(args.testfile).expanduser()
    
    if args.direct:
        _logger.info("Using direct evaluation mode")
        results = []
        
        for model_path in args.models:
            model_name = Path(model_path).stem
            _logger.info(f"Directly evaluating model: {model_name}")
            
            mean_perplexity, std_perplexity = evaluate_model_directly(model_path, testfile, args.sample_size)
            
            if not np.isinf(mean_perplexity):
                # Extract model architecture from filename
                arch = "MinLSTM" if "minLSTM" in model_name else "MinGRU"
                
                # Extract hidden sizes from filename if available
                hidden_sizes = "unknown"
                if "_hidden" in model_name:
                    try:
                        hidden_part = model_name.split("_hidden")[1]
                        hidden_sizes = hidden_part.split(".")[0]
                    except:
                        pass
                
                results.append({
                    "model_name": model_name,
                    "architecture": arch,
                    "hidden_sizes": hidden_sizes,
                    "mean_perplexity": mean_perplexity,
                    "std_perplexity": std_perplexity,
                    "min_perplexity": mean_perplexity,  # Same as mean in direct mode
                    "max_perplexity": mean_perplexity   # Same as mean in direct mode
                })
        
        if results:
            # Create DataFrame for analysis
            df = pd.DataFrame(results)
            
            # Sort by mean perplexity (lower is better)
            df = df.sort_values("mean_perplexity")
            
            # Print results
            _logger.info("\nModel Comparison Results:")
            _logger.info(df.to_string())
            
            # Create visualization if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plot_comparison(df)
        else:
            _logger.error("No models could be evaluated successfully")
    else:
        compare_models(
            args.models,
            testfile,
            args.sample_size,
            args.wandb
        )
