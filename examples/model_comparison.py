"""Model comparison for MinRNN architectures

Christoph Heindl, 2024
https://github.com/cheind/mingru
"""

import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import torch
from pathlib import Path
import wandb
from examples.nlp import generate_text_mbili
from examples.cross_validation import cross_validate_generation

_logger = logging.getLogger("model_comparison")
handler = RotatingFileHandler("tmp/minrnn.boros.comparison.log", maxBytes=512000, backupCount=100)
_logger.addHandler(handler)

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
    if use_wandb:
        wandb.init(
            project="MinRNN Model Comparison",
            name=f"Model Comparison - Sample size {sample_size}",
            config={
                "model_paths": model_paths,
                "test_file": test_file,
                "sample_size": sample_size
            }
        )
    
    results = []
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        _logger.info(f"Evaluating model: {model_name}")
        
        # Run cross-validation for this model
        perplexities = cross_validate_generation(
            model_path,
            test_file,
            sample_size,
            False  # Don't use wandb for individual models
        )
        
        # Calculate statistics
        mean_perplexity = np.mean(perplexities)
        std_perplexity = np.std(perplexities)
        min_perplexity = np.min(perplexities)
        max_perplexity = np.max(perplexities)
        
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
        
        if use_wandb:
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
    
    # Create visualization
    plot_comparison(df)
    
    if use_wandb:
        # Log the table
        wandb.log({"comparison_table": wandb.Table(dataframe=df)})
        
        # Log the plot
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
        The matplotlib figure
    """
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
    parser.add_argument("--testfile", required=True,
                       help="Path to text file for testing")
    parser.add_argument("--sample-size", type=int, default=32, 
                       help="Number of tokens to use as input for generation")
    parser.add_argument("--wandb", type=bool, default=False,
                       help="Enable wandb logging")
    args = parser.parse_args()
    
    compare_models(
        args.models,
        args.testfile,
        args.sample_size,
        args.wandb
    )
