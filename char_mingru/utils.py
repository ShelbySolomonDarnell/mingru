"""Utility functions for character-level MinGRU

This module provides utility functions for the character-level MinGRU implementation.
"""

import torch
import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

_logger = logging.getLogger("char_mingru")

def detach_tensors_in_list(the_tensor_container):
    """Detach tensors from computation graph.
    
    This function handles tensors in lists or tuples, making it compatible with both
    MinGRU (which uses lists) and MinLSTM (which uses tuples of lists).
    
    Args:
        the_tensor_container: A list or tuple of tensors, or a single tensor
        
    Returns:
        Container of the same type with detached tensors
    """
    # Handle None case
    if the_tensor_container is None:
        return None
        
    # Handle single tensor case
    if torch.is_tensor(the_tensor_container):
        return the_tensor_container.detach().clone()
        
    # Handle tuple case
    if isinstance(the_tensor_container, tuple):
        # Preserve tuple structure exactly
        return tuple(detach_tensors_in_list(item) for item in the_tensor_container)
    
    # Handle list case
    if isinstance(the_tensor_container, list):
        # Preserve list structure exactly
        result = []
        for the_state in the_tensor_container:
            if torch.is_tensor(the_state):
                result.append(the_state.detach().clone())
            else:
                # Handle nested containers (like lists within lists)
                result.append(detach_tensors_in_list(the_state))
        return result
        
    # If we get here, we have an unsupported type
    raise TypeError(f"Unsupported container type: {type(the_tensor_container)}")


def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save model weights and optional training state to a file.
    
    Args:
        model: Model to save
        path: Path where to save the model
        optimizer: Optional optimizer to save state
        epoch: Optional current epoch number
        metadata: Optional dictionary with additional metadata
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        save_dict['epoch'] = epoch
        
    if metadata is not None:
        save_dict.update(metadata)
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    torch.save(save_dict, path)
    _logger.info(f"Model saved to {path}")


def load_model(
    model_class,
    path: str,
    device: Optional[torch.device] = None,
):
    """Load a model from a saved checkpoint.
    
    Args:
        model_class: Model class to instantiate
        path: Path to the saved model
        device: Device to load the model to
        
    Returns:
        Tuple of (loaded_model, metadata)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(path, map_location=device)
    
    # Extract metadata
    metadata = {k: v for k, v in checkpoint.items() 
               if k not in ['model_state_dict', 'optimizer_state_dict']}
    
    # Create model configuration
    cfg = {
        'vocab_size': len(metadata.get('char_to_idx', {})),
        'emb_size': 128,  # Default value
        'hidden_sizes': [128, 256, 512],  # Default value
        'dropout': 0.0,  # Default value
        'norm': True,  # Default value
    }
    
    # Create model
    model = model_class(cfg).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, metadata
