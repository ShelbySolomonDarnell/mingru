"""Training and evaluation functions for character-level MinGRU

This module provides functions for training, validating, and generating text with
character-level MinGRU models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from itertools import islice
from typing import Optional
import wandb

from .model import CharNLPModel
from .dataset import CharTokenDataset

_logger = logging.getLogger("char_mingru")

def train(
    model: CharNLPModel,
    train_ds: CharTokenDataset,
    val_ds: CharTokenDataset,
    cfg: dict,
    device: torch.device,
):
    """Train a character-level MinGRU model.
    
    Args:
        model: The model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        cfg: Configuration dictionary
        device: Device to train on
    """
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    # Create optimizer and scheduler
    if cfg["optim"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=5e-4,
            eps=1e-8,
        )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        cfg["num_epochs"] - 2,
        gamma=0.1,
    )
    
    # Create loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize wandb if enabled
    if cfg.get("wandb", False):
        wandb.init(
            project="CharMinGRU Training",
            name=f"CharMinGRU epochs {cfg['num_epochs']}, optimizer {cfg['optim']}",
            config=cfg,
        )
    
    # Initialize training variables
    best_acc = 0
    detached_hidden_state = []
    
    # Training loop
    for epoch in range(cfg["num_epochs"]):
        model.train()
        
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Reset hidden state at the beginning of each epoch
            if step % (len(train_loader) - 1) == 0:
                detached_hidden_state = None
            
            # Forward pass
            logits, hidden_state = model(x, detached_hidden_state)
            
            # Ensure hidden_state is a list
            if not isinstance(hidden_state, list) and hidden_state is not None:
                hidden_state = [hidden_state] if torch.is_tensor(hidden_state) else list(hidden_state)
            
            # Detach hidden state for next iteration
            from .utils import detach_tensors_in_list
            detached_hidden_state = detach_tensors_in_list(hidden_state)
            
            # Calculate loss
            loss = criterion(logits.permute(0, 2, 1), y)
            
            # Check for NaN values
            if torch.isnan(loss).any():
                _logger.warning(f"NaN detected in loss at step {step+1}. Skipping backward pass.")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Calculate perplexity
            perplexity = torch.exp(torch.clamp(loss, 0, 20))
            
            # Log progress
            if (step + 1) % 20 == 0:
                _logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")
                
                if cfg.get("wandb", False):
                    wandb.log({
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "loss": loss,
                        "perplexity": perplexity,
                    })
            
            # Validate periodically
            if (step + 1) % 200 == 0:
                val_acc, val_loss = validate(model, val_ds, cfg["batch_size"], device)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, "
                    f"Validation Accuracy: {val_acc*100:.2f}%, "
                    f"Validation Loss: {val_loss:.2f}"
                )
                
                # Save best model
                if val_acc > best_acc:
                    _logger.info(f"New best model at epoch {epoch+1} step {step+1}")
                    model_name = f"char_mingru_best.epochs{cfg['num_epochs']}_hidden{'_'.join(map(str, cfg['hidden_sizes']))}.pt"
                    model_path = Path("tmp") / model_name
                    
                    # Save model
                    from .utils import save_model
                    save_model(
                        model,
                        str(model_path),
                        optimizer=optimizer,
                        epoch=epoch,
                        metadata={
                            "validation_accuracy": val_acc,
                            "validation_loss": val_loss,
                            "step": step,
                            "optimizer": cfg["optim"],
                            "char_to_idx": train_ds.char_to_idx,
                            "idx_to_char": train_ds.idx_to_char,
                        },
                    )
                    best_acc = val_acc
                
                # Generate sample text
                sample_text = generate_text(
                    model,
                    "\n",
                    train_ds.idx_to_char,
                    train_ds.char_to_idx,
                    num_chars=100,
                    device=device,
                )
                _logger.info(f"Sample text:\n{sample_text}")
                
                if cfg.get("wandb", False):
                    wandb.log({
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "validation_accuracy": val_acc * 100,
                        "validation_loss": val_loss,
                        "sample_text": sample_text,
                    })
                
                model.train()
        
        # Update learning rate
        scheduler.step()
    
    # Finish wandb run
    if cfg.get("wandb", False):
        wandb.finish()


@torch.no_grad()
def validate(
    model: CharNLPModel,
    val_ds: CharTokenDataset,
    batch_size: int,
    device: torch.device,
):
    """Validate a character-level MinGRU model.
    
    Args:
        model: The model to validate
        val_ds: Validation dataset
        batch_size: Batch size
        device: Device to validate on
        
    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    
    # Create validation loader
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Initialize metrics
    total = 0
    total_loss = 0
    total_correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # Validation loop
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits, _ = model(x)
        
        # Calculate loss
        loss = criterion(logits.permute(0, 2, 1), y)
        
        # Calculate accuracy
        total_correct += (logits.argmax(2) == y).sum().item()
        total += x.shape[0] * x.shape[1]
        total_loss += loss.item()
    
    # Calculate metrics
    accuracy = total_correct / total
    avg_loss = total_loss / len(val_loader)
    
    return accuracy, avg_loss


@torch.no_grad()
def generate_text(
    model: CharNLPModel,
    prefix: str,
    idx_to_char: dict,
    char_to_idx: dict,
    num_chars: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """Generate text with a character-level MinGRU model.
    
    Args:
        model: The model to generate text with
        prefix: Prefix text to start generation
        idx_to_char: Mapping from character indices to characters
        char_to_idx: Mapping from characters to indices
        num_chars: Number of characters to generate
        temperature: Temperature for sampling
        top_k: Number of top candidates to sample from
        device: Device to generate on
        
    Returns:
        Generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Convert prefix to tensor
    prefix_ids = [char_to_idx.get(ch, 0) for ch in prefix]
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Initialize generation
    generated = list(prefix)
    h = None
    
    # Generate characters
    for _ in range(num_chars):
        # Forward pass
        logits, h = model(prefix_tensor, h)
        
        # Get probabilities for next character
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k sampling
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("inf")
        
        # Sample next character
        probs = F.softmax(logits, dim=-1)
        next_char_idx = torch.multinomial(probs, num_samples=1).item()
        
        # Add to generated text
        next_char = idx_to_char[next_char_idx]
        generated.append(next_char)
        
        # Update prefix tensor for next iteration
        prefix_tensor = torch.tensor([[next_char_idx]], dtype=torch.long, device=device)
    
    return "".join(generated)
