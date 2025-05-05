"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import logging
from logging.handlers import RotatingFileHandler
from typing import List
import warnings
from itertools import islice
from pathlib import Path
import sh
import pathlib
import deepspeed
import tensorboard
from deepspeed.accelerator import get_accelerator

import os
import datetime
import sys
import json
# Try to import wandb, but don't fail if it's not available
import wandb
import numpy as np
import tiktoken
import torch
import schedulefree
import torch.nn.functional as F
import torch.utils.data.dataloader
import torch.distributed as dist
import torch.multiprocessing as mp
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
from torch.nn import Linear

sys.path.append("..")
from examples.utils import *
from examples.utils import cfg as _cfg
from mingru  import *
from minlstm import *

warnings.filterwarnings("ignore")

_logger = logging.getLogger("nlp")
handler = RotatingFileHandler("/home/shelbys/code/werernns/mingru/tmp/minrnn.boros.log", maxBytes=512000, backupCount=100)
_logger.addHandler(handler)

def log_dist(message: str,
            ranks: List[int] = [],
            level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')

def get_architecture_name():
    """Get the architecture name from the config."""
    return _cfg.get("MAIN", "arch", fallback="minGRU")


class TokenIdDataset(torch.utils.data.Dataset):

    def __init__(self, tokenids: np.ndarray, seqlen: int):
        super().__init__()
        self.tokenids = tokenids
        self.seqlen = seqlen

    def __len__(self):
        return len(self.tokenids) - self.seqlen - 1

    def __getitem__(self, index):
        x = self.tokenids[index : index + self.seqlen].astype(np.int64)
        y = self.tokenids[index + 1 : index + 1 + self.seqlen].astype(np.int64)
        return x, y

    @staticmethod
    def from_textfile(path: str, seqlen: int):
        """Tokenizes (GPT-2) content of textfile and returns train/val datasets."""
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]

        enc = tiktoken.get_encoding("gpt2")
        train_ids = np.array(enc.encode_ordinary(train_data))
        val_ids = np.array(enc.encode_ordinary(val_data))

        train_ds = TokenIdDataset(train_ids, seqlen)
        val_ds = TokenIdDataset(val_ids, seqlen)

        return train_ds, val_ds
    

class NLPModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Store whether this is an LSTM model for TorchScript compatibility
        self._is_lstm_model = cfg["arch"] == 'minLSTM'
        
        self.emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_size"])
        self.rnn = MinLSTM( 
            input_size=cfg["emb_size"],
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            residual=True,
            bias=True,
            norm=cfg["norm"],
        ) if self._is_lstm_model else MinGRU(
            input_size=cfg["emb_size"],
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            residual=True,
            bias=False,
            norm=cfg["norm"],
        )

        model_bias = True if self._is_lstm_model else False
        self.ln = torch.nn.LayerNorm(cfg["hidden_sizes"][-1], model_bias)
        self.fc = torch.nn.Linear(cfg["hidden_sizes"][-1], cfg["vocab_size"])

    def save_model(self, path: str, optimizer=None, epoch=None, metadata=None):
        """Save model weights and optional training state to a file.
        
        This method saves the model in a format that can be easily loaded
        without relying on TorchScript, which can have compatibility issues.
        
        Args:
            path: Path where to save the model
            optimizer: Optional optimizer to save state
            epoch: Optional current epoch number
            metadata: Optional dictionary with additional metadata
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'arch': 'minLSTM' if self._is_lstm_model else 'minGRU',
            'hidden_sizes': self.rnn.layer_sizes[1:],
            'vocab_size': self.emb.num_embeddings,
            'emb_size': self.emb.embedding_dim
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
    
    @staticmethod
    def load_model(path: str, device=None):
        """Load a model from a saved checkpoint.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded NLPModel instance and optional metadata
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(path, map_location=device)
        
        # Create config from saved parameters
        hidden_sizes = checkpoint.get('hidden_sizes', [256, 512, 1024])
        # Ensure hidden_sizes is a list
        if isinstance(hidden_sizes, tuple):
            hidden_sizes = list(hidden_sizes)
            
        cfg = {
            'arch': checkpoint.get('arch', 'minGRU'),
            'hidden_sizes': hidden_sizes,
            'vocab_size': checkpoint.get('vocab_size', 50257),
            'emb_size': checkpoint.get('emb_size', 768),
            'dropout': 0.0,  # Default value
            'norm': True     # Default value
        }
        
        # Create model
        model = NLPModel(cfg).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return model and any additional metadata
        metadata = {k: v for k, v in checkpoint.items() 
                   if k not in ['model_state_dict', 'optimizer_state_dict']}
        
        return model, metadata
        
    def forward(self, ids: torch.IntTensor, h: list[torch.Tensor] | None = None, c: list[torch.Tensor] | None = None):
        """Forward pass through the NLP model.
        
        This method processes token IDs through the model's components:
        1. Embeds token IDs into continuous vectors
        2. Processes embeddings through the RNN (either MinLSTM or MinGRU)
        3. Applies layer normalization to the RNN output
        4. Projects to vocabulary logits via the fully connected layer
        
        The method handles both MinLSTM and MinGRU architectures:
        - MinLSTM: Requires both hidden state (h) and cell state (c)
        - MinGRU: Requires only hidden state (h)
        
        Args:
            ids: Integer tensor of token IDs with shape [batch_size, seq_len]
            h: Optional list of hidden state tensors from previous steps
            c: Optional list of cell state tensors (only used with MinLSTM)
            
        Returns:
            tuple: (logits, hidden_state) where:
                - logits: Tensor with shape [batch_size, seq_len, vocab_size]
                - hidden_state: Either a tuple (h, c) for MinLSTM or just h for MinGRU
        """
        x = self.emb(ids)
        
        # Use a string attribute check instead of isinstance for TorchScript compatibility
        is_lstm = hasattr(self, '_is_lstm_model') and self._is_lstm_model
        
        # Handle different RNN architectures
        if is_lstm:
            # For TorchScript compatibility, we need to handle the case differently
            if h is not None and c is not None:
                try:
                    # Try using forward_with_separate_states if available
                    x, h_out, c_out = self.rnn.forward_with_separate_states(x, h, c)
                    hidden_state = (h_out, c_out)
                except AttributeError:
                    # Fall back to regular forward if the method is not available
                    x, hidden_state = self.rnn(x, (h, c))
            else:
                # Let the RNN initialize the hidden states
                x, hidden_state = self.rnn(x)
                
            # Ensure hidden_state is always a tuple of (h, c)
            if not isinstance(hidden_state, tuple):
                # For TorchScript compatibility, avoid logging
                # Split the list in half for h and c if needed
                if isinstance(hidden_state, list) and len(hidden_state) >= 2:
                    mid = len(hidden_state) // 2
                    hidden_state = (hidden_state[:mid], hidden_state[mid:])
                else:
                    # Create a default format
                    hidden_state = (hidden_state, hidden_state) if torch.is_tensor(hidden_state) else ([], [])
        else:  # MinGRU
            # Use forward_with_separate_states for consistency with MinLSTM interface
            if h is not None:
                x, hidden_state, _ = self.rnn.forward_with_separate_states(x, h)
            else:
                x, hidden_state = self.rnn(x)
            
            # Ensure hidden_state is always a list for MinGRU
            if not isinstance(hidden_state, list) and hidden_state is not None:
                hidden_state = [hidden_state] if torch.is_tensor(hidden_state) else list(hidden_state)
            
        x = self.ln(x)
        logits = self.fc(x)
        
        return logits, hidden_state

def init_optimizer(params, the_cfg):
    result = None
    if the_cfg["optim"] == "sgd":
        result = torch.optim.SGD(
            params,
            lr=the_cfg["lr"],
            momentum=0.9,
            weight_decay=5e-4
        )
    else:
        result = schedulefree.AdamWScheduleFree(params, lr=the_cfg["lr"]) 
        """
        result = torch.optim.AdamW(  # Use AdamW instead of Adam for better stability
            params,
            lr=the_cfg["lr"],
            weight_decay=5e-4,
            eps=1e-8  # Increase epsilon to prevent division by zero
        )
        """
    return result

def init_wandb(cfg, rank):
    """Initialize wandb logging.
    
    This should only be called from rank 0 in a distributed setting.
    
    Args:
        cfg: Training configuration
        rank: Process rank
    """
    if cfg["wandb"]:
        try:
            # Only log from rank 0 in distributed training
            if rank == 0:
                wandb.init(
                    # Set the project where this run will be logged
                    project=cfg['arch'] + " Shakespeare training",
                    name=f"{cfg['arch']} epochs {cfg['num_epochs']}, optimizer {cfg['optim']}, hidden_sizes {cfg['hidden_sizes']}",
                    # Track hyper parameters and run metadata
                    config={
                        "architecture":    cfg["arch"],
                        "learning_rate":   cfg["lr"],
                        "batch_size":      cfg["batch_size"],
                        "dropout":         cfg["dropout"],
                        "dataset":         cfg["dataset"],
                        "epochs":          cfg["num_epochs"],
                        "sequence_length": cfg["seqlen"],
                        "vocabulary_size": cfg["vocab_size"],
                        "embedding_size":  cfg["emb_size"],
                        "normalize":       cfg["norm"],
                        "hidden_sizes":    cfg["hidden_sizes"],
                        "optimizer":       cfg["optim"],
                        "num_gpus":        torch.cuda.device_count()
                    }
                )
                _logger.info("Wandb logging initialized")
        except Exception as e:
            _logger.warning(f"Failed to initialize wandb: {str(e)}")

def close_wandb(is_logging):
    """Safely close wandb logging.
    
    Args:
        is_logging: Whether wandb logging is active
    """
    if is_logging:
        try:
            wandb.finish()
        except Exception as e:
            _logger.warning(f"Error closing wandb: {str(e)}")

def train(cfg):

    local_rank = cfg["local_rank"]
    dtype      = cfg["dtype"]

    """Main training function that sets up distributed training if needed."""
    try:
        dev = (torch.device(get_accelerator().device_name(), local_rank) if (local_rank > -1)
              and get_accelerator().is_available() else torch.device("cpu"))
        #dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Only log from rank 0 to avoid duplicate logs
        should_log = True if local_rank==0 else False
        
        # Load datasets
        ds_train, ds_val = TokenIdDataset.from_textfile(cfg["training_data"], cfg["seqlen"])
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=cfg["batch_size"],
            #sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        if should_log:
            print_configuration()
            _logger.info(f"Number of examples in dataset {len(dl_train.dataset)}")
            _logger.info(f"Number of batches in dataset {len(dl_train)}")

        # Create validation dataset - only on rank 0
        if should_log:
            ds_val = torch.utils.data.Subset(
                ds_val, np.random.choice(len(ds_val), 256, replace=False)
            )
            _logger.info(f"Number of examples in test dataset {len(ds_val)}")

        # Create model and move to device
        model = NLPModel(cfg).to(dev)
        #ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        log_dist("Model Creation Done", ranks=[0])
        ################################
        ###### DeepSpeed engine ########
        ################################
        log_dist("Creating DeepSpeed engine", ranks=[0])
        assert (dtype == 'fp16' or dtype == 'bf16')
        ds_config = {
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "dtype": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }
        }
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=ds_config)
        log_dist("DeepSpeed engine created", ranks=[0])

        # Use label smoothing to improve training stability
        crit = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        #opt = init_optimizer(ddp_model.parameters(), cfg)
        opt = init_optimizer(model.parameters(), cfg)
        if (cfg["optim"] == "adamw"):
            opt.train()

        sched = torch.optim.lr_scheduler.StepLR(
            opt,
            cfg["num_epochs"] - 2,
            gamma=0.1,
        )

        # Initialize online logging and analysis - only on rank 0
        #if should_log:
            #init_wandb(cfg, rank)
            
        is_lstm = cfg["arch"] == "minLSTM"
        # Initialize hidden state variables based on architecture
        if is_lstm:
            detached_h_state = []
            detached_c_state = []
        else:  # minGRU
            detached_hidden_state = None
            
        best_acc = 0  # Start with 0 since we're tracking accuracy (0-1)
        model.train() # set model to training mode
        for epoch in range(cfg["num_epochs"]):
            # Set epoch for sampler to ensure different shuffling each epoch
            #train_sampler.set_epoch(epoch)
            
            # Reset hidden states at the beginning of each epoch
            if is_lstm:
                detached_h_state = []
                detached_c_state = []
            else:  # minGRU
                detached_hidden_state = None

            # move training data to the device 
            for step, (x, y) in enumerate(dl_train):
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                    
                # Forward pass with appropriate hidden state handling
                if is_lstm:
                    if detached_h_state and detached_c_state:
                        # Use the separate states method for MinLSTM
                        #x_emb = ddp_model.module.emb(x)
                        x_emb = model.emb(x)
                        try:
                            # Try using forward_with_separate_states if available
                            #rnn_out, h_out, c_out = ddp_model.module.rnn.forward_with_separate_states(x_emb, detached_h_state, detached_c_state)
                            rnn_out, h_out, c_out = model.rnn.forward_with_separate_states(x_emb, detached_h_state, detached_c_state)
                            #y_hat = ddp_model.module.fc(ddp_model.module.ln(rnn_out))
                            y_hat = model.fc(model.ln(rnn_out))
                        except AttributeError:
                            # Fall back to regular forward if the method is not available
                            #rnn_out, (h_out, c_out) = ddp_model.module.rnn(x_emb, (detached_h_state, detached_c_state))
                            #y_hat = model.module.fc(model.module.ln(rnn_out))
                            rnn_out, (h_out, c_out) = model.rnn(x_emb, (detached_h_state, detached_c_state))
                            y_hat = model.fc(model.ln(rnn_out))
                        
                        # Detach states for next iteration
                        detached_h_state = detach_tensors_in_list(h_out)
                        detached_c_state = detach_tensors_in_list(c_out)
                    else:
                        # First iteration, let the model initialize states
                        #y_hat, hidden_state = ddp_model(x)
                        y_hat, hidden_state = model(x)
                        # Unpack and detach
                        h_state, c_state = hidden_state
                        detached_h_state = detach_tensors_in_list(h_state)
                        detached_c_state = detach_tensors_in_list(c_state)
                else:  # minGRU
                    if detached_hidden_state is not None:
                        # Use the separate states method for consistency
                        #x_emb = ddp_model.module.emb(x)
                        #rnn_out, h_out, _ = ddp_model.module.rnn.forward_with_separate_states(x_emb, detached_hidden_state)
                        x_emb = model.emb(x)
                        rnn_out, h_out, _ = model.rnn.forward_with_separate_states(x_emb, detached_hidden_state)
                        #y_hat = ddp_model.module.fc(ddp_model.module.ln(rnn_out))
                        y_hat = model.fc(model.ln(rnn_out))
                        detached_hidden_state = detach_tensors_in_list(h_out)
                    else:
                        #y_hat, hidden_state = ddp_model(x)
                        y_hat, hidden_state = model(x)
                        
                        # Ensure hidden_state is a list before detaching
                        if not isinstance(hidden_state, list) and hidden_state is not None:
                            hidden_state = [hidden_state] if torch.is_tensor(hidden_state) else list(hidden_state)
                        
                        detached_hidden_state = detach_tensors_in_list(hidden_state)

                # Calculate loss and backpropagate
                loss = crit(y_hat.permute(0, 2, 1), y)
                
                # Check for NaN values in loss
                if torch.isnan(loss).any():
                    if should_log:
                        _logger.warning(f"NaN detected in loss at step {step+1}. Skipping backward pass.")
                    continue
                    
                opt.zero_grad()
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                #torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                opt.step()
                
                # Calculate perplexity with safety check
                perplexed = torch.exp(torch.clamp(loss, 0, 20))  # Clamp to prevent overflow

                # Logging - only from rank 0
                if should_log and (step + 1) % 20 == 0:
                    _logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, training perplexity: {perplexed:.4f}")
                    if cfg["wandb"]:
                        wandb.log({"step": step+1, "loss": loss, "training_perplexity": perplexed})
                
                # Validation - only from rank 0
                if should_log and (step + 1) % 200 == 0:
                    val_acc, val_loss = validate(model, dev, ds_val)  # Use non-DDP model for validation
                    _logger.info(
                        f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
                    )
                    if val_acc > best_acc:
                        _logger.info(f"New best model at epoch {epoch} step {step+1}")
                        model_name = f"nlp_best.epochs{cfg['num_epochs']}_{cfg['arch']}_hidden{'_'.join(map(str, cfg['hidden_sizes']))}.pt"
                        model_path = f"tmp/{model_name}"
                        
                        # Save model using our custom method
                        model.save_model(
                            model_path,
                            optimizer=opt,
                            epoch=epoch,
                            metadata={
                                'validation_accuracy': val_acc,
                                'validation_loss': val_loss,
                                'step': step,
                                'optimizer': cfg['optim']
                            }
                        )
                        best_acc = val_acc
                    
                    # Generate sample text
                    demo, sample_perplexity = generate_text(model, dev, prefix="\n", num_tokens=32, top_k=200)
                    if cfg["wandb"]:
                        wandb.log({
                            "Epoch": epoch+1,
                            "Step": step+1,
                            "Validation Accuracy": val_acc*100, 
                            "Validation Loss": val_loss, 
                            "Sample perplexity": sample_perplexity
                        })
                    _logger.info(f"Sample perplexity: {sample_perplexity}\nSample model output: {demo}")
                    
                    # Make sure to set model back to training mode
                    model.train()
                    #ddp_model.train()
                    if (cfg["optim"] == "adamw"):
                        opt.train()

            # Step the scheduler at the end of each epoch
            sched.step()
            
            # Make sure all processes are synchronized at the end of each epoch
            dist.barrier()
            
        # Clean up at the end of training
        if should_log:
            close_wandb(cfg["wandb"])
        
    except Exception as e:
        rank = 0 # remove when doing distributed
        # Log the error and clean up
        print(f"Error on rank {rank}: {str(e)}")
        import traceback
        traceback.print_exc()


@torch.no_grad()
def validate(
    model: NLPModel,
    dev: torch.device,
    ds: TokenIdDataset,
):
    """Validate model performance on a validation dataset.
    
    This function should be called with the non-DDP model to avoid
    synchronization issues during validation.
    
    Args:
        model: The model to validate (not wrapped in DDP)
        dev: Device to run validation on
        ds: Validation dataset
        
    Returns:
        Tuple of (accuracy, average_loss)
    """
    # Set model to evaluation mode
    model.eval()

    total = 0
    total_loss = 0
    total_correct = 0
    crit = torch.nn.CrossEntropyLoss()

    # Create dataloader for validation
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=min(32, len(ds)),  # Use smaller batch size for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    try:
        for ids, labels in dl:
            ids = ids.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            
            # Forward pass without tracking gradients
            logits, _ = model(ids)
            loss = crit(logits.permute(0, 2, 1), labels)

            # Calculate accuracy
            total_correct += (logits.argmax(2) == labels).sum().item()
            total += ids.shape[0] * ids.shape[1]
            total_loss += loss.item() * ids.shape[0]  # Weight by batch size
    except Exception as e:
        _logger.error(f"Error during validation: {str(e)}")
        return 0.0, float('inf')  # Return worst-case values on error

    # Calculate final metrics
    acc = total_correct / max(total, 1)  # Avoid division by zero
    avg_loss = total_loss / max(total, 1)

    return acc, avg_loss


@torch.no_grad()
def generate_text(
    model: NLPModel,
    dev: torch.device,
    prefix: str,
    num_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
) -> tuple[str, torch.Tensor]:
    enc = tiktoken.get_encoding("gpt2")
    ids = (
        torch.tensor(
            enc.encode_ordinary(prefix),
            dtype=int,
        )
        .to(dev)
        .unsqueeze(0)
    )
    
    all_probs = []
    generated_tokens = []

    gen = generate_tokens(model, prefix_ids=ids, temperature=temperature, top_k=top_k)
    generated_tokens, all_probs = zip(*list(islice(gen, num_tokens))) # splits a list of tuples
    new = torch.cat(generated_tokens, dim=1)
    g_tokens = new.squeeze(0).unsqueeze(1) # reformatted the tokens for calculations
    
    # Perplexity Calculation with error handling
    try:
        log_probs = torch.log(torch.stack(all_probs).gather(2, g_tokens.unsqueeze(2))).squeeze(2)
        perplexity = torch.exp(-torch.sum(log_probs) / num_tokens)
    except Exception as e:
        _logger.error(f"Error calculating perplexity: {str(e)}")
        perplexity = torch.tensor(float('inf'))

    return enc.decode(new[0].cpu().tolist()), perplexity

@torch.no_grad()
def generate_tokens(model, prefix_ids, temperature=1.0, top_k=None):
    assert prefix_ids.shape[1] > 0, "Need at least one start token"
    inp = prefix_ids
    
    # Initialize hidden states based on model architecture
    # Use a direct attribute check instead of isinstance for TorchScript compatibility
    is_lstm = hasattr(model, '_is_lstm_model') and model._is_lstm_model
    
    h = None
    c = None if not is_lstm else None
    all_probs = [] # Store all probabilities

    try:
        while True:
            if is_lstm:
                # For MinLSTM, use the separate states method
                if h is not None and c is not None:
                    try:
                        # Try using forward_with_separate_states if available
                        logits, h, c = model.rnn.forward_with_separate_states(model.emb(inp), h, c)
                        logits = model.fc(model.ln(logits))
                    except AttributeError:
                        # Fall back to regular forward if the method is not available
                        rnn_out, (h, c) = model.rnn(model.emb(inp), (h, c))
                        logits = model.fc(model.ln(rnn_out))
                else:
                    # First call, initialize states
                    logits, hidden_state = model.forward(inp)
                    h, c = hidden_state
            else:
                # For MinGRU, use the separate states method for consistency
                if h is not None:
                    logits, h, _ = model.rnn.forward_with_separate_states(model.emb(inp), h)
                    logits = model.fc(model.ln(logits))
                else:
                    # First call, initialize states
                    logits, h = model.forward(inp)
                
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            yield idx_next, probs # Yield both token and probabilities
            inp = idx_next
    except Exception as e:
        _logger.error(f"Error in generate_tokens: {str(e)}")
        # Yield a dummy token and probability to avoid breaking the caller
        yield torch.zeros_like(prefix_ids[:, :1]), torch.ones((1, 1, 50257)) / 50257

@torch.no_grad()
def sample(cfg):
    # Try to load with TorchScript first for backward compatibility
    try:
        model = torch.jit.load(cfg["ckpt"])
        _logger.info("Loaded model with TorchScript")
    except Exception as e:
        _logger.info(f"TorchScript loading failed: {str(e)}, trying regular loading")
        try:
            # Fall back to our custom loading method
            model, metadata = NLPModel.load_model(cfg["ckpt"])
            if metadata:
                _logger.info(f"Model metadata: {metadata}")
        except Exception as e2:
            _logger.error(f"Failed to load model using both methods: {str(e2)}")
            print(f"Error: Could not load model from {cfg['ckpt']}. Please check if the file exists and is in the correct format.")
            return None, None
    
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    
    output, perplex = generate_text(model, dev, cfg["precond"], cfg["num_tokens"], top_k=200)
    _logger.info(f"Sample perplexity is {perplex}")
    print("[Sampling perplexity] {0}\n{1}\n".format(perplex, output))
    
    # Return the results for potential further use
    return output, perplex


if __name__ == "__main__":
    import argparse

    # Set up logging
    os.makedirs("/home/shelbys/code/werernns/mingru/tmp", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler("/home/shelbys/code/werernns/mingru/tmp/nlp-boros.log.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )

    # Load configuration
    cfg = {
        "dataset":        _cfg.get("MAIN","datasetA", fallback="tiny-shakespeare"),
        "training_data":  _cfg.get("DATA", "train"),
        "testing_data":   _cfg.get("DATA", "test"),
        "arch":           _cfg.get("MAIN", "arch", fallback="minGRU"),
        "optim":          _cfg.get("MAIN","optim", fallback="adamw"),
        "lr":             _cfg.getfloat("MAIN","lr", fallback=0.001),
        "batch_size":     _cfg.getint("MAIN","batch_size", fallback=64),
        "num_epochs":     _cfg.getint("MAIN","num_epochs", fallback=5),
        "dropout":        _cfg.getfloat("MAIN","dropout", fallback=0.1),
        "norm":           _cfg.getboolean("MAIN", "norm", fallback=True),
        "hidden_sizes":   json.loads(_cfg.get("MAIN", "hidden_sizes", fallback="[256, 512, 1024]")),
        "emb_size":       _cfg.getint("MAIN", "emb_size", fallback=768),
        "vocab_size":     _cfg.getint("MAIN", "vocab_size", fallback=50257),
        "seqlen":         _cfg.getint("MAIN", "seqlen", fallback=128),
        "dtype":          _cfg.get("DS", "dtype", fallback="bf16"),
        "local_rank":     _cfg.getint("DS", "local_rank", fallback=-1),
        "checkpoint_dir": _cfg.get("DS","checkpoint_dir", fallback="results/"),
        "load_checkpoint_dir": _cfg.get("DS", "load_checkpoint_dir", fallback="None"),
    }
    

    parser = argparse.ArgumentParser()
    # Add local_rank as a global argument so it can be passed directly to the script
    subparsers = parser.add_subparsers(dest="cmd")
    train_parser = subparsers.add_parser("train", help="train")
    train_parser.add_argument("--wandb", type=bool, default=False)
    sample_parser = subparsers.add_parser("sample", help="sample")
    sample_parser.add_argument("--precond", help="preconditioning text", default="\n")
    sample_parser.add_argument("--num-tokens", type=int, default=256)
    sample_parser.add_argument("--wandb", type=bool, default=False)
    sample_parser.add_argument("ckpt")
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="Local rank passed from distributed launcher")
    args = parser.parse_args()

    if args.cmd == "train":
        cfg.update(vars(args))
        _logger.info(f"New training session with {cfg}")
        print(f"New training session with {cfg}")
        train(cfg)

    elif args.cmd == "sample":
        cfg.update(vars(args))
        _logger.info(f"New sampling session with {cfg}")
        output, perplex = sample(cfg)
        if output is None:
            sys.exit(1)
    else:
        parser.print_help()
        parser.error("too few arguments")
