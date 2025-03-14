"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import logging
from logging.handlers import RotatingFileHandler
import warnings
from itertools import islice
from pathlib import Path

import sys
import json
import wandb
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.nn import Linear
from examples.utils import *
from examples.utils import cfg as _cfg

import mingru
import minlstm

warnings.filterwarnings("ignore")

_logger = logging.getLogger("nlp")
handler = RotatingFileHandler("tmp/minrnn.boros.log", maxBytes=512000, backupCount=100)
_logger.addHandler(handler)

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

        self.emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_size"])
        self.rnn = minlstm.MinLSTM( 
            input_size=cfg["emb_size"],
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            residual=True,
            bias=True,
            norm=cfg["norm"],
        ) if cfg["arch"]=='minLSTM' else mingru.MinGRU(
            input_size=cfg["emb_size"],
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            residual=True,
            bias=False,
            norm=cfg["norm"],
        )

        model_bias = True if cfg["arch"]=='minLSTM' else False
        self.ln = torch.nn.LayerNorm(cfg["hidden_sizes"][-1], model_bias)
        self.fc = torch.nn.Linear(cfg["hidden_sizes"][-1], cfg["vocab_size"])

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
        
        # Handle different RNN architectures
        if isinstance(self.rnn, minlstm.MinLSTM):
            # MinLSTM expects h and c together as a tuple
            if h is not None and c is not None:
                # Pass both hidden states as a tuple
                x, hidden_state = self.rnn(x, (h, c))
            else:
                # Let the RNN initialize the hidden states
                x, hidden_state = self.rnn(x)
            # Ensure hidden_state is always a tuple of (h, c)
            if not isinstance(hidden_state, tuple):
                _logger.warning("MinLSTM returned non-tuple hidden state, fixing format")
                if isinstance(hidden_state, list):
                    # Split the list in half for h and c if needed
                    mid = len(hidden_state) // 2
                    hidden_state = (hidden_state[:mid], hidden_state[mid:])
        else:  # MinGRU
            x, hidden_state = self.rnn(x, h)
            # Ensure hidden_state is always a list for MinGRU
            if not isinstance(hidden_state, list) and hidden_state is not None:
                _logger.warning("MinGRU returned non-list hidden state, fixing format")
                hidden_state = [hidden_state] if torch.is_tensor(hidden_state) else list(hidden_state)
            
        x = self.ln(x)
        logits = self.fc(x)
        
        return logits, hidden_state

def init_optimizer(params, the_cfg):
    result = None
    if the_cfg["optim"] == "sgd":
        result = torch.optim.SGD(
            params,
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=5e-4
        )
    else:
        result = torch.optim.Adam(
            params,
            lr=cfg["lr"],
            weight_decay=5e-4,
    )
    return result

def train(cfg):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val = TokenIdDataset.from_textfile(cfg["textfile"], cfg["seqlen"])
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    _logger.info(f"Number of examples in dataset {len(dl_train.dataset)}")
    _logger.info(f"Number of batches in dataset {len(dl_train)}")

    ds_val = torch.utils.data.Subset(
        ds_val, np.random.choice(len(ds_val), 256, replace=False)
    )

    _logger.info(f"Number of examples in test dataset {len(ds_val.dataset)}")
    _logger.info(f"Number of batches in test dataset {len(ds_val)}")

    model = NLPModel(cfg).to(dev)

    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    opt = init_optimizer(model.parameters(),cfg)

    sched = torch.optim.lr_scheduler.StepLR(
        opt,
        cfg["num_epochs"] - 2,
        gamma=0.1,
    )

    best_acc = 0  # Start with 0 since we're tracking accuracy (0-1)
    if cfg["wandb"]:
        wandb.init(
            # Set the project where this run will be logged
            project=cfg['arch'] + " Shakespeare training",
            #name=f"epoch_{epoch}",
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
                "optimizer":       cfg["optim"]
            }
        )
    # Initialize hidden state variables based on architecture
    if cfg["arch"] == "minLSTM":
        detached_h_state = []
        detached_c_state = []
    else:  # minGRU
        detached_hidden_state = []
        
    for epoch in range(cfg["num_epochs"]):
        for step, (x, y) in enumerate(dl_train):
            x = x.to(dev)
            y = y.to(dev)

            if (step % (len(dl_train)-1)) == 0:
                # Reset hidden states at the beginning of each epoch
                if cfg["arch"] == "minLSTM":
                    detached_h_state = []
                    detached_c_state = []
                else:  # minGRU
                    detached_hidden_state = None
                    
            # Forward pass with appropriate hidden state handling
            if cfg["arch"] == "minLSTM":
                # For MinLSTM, pass h and c together
                h_input = detached_h_state if detached_h_state else None
                c_input = detached_c_state if detached_c_state else None
                y_hat, hidden_state = model.forward(x, h_input, c_input)
                
                # Ensure hidden_state is a tuple before detaching
                if not isinstance(hidden_state, tuple):
                    _logger.warning("Received non-tuple hidden state from MinLSTM, fixing format")
                    if isinstance(hidden_state, list):
                        # Split the list in half for h and c if needed
                        mid = len(hidden_state) // 2
                        hidden_state = (hidden_state[:mid], hidden_state[mid:])
                    else:
                        # Create a default format if we can't determine the structure
                        hidden_state = (hidden_state, hidden_state) if torch.is_tensor(hidden_state) else ([], [])
                
                # Detach the entire hidden state tuple
                detached_hidden_state = detach_tensors_in_list(hidden_state)
                # Unpack for next iteration
                detached_h_state, detached_c_state = detached_hidden_state
            else:  # minGRU
                y_hat, hidden_state = model.forward(x, detached_hidden_state if detached_hidden_state != [] else None)
                
                # Ensure hidden_state is a list before detaching
                if not isinstance(hidden_state, list) and hidden_state is not None:
                    _logger.warning("Received non-list hidden state from MinGRU, fixing format")
                    hidden_state = [hidden_state] if torch.is_tensor(hidden_state) else list(hidden_state)
                
                detached_hidden_state = detach_tensors_in_list(hidden_state)

            loss = crit(y_hat.permute(0, 2, 1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            perplexed = torch.exp(loss)

            #_logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, perplexity: {perplexed:.4f}")
            if (step + 1) % 20 == 0:
                _logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, training perplexity: {perplexed:.4f}")
                wandb.log({"step":step+1, "loss":loss, "training_perplexity":perplexed}) if cfg["wandb"] else None
            if (step + 1) % 200 == 0:
                val_acc, val_loss = validate(model, dev, ds_val)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
                )
                if val_acc > best_acc:
                    _logger.info(f"New best model at epoch {epoch} step {step+1}")
                    scripted = torch.jit.script(model)
                    model_name = f"nlp_best.epochs{cfg['num_epochs']}_{cfg['arch']}_hidden{'_'.join(map(str, cfg['hidden_sizes']))}.pt"
                    torch.jit.save(
                        scripted,
                        f"tmp/{model_name}",
                    )
                    best_acc = val_acc
                demo, sample_perplexity = generate_text_mbili(model, dev, prefix="\n", num_tokens=32, top_k=200)
                wandb.log(
                    {"Epoch":epoch+1,"Step":step+1,"Validation Accuracy":val_acc*100, "Validation Loss": val_loss, "Sample perplexity": sample_perplexity}
                ) if cfg["wandb"] else None
                _logger.info(f"Sample perplexity: {sample_perplexity}\nSample model output: {demo}")
                model.train()

        sched.step()
        detached_hidden_state = []
    wandb.finish() if cfg["wandb"] else None


@torch.no_grad()
def validate(
    model: NLPModel,
    dev: torch.device,
    ds: TokenIdDataset,
):
    model.eval()

    total = 0
    total_loss = 0
    total_correct = 0
    crit = torch.nn.CrossEntropyLoss()

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    for ids, labels in dl:
        ids = ids.to(dev)
        labels = labels.to(dev)
        logits, _ = model(ids)
        loss = crit(logits.permute(0, 2, 1), labels)

        total_correct += (logits.argmax(2) == labels).sum().item()
        total += ids.shape[0] * ids.shape[1]
        total_loss += loss.item()

    acc = total_correct / total
    avg_loss = total_loss / len(dl)

    return acc, avg_loss


@torch.no_grad()
def generate_text(
    model: NLPModel,
    dev: torch.device,
    prefix: str,
    num_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
) -> str:
    #perplexed = Perplexity()
    enc = tiktoken.get_encoding("gpt2")
    ids = (
        torch.tensor(
            enc.encode_ordinary(prefix),
            dtype=int,
        )
        .to(dev)
        .unsqueeze(0)
    )
    gen = generate_tokens(
        model,
        prefix_ids=ids,
        temperature=temperature,
        top_k=top_k,
    )
    #new = torch_cat_with_check(list(islice(gen, num_tokens)), dim=1)
    new = torch.cat(list(islice(gen, num_tokens)), dim=1)
    #perplexed.update(ids.squeeze(0), new)
    #perplexed.compute()
    #_logger.info(f"Perplexity: {perplexed}")
    #wandb.log({"perplexity":perplexed})
    return enc.decode(new[0].cpu().tolist())

@torch.no_grad()
def generate_text_mbili(
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

    gen = generate_tokens_mbili(model, prefix_ids=ids, temperature=temperature, top_k=top_k)
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
def generate_tokens_mbili(model, prefix_ids, temperature=1.0, top_k=None):
    assert prefix_ids.shape[1] > 0, "Need at least one start token"
    inp = prefix_ids
    
    # Initialize hidden states based on model architecture
    if hasattr(model, 'rnn') and isinstance(model.rnn, minlstm.MinLSTM):
        h = None
        c = None
        is_lstm = True
    else:
        h = None
        is_lstm = False
        
    all_probs = [] # Store all probabilities

    try:
        while True:
            if is_lstm:
                # For MinLSTM, pass both h and c
                logits, hidden_state = model.forward(inp, h, c)
                # Unpack for next iteration
                h, c = hidden_state
            else:
                # For MinGRU, just pass h
                logits, h = model.forward(inp, h)
                
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            yield idx_next, probs # Yield both token and probabilities
            inp = idx_next
    except Exception as e:
        _logger.error(f"Error in generate_tokens_mbili: {str(e)}")
        # Yield a dummy token and probability to avoid breaking the caller
        yield torch.zeros_like(prefix_ids[:, :1]), torch.ones((1, 1, 50257)) / 50257




@torch.no_grad()
def generate_tokens(model, prefix_ids, temperature=1.0, top_k=None):
    assert prefix_ids.shape[1] > 0, "Need at least one start token"
    inp = prefix_ids
    
    # Initialize hidden states based on model architecture
    if hasattr(model, 'rnn') and isinstance(model.rnn, minlstm.MinLSTM):
        h = None
        c = None
        is_lstm = True
    else:
        h = None
        is_lstm = False

    try:
        while True:
            if is_lstm:
                # For MinLSTM, pass both h and c
                logits, hidden_state = model.forward(inp, h, c)
                # Unpack for next iteration
                h, c = hidden_state
            else:
                # For MinGRU, just pass h
                logits, h = model.forward(inp, h)
                
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            yield idx_next
            inp = idx_next
    except Exception as e:
        _logger.error(f"Error in generate_tokens: {str(e)}")
        yield torch.zeros_like(prefix_ids[:, :1])


@torch.no_grad()
def sample(cfg):
    model = torch.jit.load(cfg["ckpt"])
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output, perplex = generate_text_mbili(model, dev, cfg["precond"], cfg["num_tokens"], top_k=200)
    _logger.info(f"Sample perplexity is {perplex}")
    print("[Sampling perplexity] {0}\n{1}\n".format(perplex, output))
    
    # Return the results for potential further use
    return output, perplex


if __name__ == "__main__":

    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            #logging.RotatingFileHandler("tmp/nlp-boros.log.txt", maxBytes=512000, backupCount=100),
            #logging.FileHandler("tmp/nlp-boros.log.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )

    cfg = {
        "dataset":      _cfg.get("MAIN","datasetA"), #"tiny-shakespeare",
        "arch":         _cfg.get("MAIN", "arch"), #"minGRU" or "minLSTM",
        "lr":           _cfg.getfloat("MAIN","lr"),
        "batch_size":   _cfg.getint("MAIN","batch_size"), #cfg["batch_size"],
        "num_epochs":   _cfg.getint("MAIN","num_epochs"), #cfg["num_epochs"],
        "dropout":      _cfg.getfloat("MAIN","dropout"), #cfg["dropout"],
        "norm":         _cfg.getboolean("MAIN", "norm"), #cfg["norm"],
        "hidden_sizes": json.loads(_cfg.get("MAIN", "hidden_sizes")), #cfg["hidden_sizes"]
        "emb_size":     _cfg.getint("MAIN", "emb_size"), #cfg["emb_size"],
        "vocab_size":   _cfg.getint("MAIN", "vocab_size"), #cfg["vocab_size"],
        "seqlen":       _cfg.getint("MAIN", "seqlen") #cfg["seqlen"],
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    train_parser = subparsers.add_parser("train", help="train")
    train_parser.add_argument("textfile", help="Path to text file to train on.")
    train_parser.add_argument("--wandb", type=bool, default=False)
    train_parser.add_argument("--optim", type=str, default="adamw")
    sample_parser = subparsers.add_parser("sample", help="sample")
    sample_parser.add_argument("--precond", help="preconditioning text", default="\n")
    sample_parser.add_argument("--num-tokens", type=int, default=256)
    sample_parser.add_argument("--wandb", type=bool, default=False)
    sample_parser.add_argument("ckpt")
    args = parser.parse_args()

    if args.cmd == "train":
        cfg.update(vars(args))
        _logger.info(f"New training session with {cfg}")
        train(cfg)
        # train(cfg)
    elif args.cmd == "sample":
        cfg.update(vars(args))
        _logger.info(f"New sampling session with {cfg}")
        sample(cfg)
    else:
        parser.print_help()
        parser.error("too few arguments")
