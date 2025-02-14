"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import logging
import warnings
from itertools import islice
from pathlib import Path

import wandb
import numpy as np
import tiktoken
import torch
import torcheval
import torchvision
import torch.nn.functional as F
import torch.utils.data.dataloader
from torcheval.metrics.text import Perplexity
from examples.utils import *

import mingru

warnings.filterwarnings("ignore")

_logger = logging.getLogger("nlp")


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
        self.rnn = mingru.MinGRU(
            input_size=cfg["emb_size"],
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            residual=True,
            bias=True,
            norm=cfg["norm"],
        )
        self.ln = torch.nn.LayerNorm(cfg["hidden_sizes"][-1], bias=False)
        self.fc = torch.nn.Linear(cfg["hidden_sizes"][-1], cfg["vocab_size"])

    def forward(self, ids: torch.IntTensor, h: list[torch.Tensor] | None = None):
        x = self.emb(ids)
        x, h = self.rnn(x, h)
        x = self.ln(x)
        logits = self.fc(x)
        return logits, h


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

    model = NLPModel(cfg).to(dev)

    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.StepLR(
        opt,
        cfg["num_epochs"] - 2,
        gamma=0.1,
    )

    best_acc = 0
    if cfg["wandb"]:
        wandb.init(
            # Set the project where this run will be logged
            project="minGRU Shakespeare training",
            #name=f"epoch_{epoch}",
            name=f"minGRU epochs {cfg['num_epochs']}, hidden_sizes {cfg['hidden_sizes']}",
            # Track hyper parameters and run metadata
            config={
                "learning_rate": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "dropout": cfg["dropout"],
                "architecture": "minGRU",
                "dataset": "tiny-shakespeare",
                "epochs": cfg["num_epochs"],
                "sequence_length": cfg["seqlen"],
                "vocabulary_size": cfg["vocab_size"],
                "embedding_sizes": cfg["emb_size"],
                "normalize": cfg["norm"],
                "hidden_sizes": cfg["hidden_sizes"]
            }
        )
    detached_hidden_state = []
    for epoch in range(cfg["num_epochs"]):
        for step, (x, y) in enumerate(dl_train):
            x = x.to(dev)
            y = y.to(dev)
            """
            if detached_hidden_state != None and (step+1) % 20 == 0:
                synopsis = ""
                for the_state in detached_hidden_state:
                    synopsis += f"{the_state.shape}\t"
                #_logger.info(synopsis)
            if step == 0:
                y_hat, _ = model.forward(x)
            else: 
                detached_hidden_state = detach_tensors_in_list(hidden_state)
            """
            if (step % 775) == 0:
                detached_hidden_state = None
            y_hat, hidden_state = model.forward(x, detached_hidden_state if detached_hidden_state != [] else None)
            detached_hidden_state = detach_tensors_in_list(hidden_state)

            loss = crit(y_hat.permute(0, 2, 1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            perplexed = torch.exp(loss)
            _logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, perplexity: {perplexed:.4f}")
            if (step + 1) % 20 == 0:
                #_logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, perplexity: {perplexed:.4f}")
                wandb.log({"step":step+1, "loss":loss, "perplexity":perplexed}) if cfg["wandb"] else None
            if (step + 1) % 400 == 0:
                val_acc, val_loss = validate(model, dev, ds_val)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
                )
                if val_acc > best_acc:
                    _logger.info(f"New best model at epoch {epoch} step {step+1}")
                    scripted = torch.jit.script(model)
                    torch.jit.save(
                        scripted,
                        f"tmp/"
                        + Path(cfg["textfile"]).with_suffix(".nlp_best.pt").name,
                    )
                    best_acc = val_acc
                demo = generate_text(model, dev, prefix="\n", num_tokens=32, top_k=200)
                _logger.info(f"Sample model output: {demo}")
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
def generate_tokens(model, prefix_ids, temperature=1.0, top_k=None):
    assert prefix_ids.shape[1] > 0, "Need at least one start token"
    inp = prefix_ids
    h = None

    while True:
        logits, h = model.forward(inp, h)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            #_logger.info(f"v: {v}, top_k: {top_k}")
        probs = F.softmax(logits, dim=-1)
        #new_probs = [the_prob for the_prob in probs if the_prob != 0]
        #_logger.info(f"Probs: {probs}")
        idx_next = torch.multinomial(probs, num_samples=1)
        #_logger.info(f"Size of probs {probs.shape}")
        #_logger.info(f"Size of idx_next {idx_next.shape}")
        yield idx_next
        inp = idx_next


@torch.no_grad()
def sample(cfg):
    model = torch.jit.load(cfg["ckpt"])
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = generate_text(model, dev, args.precond, args.num_tokens, top_k=200)
    print(output)


if __name__ == "__main__":

    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler("tmp/nlp.log.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )

    cfg = {
        "seqlen": 256,
        "vocab_size": 50257,
        "emb_size": 768,
        "hidden_sizes": [512, 1024, 2048, 4096],
        #"hidden_sizes": [512, 1024, 2048, 4096, 8192],
        "norm": True,
        "dropout": 0.15,
        "num_epochs": 3,
        "batch_size": 64,
        "lr": 1e-3,
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    train_parser = subparsers.add_parser("train", help="train")
    train_parser.add_argument("textfile", help="Path to text file to train on.")
    train_parser.add_argument("--wandb", type=bool, default=False)
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
        _logger.info(f"New sampling session with {cfg}")
        cfg.update(vars(args))
        sample(cfg)
    else:
        parser.print_help()
        parser.error("too few arguments")
