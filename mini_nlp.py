"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import warnings
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader

import tiktoken

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
        )
        self.ln = torch.nn.LayerNorm(cfg["hidden_sizes"][-1], bias=False)
        self.fc = torch.nn.Linear(cfg["hidden_sizes"][-1], cfg["vocab_size"])

    def forward(self, ids: torch.IntTensor, h: list[torch.Tensor] | None = None):
        x = self.emb(ids)
        x, h = self.rnn(x, h)
        x = self.ln(x)
        logits = self.fc(x)
        return logits, h

    @torch.no_grad()
    def generate(self, cond_idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # cond_idx (B,S)

        inp = cond_idx
        h = None

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size

            logits, h = self.forward(inp, h)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            yield idx_next
            inp = idx_next


def train(cfg):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val = TokenIdDataset.from_textfile(cfg["input"], cfg["seqlen"])
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    ds_val = torch.utils.data.Subset(
        ds_val, np.random.choice(len(ds_val), 256, replace=False)
    )

    model = NLPModel(cfg).to(dev)

    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=5e-5,
    )
    sched = torch.optim.lr_scheduler.StepLR(
        opt,
        cfg["num_epochs"] - 2,
        gamma=0.1,
    )

    best_acc = 0
    for epoch in range(cfg["num_epochs"]):
        for step, (x, y) in enumerate(dl_train):
            x = x.to(dev)
            y = y.to(dev)
            y_hat, _ = model.forward(x)
            loss = crit(y_hat.permute(0, 2, 1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 20 == 0:
                _logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}")
            if (step + 1) % 400 == 0:
                val_acc, val_loss = validate(model, dev, ds_val)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
                )
                if val_acc > best_acc:
                    scripted = torch.jit.script(model)
                    torch.jit.save(scripted, "nlp_best.pt")
                    _logger.info("New best model")
                    best_acc = val_acc
                we = "All:"
                them = generate_and_decode(model, dev, we, num_tokens=32, top_k=200)
                _logger.info(f"{we}{them}")
                model.train()

        sched.step()

    # classifier = UCF101Cla, 256, 50257ssifier(cfg).to(dev)

    # transform = get_train_transform()
    # fold = cfg["ucf101_fold"]
    # ds = get_dataset(cfg, train=True, fold=fold, transform=transform)

    # indices = np.random.permutation(len(ds))
    # ds_train = torch.utils.data.Subset(ds, indices[:-200])
    # ds_val = torch.utils.data.Subset(ds, indices[-200:])

    # dl_train = torch.utils.data.DataLoader(
    #     ds_train,
    #     batch_size=cfg["batch_size"],
    #     shuffle=True,
    #     num_workers=cfg["dl_workers"],
    #     collate_fn=custom_collate,
    # )
    # dl_val = torch.utils.data.DataLoader(
    #     ds_val,
    #     batch_size=cfg["batch_size"],
    #     shuffle=True,
    #     num_workers=cfg["dl_workers"],
    #     collate_fn=custom_collate,
    # )
    # crit = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=cfg["lr"],
    #     weight_decay=5e-5,
    # )
    # sched = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     cfg["num_epochs"] - 2,
    #     gamma=0.1,
    # )

    # step = 0
    # best_acc = 0.0
    # best_loss = 1e5
    # for epoch in range(cfg["num_epochs"]):
    #     for video, labels in dl_train:
    #         video = video.to(dev)
    #         labels = labels.to(dev)
    #         logits, _ = classifier(video)
    #         loss = crit(logits, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         correct = (logits.argmax(1) == labels).sum().item()
    #         accuracy = 100 * correct / len(logits)
    #         if step % 20 == 0:
    #             _logger.info(
    #                 f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
    #             )
    #         if (step + 1) % 500 == 0:
    #             val_acc, val_loss = validate(classifier, dev, dl_val)
    #             _logger.info(
    #                 f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
    #             )
    #             if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
    #                 scripted = torch.jit.script(classifier)
    #                 torch.jit.save(scripted, "ucf101_classifier_best.pt")
    #                 _logger.info("New best model")
    #                 best_acc = val_acc
    #                 best_loss = val_loss
    #         step += 1
    #     sched.step()

    # return classifier


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


from itertools import islice


@torch.no_grad()
def generate_and_decode(
    model: NLPModel,
    dev: torch.device,
    cond: str,
    num_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode_ordinary(cond)).to(dev).unsqueeze(0)
    gen = model.generate(
        ids,
        max_new_tokens=num_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    new = torch.cat(list(islice(gen, num_tokens)), dim=1)
    return enc.decode(new[0].cpu().tolist())


@torch.no_grad()
def sample(cfg, model):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    import os
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler("shakespeare.log.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )

    cfg = {
        "seqlen": 256,
        "vocab_size": 50257,
        "emb_size": 768,
        "hidden_sizes": [64, 128, 256, 256, 512],
        "dropout": 0.0,
        "num_epochs": 7,
        "batch_size": 64,
        "lr": 1e-4,
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    train_parser = subparsers.add_parser("train", help="train")
    train_parser.add_argument("input", help="Path to TinyShakespeare file.")
    sample_parser = subparsers.add_parser("sample", help="sample")
    sample_parser.add_argument("--ckpt", default="shakespeare_best.pt")
    args = parser.parse_args()

    if args.cmd == "train":
        cfg["input"] = args.input
        _logger.info(f"New training session with {cfg}")
        train(cfg)
        # train(cfg)

    elif args.cmd == "sample":
        _logger.info(f"New sampling session with {cfg}")
        model = torch.jit.load(args.ckpt)
        sample(cfg, model)
