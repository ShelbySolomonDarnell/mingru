"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import logging
import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torchvision import tv_tensors
from torchvision.datasets import UCF101
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import v2

import mingru

warnings.filterwarnings("ignore")

_logger = logging.getLogger("ucf101")


class ToVideo(torch.nn.Module):
    def forward(self, data):
        # Do some transformations
        return tv_tensors.Video(data)


def get_train_transform():
    return v2.Compose(
        [
            ToVideo(),
            # v2.RandomPerspective(distortion_scale=0.2),
            v2.RandomChoice(
                [v2.RandomResizedCrop(224, scale=(0.3, 1.0)), v2.RandomCrop(224)]
            ),
            v2.RandomChannelPermutation(),
            v2.ColorJitter(brightness=0.5, hue=0.3, saturation=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.FiveCrop(224),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_quick_test_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataset(cfg, train: bool, fold: int, transform: Callable):
    ds = UCF101(
        cfg["ucf101_path"],
        cfg["ucf101_annpath"],
        frames_per_clip=10,
        step_between_clips=10,
        fold=fold,
        train=train,
        output_format="TCHW",
        num_workers=cfg["ucf101_workers"],
        transform=transform,
    )

    # Compute video index for each dataset index
    idx2video = np.array(
        [ds.video_clips.get_clip_location(idx)[0] for idx in range(len(ds))]
    )
    video2label = np.array(
        [ds.samples[ds.indices[vidx]][1] for vidx in np.unique(idx2video)]
    )

    return ds, idx2video, video2label


def split_train_val(
    ds: UCF101,
    idx2video: np.ndarray,
    video2label: np.ndarray,
    video_split_ratio: float = 0.95,
    max_val_clips_per_video: int = 1,
):
    """Splits training set into training and validation on a per-video basis.

    To avoid overfitting on UCF101 we found it important to split train/val
    not based on clips, but on a per video basis. This improves accuracy
    for the base model by 8% on test fold 1.
    """

    idx_train = []
    idx_val = []

    for i in range(101):
        videos = np.where(video2label == i)[0]
        videos = np.random.permutation(videos)
        r = min(int(len(videos) * video_split_ratio), len(videos) - 2)
        train_video_idx = videos[:r]
        val_video_idx = videos[r:]
        train_clip_idx = np.concatenate(
            [np.where(idx2video == v)[0] for v in train_video_idx]
        )
        val_clip_idx = np.concatenate(
            [np.where(idx2video == v)[0] for v in val_video_idx]
        )
        idx_train.extend(train_clip_idx.tolist())
        idx_val.extend(val_clip_idx[:max_val_clips_per_video].tolist())

    ds_train = torch.utils.data.Subset(ds, idx_train)
    ds_val = torch.utils.data.Subset(ds, idx_val)

    return ds_train, ds_val


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


# fold1: 96% top1 on val, 78% on test split
class UCF101Classifier(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        parts = [0, 5, 10, 17, 24, 31]
        output_sizes = [64, 128, 256, 512, 512]
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.feature_parts = torch.nn.ModuleList(
            [backbone[f:t] for f, t in zip(parts[:-1], parts[1:])]
        )

        rnns = [
            mingru.MinConv2dGRU(
                input_size=s,
                hidden_sizes=[s, s],
                kernel_sizes=3,
                strides=2,
                paddings=1,
                dropout=cfg["dropout"],
                norm=cfg["norm"],
                residual=True,
                bias=True,
            )
            for s in output_sizes
        ]
        if cfg["bidirectional"]:
            rnns = [mingru.Bidirectional(rnn) for rnn in rnns]

        d = 2 if cfg["bidirectional"] else 1
        h, w = 2, 2
        mult = h * w * d
        fcs = [torch.nn.Linear(s * mult, 101) for s in output_sizes]

        self.heads = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"rnn": rnn, "fc": fc}) for rnn, fc in zip(rnns, fcs)]
        )

    def forward(self, video):
        B, S = video.shape[:2]

        x = video.flatten(0, 1)
        features = []
        for part in self.feature_parts:
            x = part(x)
            features.append(x.unflatten(0, (B, S)))

        logits = []
        for idx, head in enumerate(self.heads):
            out, _ = head.rnn(features[idx])
            out = F.adaptive_avg_pool2d(out[:, -1], (2, 2))
            logits.append(head.fc(out.flatten(1)))

        logits = torch.stack(logits)
        logits = logits.mean(0)
        return logits, None


def train(cfg):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = UCF101Classifier(cfg).to(dev)

    transform = get_train_transform()
    fold = cfg["ucf101_fold"]
    ds, idx2video, video2label = get_dataset(
        cfg, train=True, fold=fold, transform=transform
    )

    ds_train, ds_val = split_train_val(
        ds,
        idx2video,
        video2label,
        video_split_ratio=0.95,
        max_val_clips_per_video=2,
    )
    _logger.info(f"Train DS: {len(ds_train)}, Val DS: {len(ds_val)}")

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )
    crit = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=cfg["lr"], weight_decay=5e-5
    )
    sched = torch.optim.lr_scheduler.StepLR(optimizer, cfg["num_epochs"] - 2, gamma=0.1)

    step = 0
    best_acc = 0.0
    best_loss = 1e5
    for epoch in range(cfg["num_epochs"]):
        for video, labels in dl_train:
            video = video.to(dev)
            labels = labels.to(dev)
            logits, _ = classifier(video)
            loss = crit(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (logits.argmax(1) == labels).sum().item()
            accuracy = 100 * correct / len(logits)
            if step % 20 == 0:
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
                )
            if (step + 1) % 400 == 0:
                val_acc, val_loss = validate(classifier, dev, dl_val)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
                )
                if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                    scripted = torch.jit.script(classifier)
                    torch.jit.save(scripted, f"tmp/video_classifier_best_{step:04d}.pt")
                    _logger.info("New best model")
                    best_acc = val_acc
                    best_loss = val_loss
            step += 1
        sched.step()

    return classifier


@torch.no_grad()
def validate(classifier: torch.nn.Module, dev: torch.device, dl, verbose: bool = False):
    classifier.eval()

    total = 0
    total_correct = 0
    total_loss = 0
    crit = torch.nn.CrossEntropyLoss()

    for video, labels in dl:
        video = video.to(dev)
        labels = labels.to(dev)
        logits, _ = classifier(video)
        loss = crit(logits, labels)

        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(video)
        total_loss += loss.item()
        if verbose:
            _logger.info(f"Acc: {total_correct/total}")

    acc = total_correct / total
    avg_loss = total_loss / total
    classifier.train()
    return acc, avg_loss


@torch.no_grad()
def test(cfg, classifier):
    """Test classifier

    Follows the test protocol of
    https://arxiv.org/pdf/1511.06432

    For each video we randomly choose 25 clips of length 10. For each image of each clip
    we obtain 10 inputs: five-crops plus horizontal flipping. The final pre-dection for
    a video is then computed as the majority vote of 25*10*10 predictions (in the original
    paper averaging is used instead). Note that for the base classifier, predicition
    averaging is already applied for each of the 5 feature resolutions.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    ds, _, _ = get_dataset(cfg, train=False, transform=None, fold=cfg["ucf101_fold"])
    trans = get_test_transform()

    total = 0
    total_correct = 0

    # For each video
    labels = [ds.samples[i][1] for i in ds.indices]
    n_videos = len(ds.video_clips.video_paths)
    for vidx, label in zip(range(n_videos), labels):
        clips = ds.video_clips.subset([vidx])
        # randomly select up to 25 clips
        cids = np.random.permutation(len(clips))[:25]
        all_logits = []
        for cidx in cids:
            video, _, _, _ = clips.get_clip(cidx)

            # Five-crops + flip
            crops = trans(video)
            crops = torch.stack(crops, 0)
            hcrops = v2.functional.horizontal_flip(crops)
            input = torch.cat((crops, hcrops), 0).to(dev)

            logits, _ = classifier(input)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, 0)
        pred = all_logits.argmax(-1)
        unique, counts = torch.unique(pred, return_counts=True)
        majority_vote = unique[counts.argmax()]
        # pred = all_logits.mean(0)
        total_correct += majority_vote.item() == label
        total += 1
        _logger.info(f"{vidx+1}/{n_videos}, acc {total_correct/total:.2f}")

    _logger.info(f"test acc {total_correct/total:.2f}")


@torch.no_grad()
def quick_test(cfg, classifier):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    trans = get_quick_test_transform()
    ds, _, _ = get_dataset(cfg, train=False, transform=trans, fold=cfg["ucf101_fold"])
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )
    acc, loss = validate(classifier, dev, dl, verbose=True)
    _logger.info(f"Test accuracy {acc:.2f}")


if __name__ == "__main__":

    import argparse
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler("tmp/video_classifier.log.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )

    cfg = {
        "ucf101_path": os.environ["UCF101_PATH"],
        "ucf101_annpath": os.environ["UCF101_ANNPATH"],
        "ucf101_fold": 1,
        "ucf101_workers": 10,
        "dl_workers": 4,
        "bidirectional": False,
        "norm": True,
        "dropout": 0.15,
        "num_epochs": 7,
        "batch_size": 16,
        "lr": 1e-4,
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    train_parser = subparsers.add_parser("train", help="train")
    train_parser.add_argument(
        "-f", "--fold", type=int, choices=[1, 2, 3], required=True
    )
    test_parser = subparsers.add_parser("test", help="test")
    test_parser.add_argument("ckpt", help="Checkpoint to load")
    test_parser.add_argument("--quick-test", action="store_true", default=False)
    test_parser.add_argument("-f", "--fold", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()

    if args.cmd == "train":
        _logger.info(f"New training session with {cfg}")
        cfg["ucf101_fold"] = args.fold
        classifier = train(cfg)
        quick_test(cfg, classifier)
    elif args.cmd == "test":
        _logger.info(f"New testing session with {cfg}")
        scripted = torch.jit.load(args.ckpt)
        cfg["ucf101_fold"] = args.fold
        cfg["quick_test"] = args.quick_test
        if args.quick_test:
            quick_test(cfg, scripted)
        else:
            test(cfg, scripted)
    else:
        parser.print_help()
        parser.error("too few arguments")
