import argparse
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src.data import CHECKPOINT_DIR, DATA_DIR, history_dir, load_trainval
from src.models.cnn import build_model

IMAGE_SIZE = 224
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Stronger augmentation for models trained from scratch: random scale and position, and mild
# colour changes. No hue shift, since fur colour is what separates some breeds.
strong_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.35, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Models not listed here use train_transform
TRAIN_TRANSFORMS = {"pet_cnn_v2": strong_train_transform}

# Same scale as training (resize to 256, then 224 crop), but centred and without randomness
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(model, debug):
    _, train_idx, val_idx = load_trainval()
    if debug:
        train_idx, val_idx = train_idx[:64], val_idx[:64]

    train_ds = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                             transform=TRAIN_TRANSFORMS.get(model, train_transform))
    val_ds = OxfordIIITPet(DATA_DIR, split="trainval",
                           target_types="category", transform=eval_transform)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE,
                            num_workers=2)
    return train_loader, val_loader


def make_scheduler(optimizer, epochs, warmup):
    """Cosine decay over all epochs, optionally after a linear warmup from 10 % of the learning rate."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    if warmup == 0:
        return cosine
    ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [ramp, cosine], milestones=[warmup])


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Returns (loss, accuracy) on the training data, measured while training (with augmentation)."""
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Returns (loss, accuracy) on the validation data."""
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += loss_fn(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def save_history(path, history):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def load_checkpoint(path, device="cpu"):
    """Returns (state_dict, metadata). Also reads old checkpoints that only hold a state_dict."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state" in checkpoint:
        return checkpoint["model_state"], {k: v for k, v in checkpoint.items() if k != "model_state"}
    return checkpoint, {}


def train(args, seed):
    """Trains one model with one seed. Returns the best validation accuracy."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {args.model}, seed {seed} (device: {device}) ===")

    train_loader, val_loader = make_loaders(args.model, args.debug)
    model = build_model(args.model).to(device)
    # Label smoothing only in the training loss, so val loss stays comparable between models
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup)

    run_name = f"{args.model}_seed{seed}" + ("_debug" if args.debug else "")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history_dir(args.model).mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"
    history_path = history_dir(args.model) / f"{run_name}.csv"

    history = []
    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = train_one_epoch(model, train_loader, train_loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, eval_loss_fn, device)
        scheduler.step()
        print(f"Epoch {epoch}: lr={lr:.2e}, train loss={train_loss:.4f} acc={train_accuracy:.4f}, "
              f"val loss={val_loss:.4f} acc={val_accuracy:.4f}")

        history.append({"epoch": epoch, "lr": lr,
                        "train_loss": round(train_loss, 4), "train_accuracy": round(train_accuracy, 4),
                        "val_loss": round(val_loss, 4), "val_accuracy": round(val_accuracy, 4)})
        save_history(history_path, history)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state": model.state_dict(),
                "model": args.model,
                "epoch": epoch,
                "val_accuracy": val_accuracy,
                "seed": seed,
                "args": vars(args),
            }, checkpoint_path)

    print(f"Best val_accuracy: {best_val_accuracy:.4f}")
    print(f"Saved {checkpoint_path} and {history_path}")
    return best_val_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=[0],
                        help="one or more seeds, trained one after another, e.g. --seeds 0 1 2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=0,
                        help="epochs of linear learning rate warmup before the cosine decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    best = {seed: train(args, seed) for seed in args.seeds}
    if len(best) > 1:
        values = np.array(list(best.values()))
        print(f"\nBest val_accuracy per seed: {', '.join(f'{s}: {v:.4f}' for s, v in best.items())}")
        print(f"Mean: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()