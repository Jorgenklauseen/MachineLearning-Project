import argparse
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src.data import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR, load_trainval
from src.models.cnn import build_model

HISTORY_DIR = RESULTS_DIR / "training"
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


def make_loaders(debug):
    _, train_idx, val_idx = load_trainval()
    if debug:
        train_idx, val_idx = train_idx[:64], val_idx[:64]

    train_ds = OxfordIIITPet(DATA_DIR, split="trainval",
                             target_types="category", transform=train_transform)
    val_ds = OxfordIIITPet(DATA_DIR, split="trainval",
                           target_types="category", transform=eval_transform)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE,
                            num_workers=2)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
    return correct / len(loader.dataset)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = make_loaders(args.debug)
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"{args.model}_seed{args.seed}" + ("_debug" if args.debug else "")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"
    history_path = HISTORY_DIR / f"{run_name}.csv"

    history = []
    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch}: lr={lr:.2e}, loss={train_loss:.4f}, val_accuracy={val_accuracy:.4f}")

        history.append({"epoch": epoch, "lr": lr, "train_loss": round(train_loss, 4),
                        "val_accuracy": round(val_accuracy, 4)})
        save_history(history_path, history)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state": model.state_dict(),
                "model": args.model,
                "epoch": epoch,
                "val_accuracy": val_accuracy,
                "args": vars(args),
            }, checkpoint_path)

    print(f"Best val_accuracy: {best_val_accuracy:.4f}")
    print(f"Saved {checkpoint_path} and {history_path}")


if __name__ == "__main__":
    main()