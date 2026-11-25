import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from data import DATA_DIR, load_trainval
from models import build_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
IMAGE_SIZE = 224
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Starting points per model. A network trained from scratch needs a higher learning rate and
# more epochs than one that is fine-tuned from ImageNet weights. Override with the flags.
DEFAULTS = {
    "pet_cnn":            {"epochs": 60, "lr": 1e-3, "patience": 15},
    "resnet50_untrained": {"epochs": 60, "lr": 1e-3, "patience": 15},
    "resnet50":           {"epochs": 15, "lr": 1e-4, "patience": 5},
}

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(debug, num_workers):
    _, train_idx, val_idx = load_trainval()
    if debug:
        train_idx, val_idx = train_idx[:64], val_idx[:64]

    train_ds = OxfordIIITPet(DATA_DIR, split="trainval",
                             target_types="category", transform=train_transform)
    val_ds = OxfordIIITPet(DATA_DIR, split="trainval",
                           target_types="category", transform=eval_transform)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE,
                            num_workers=num_workers)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
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
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += loss_fn(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="default depends on the model, see DEFAULTS")
    parser.add_argument("--lr", type=float, help="default depends on the model, see DEFAULTS")
    parser.add_argument("--patience", type=int,
                        help="stop after this many epochs without a better validation accuracy")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    print(f"Model={args.model}, seed={args.seed}, epochs={args.epochs}, lr={args.lr}, "
          f"patience={args.patience}, weight_decay={args.weight_decay}")

    train_loader, val_loader = make_loaders(args.debug, args.num_workers)
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.model}_seed{args.seed}" + ("_debug" if args.debug else "")  # debug runs are never evaluated
    checkpoint_path = args.checkpoints / f"{run_name}.pt"

    history, best_val_accuracy, best_epoch, start = [], -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "val_loss": val_loss,
                        "val_accuracy": val_accuracy})
        print(f"Epoch {epoch}: loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
              f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:   # model selection on validation data only
            best_val_accuracy, best_epoch = val_accuracy, epoch
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: no better validation accuracy for {args.patience} epochs")
            break

    minutes = (time.time() - start) / 60
    print(f"Best val_accuracy: {best_val_accuracy:.4f} (epoch {best_epoch}), {minutes:.1f} min")

    # Saved for learning curves and for reporting settings and compute in the report
    log = {"model": args.model, "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
           "patience": args.patience, "weight_decay": args.weight_decay, "batch_size": BATCH_SIZE,
           "image_size": IMAGE_SIZE, "parameters": n_params, "best_epoch": best_epoch,
           "best_val_accuracy": best_val_accuracy, "minutes": round(minutes, 2),
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "debug": args.debug, "history": history}
    history_dir = args.results / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{run_name}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
