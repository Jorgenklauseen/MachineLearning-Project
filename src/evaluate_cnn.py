"""Evaluate a trained CNN checkpoint on the test set.

1. The official test split (3669 images): accuracy with 95 % CI, macro-F1 and per-breed table.
2. The background test (data/background_test, made by make_background_test.py), if it exists:
   the same summary as evaluate_svm_background.py, so the numbers can be compared directly.

Example:
    python src/evaluate_cnn.py --model resnet50 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet

from analyze_svm import per_class_stats, save_per_class
from data import DATA_DIR, SPLIT_SEED
from evaluate_svm_background import (BG_BINS, VARIANTS, background_share, bootstrap_ci,
                                     print_row, read_metadata, summarize, write_csv)
from models import NUM_CLASSES, build_model
from train_cnn import BATCH_SIZE, CHECKPOINT_DIR, eval_transform, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class ImageFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_transform(Image.open(self.paths[i]).convert("RGB")), 0


@torch.no_grad()
def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
    preds, labels = [], []
    for images, y in loader:
        preds.append(model(images.to(device)).argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def evaluate_test_split(model, run_name, device, out):
    test = OxfordIIITPet(DATA_DIR, split="test", target_types="category", transform=eval_transform)
    pred, y = predict(model, test, device)
    correct = (pred == y).astype(float)
    lo, hi = bootstrap_ci(correct, np.random.default_rng(SPLIT_SEED))
    f1 = f1_score(y, pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)
    print(f"\nTest split: n={len(y)}  acc={correct.mean():.1%} [{lo:.1%}, {hi:.1%}]  macro-F1={f1:.3f}")

    write_csv(out / f"{run_name}_test_summary.csv", [{
        "model": run_name, "n": len(y), "accuracy": correct.mean(),
        "ci_low": lo, "ci_high": hi, "macro_f1": f1,
    }])
    save_per_class(test.classes, per_class_stats(y, pred, NUM_CLASSES), out / f"{run_name}_per_class.csv")


def evaluate_background(model, run_name, device, data, out):
    if not (data / "metadata.csv").exists():
        print(f"\nSkipping background test: {data} not found. Run make_background_test.py first.")
        return
    names, y, breeds = read_metadata(data)
    bg_share = background_share(data, names)
    bg_bin = np.digitize(bg_share, BG_BINS) - 1

    preds = {v: predict(model, ImageFiles([data / v / f"{n}.png" for n in names]), device)[0]
             for v in VARIANTS}
    rng = np.random.default_rng(SPLIT_SEED)
    rows = [summarize(run_name, v, y, preds[v], preds["original"], breeds, rng) for v in VARIANTS]
    for b in range(len(BG_BINS) - 1):
        m = bg_bin == b
        if m.any():
            label = f"bg_only [{BG_BINS[b]:.0%}-{min(BG_BINS[b + 1], 1):.0%} bg]"
            rows.append(summarize(run_name, label, y[m], preds["bg_only"][m],
                                  preds["original"][m], breeds, rng))
    print("\nBackground test:")
    for r in rows:
        print_row(r)

    predictions = [{"model": run_name, "name": name, "true": breeds[int(y[i])],
                    "bg_share": round(float(bg_share[i]), 4),
                    **{f"pred_{v}": breeds[int(preds[v][i])] for v in VARIANTS}}
                   for i, name in enumerate(names)]
    write_csv(out / f"experiment2_{run_name}_summary.csv", rows)
    write_csv(out / f"experiment2_{run_name}_predictions.csv", predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="resnet50, resnet50_untrained or pet_cnn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "background_test")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    run_name = f"{args.model}_seed{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, meta = load_checkpoint(CHECKPOINT_DIR / f"{run_name}.pt", device)
    if meta:
        print(f"{run_name}: best epoch {meta['epoch']}, val_accuracy {meta['val_accuracy']:.4f}")
    else:
        print(f"{run_name}: old checkpoint without metadata (trained before the eval transform change)")

    model = build_model(args.model)
    model.load_state_dict(state)
    model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    evaluate_test_split(model, run_name, device, args.out)
    evaluate_background(model, run_name, device, args.data, args.out)


if __name__ == "__main__":
    main()
