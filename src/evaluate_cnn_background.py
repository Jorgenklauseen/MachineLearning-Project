"""Evaluate the trained CNNs (every seed found in checkpoints/) on the original, pet_only and
bg_only test images made by make_background_test.py. Same measures and output format as
evaluate_svm_background.py, plus a seed column."""
import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import DATA_DIR, SPLIT_SEED
from evaluate_svm_background import (BG_BINS, CHANCE, PET_DIR, VARIANTS, background_share,
                                     print_row, read_metadata, summarize, write_csv)
from models import build_model
from train_cnn import BATCH_SIZE, eval_transform

ROOT = Path(__file__).resolve().parents[1]
# Architecture to build before loading a checkpoint. The weights come from the checkpoint,
# so the ImageNet weights are not downloaded again for resnet50.
ARCHITECTURE = {"pet_cnn": "pet_cnn", "resnet50": "resnet50_untrained",
                "resnet50_untrained": "resnet50_untrained"}


class ImageFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_transform(Image.open(self.paths[i]).convert("RGB"))


def load_model(name, path, device):
    model = build_model(ARCHITECTURE[name])
    checkpoint = torch.load(path, map_location=device)
    if "model_state" in checkpoint:          # checkpoints saved with extra metadata
        checkpoint = checkpoint["model_state"]
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, paths, device, num_workers):
    loader = DataLoader(ImageFiles(paths), batch_size=BATCH_SIZE, num_workers=num_workers)
    return torch.cat([model(x.to(device)).argmax(dim=1).cpu() for x in loader]).numpy()


def find_checkpoints(folder, models):
    """(model, seed, path) for every checkpoint named <model>_seed<n>.pt."""
    found = []
    for name in models:
        for path in sorted(folder.glob(f"{name}_seed*.pt")):
            match = re.fullmatch(rf"{name}_seed(\d+)\.pt", path.name)
            if match:
                found.append((name, int(match.group(1)), path))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "background_test")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", default="",
                        help="added to the output file names, e.g. _noise for the noise-filled images")
    parser.add_argument("--models", nargs="+", default=["pet_cnn", "resnet50"],
                        choices=sorted(ARCHITECTURE))
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, y, breeds = read_metadata(args.data)
    bg_share = background_share(args.data, names)
    bg_bin = np.digitize(bg_share, BG_BINS) - 1
    print(f"{len(names)} test images, device {device}. Chance level: {CHANCE:.1%}")

    checkpoints = find_checkpoints(args.checkpoints, args.models)
    if not checkpoints:
        raise SystemExit(f"No checkpoints for {args.models} in {args.checkpoints}. Run train_cnn.py first.")

    summary, predictions = [], []
    for model_name, seed, path in checkpoints:
        model = load_model(model_name, path, device)
        print(f"\n{model_name}, seed {seed}")

        preds = {v: predict(model, [args.data / v / f"{n}.png" for n in names], device, args.num_workers)
                 for v in VARIANTS}
        pred_jpg = predict(model, [PET_DIR / "images" / f"{n}.jpg" for n in names], device, args.num_workers)
        print(f"  check: original PNG vs JPG give the same prediction for "
              f"{(pred_jpg == preds['original']).mean():.1%} of images")

        rng = np.random.default_rng(SPLIT_SEED)
        rows = [summarize(model_name, v, y, preds[v], preds["original"], breeds, rng) for v in VARIANTS]
        for b in range(len(BG_BINS) - 1):
            m = bg_bin == b
            if m.any():
                label = f"bg_only [{BG_BINS[b]:.0%}-{min(BG_BINS[b + 1], 1):.0%} bg]"
                rows.append(summarize(model_name, label, y[m], preds["bg_only"][m],
                                      preds["original"][m], breeds, rng))
        for r in rows:
            print_row(r)
        summary += [{"model": r.pop("model"), "seed": seed, **r} for r in rows]

        for i, name in enumerate(names):
            predictions.append({"model": model_name, "seed": seed, "name": name,
                                "true": breeds[int(y[i])], "bg_share": round(float(bg_share[i]), 4),
                                **{f"pred_{v}": breeds[int(preds[v][i])] for v in VARIANTS}})

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / f"experiment2_cnn{args.tag}_summary.csv", summary)
    write_csv(args.out / f"experiment2_cnn{args.tag}_predictions.csv", predictions)


if __name__ == "__main__":
    main()
