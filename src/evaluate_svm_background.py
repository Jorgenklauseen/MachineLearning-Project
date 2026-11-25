import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

from data import DATA_DIR, SPLIT_SEED
from models import NUM_CLASSES, SVM_IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]
PET_DIR = DATA_DIR / "oxford-iiit-pet"
VARIANTS = ["original", "pet_only", "bg_only"]
MODELS = {"svm_hog": False, "svm_hog_color": True} 
GREY = np.array([124, 116, 104], dtype=np.uint8)      
CHANCE = 1 / NUM_CLASSES
N_BOOT = 2000
BG_BINS = [0, 0.10, 0.25, 0.50, 1.01]


def read_metadata(folder):
    with open(folder / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    y = np.array([int(r["class_id"]) for r in rows])
    breeds = {}
    for line in (PET_DIR / "annotations" / "test.txt").read_text().splitlines():
        name, class_id, _, _ = line.split()
        breeds[int(class_id) - 1] = name.rsplit("_", 1)[0]
    return names, y, breeds


def load_images(paths, color):
    """Same preprocessing as load_images in train_svm.py: 128x128, greyscale or RGB."""
    mode = "RGB" if color else "L"
    return np.array([
        np.asarray(Image.open(p).convert("RGB").convert(mode).resize((SVM_IMAGE_SIZE, SVM_IMAGE_SIZE)))
        for p in paths
    ])


def background_share(folder, names):
    """Share of each bg_only image that is still background (not covered by the fill).
    Read from metadata.csv when it has the column (always with the noise fill)."""
    with open(folder / "metadata.csv", newline="", encoding="utf-8") as f:
        saved = {r["name"]: r.get("bg_share") for r in csv.DictReader(f)}
    if all(saved.get(n) not in (None, "") for n in names):
        return np.array([float(saved[n]) for n in names])
    shares = []
    for name in names:
        img = np.asarray(Image.open(folder / "bg_only" / f"{name}.png").convert("RGB"))
        shares.append(1 - np.all(img == GREY, axis=-1).mean())
    return np.array(shares)



def bootstrap_ci(correct, rng):
    """95 % confidence interval for accuracy, by resampling the test images."""
    idx = rng.integers(0, len(correct), size=(N_BOOT, len(correct)))
    return np.percentile(correct[idx].mean(axis=1), [2.5, 97.5])


def summarize(model, variant, y, pred, pred_orig, breeds, rng):
    correct = (pred == y).astype(float)
    acc, acc_orig = correct.mean(), (pred_orig == y).mean()
    lo, hi = bootstrap_ci(correct, rng)
    counts = np.bincount(pred, minlength=NUM_CLASSES)
    return {
        "model": model, "variant": variant, "n": len(y),
        "accuracy": acc, "ci_low": lo, "ci_high": hi,
        "macro_f1": f1_score(y, pred, average="macro", labels=range(NUM_CLASSES), zero_division=0),
        "rel_change": (acc - acc_orig) / acc_orig if acc_orig > 0 else np.nan,
        "changed_answers": (pred != pred_orig).mean(),
        "top_pred": breeds[int(counts.argmax())],       
        "top_pred_share": counts.max() / len(pred),
    }


def print_row(r):
    print(f"  {r['variant']:<22} n={r['n']:<5} acc={r['accuracy']:6.1%} "
          f"[{r['ci_low']:5.1%}, {r['ci_high']:5.1%}]  F1={r['macro_f1']:.3f}  "
          f"rel={r['rel_change']:+6.1%}  changed={r['changed_answers']:5.1%}  "
          f"top={r['top_pred']} ({r['top_pred_share']:.0%})")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "background_test")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", default="",
                        help="added to the output file names, e.g. _noise for the noise-filled images")
    args = parser.parse_args()

    names, y, breeds = read_metadata(args.data)
    bg_share = background_share(args.data, names)
    bg_bin = np.digitize(bg_share, BG_BINS) - 1
    print(f"{len(names)} test images. Background left in bg_only: "
          f"median {np.median(bg_share):.0%}, {(bg_share < 0.10).sum()} images under 10 %.")
    print(f"Chance level: {CHANCE:.1%}")

    summary, predictions = [], []
    for model_name, color in MODELS.items():
        path = args.checkpoints / f"{model_name}.joblib"
        if not path.exists():
            print(f"\nSkipping {model_name}: {path} not found. Run train_svm.py{' --color' if color else ''}.")
            continue
        model = joblib.load(path)
        print(f"\n{model_name}")

        preds = {v: model.predict(load_images([args.data / v / f"{n}.png" for n in names], color))
                 for v in VARIANTS}

        pred_jpg = model.predict(load_images([PET_DIR / "images" / f"{n}.jpg" for n in names], color))
        same = (pred_jpg == preds["original"]).mean()
        print(f"  check: original PNG vs JPG give the same prediction for {same:.1%} of images")

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
        summary += rows

        for i, name in enumerate(names):
            predictions.append({"model": model_name, "name": name, "true": breeds[int(y[i])],
                                "bg_share": round(float(bg_share[i]), 4),
                                **{f"pred_{v}": breeds[int(preds[v][i])] for v in VARIANTS}})

    if not summary:
        raise SystemExit("\nNo SVM models found, nothing to evaluate.")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / f"experiment2_svm{args.tag}_summary.csv", summary)
    write_csv(args.out / f"experiment2_svm{args.tag}_predictions.csv", predictions)


if __name__ == "__main__":
    main()
