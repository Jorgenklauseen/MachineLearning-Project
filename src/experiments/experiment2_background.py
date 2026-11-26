"""Experiment 2: does the model use the pet or the background?

Evaluates every trained model (SVMs and CNNs) on the three variants made by
make_background_test.py (original, pet_only, bg_only) and writes one shared table.
Models without a checkpoint are skipped. With several seeds, each CNN also gets rows with the
mean, std and median over the seeds for every variant.

Examples:
    python -m src.experiments.experiment2_background                  # all models, CNN seed 0
    python -m src.experiments.experiment2_background --seeds 0 1 2    # CNNs with three training seeds
    python -m src.experiments.experiment2_background --models svm_hog_color resnet50
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

from src.data import COMPARISON_DIR, DATA_DIR, SPLIT_SEED
from src.experiments.common_evaluation import (ALL_MODELS, SVM_MODELS, bootstrap_ci, checkpoint_path,
                                              load_model, predict, run_name, seed_summary, test_split,
                                              write_csv)

PET_DIR = DATA_DIR / "oxford-iiit-pet"
VARIANTS = ["original", "pet_only", "bg_only"]
GREY = np.array([124, 116, 104], dtype=np.uint8)  # same grey as make_background_test.py
BG_BINS = [0, 0.10, 0.25, 0.50, 1.01]


def read_metadata(folder):
    with open(folder / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["name"] for r in rows], np.array([int(r["class_id"]) for r in rows])


def background_share(folder, names):
    """Share of each bg_only image that is still background (not covered by grey)."""
    shares = []
    for name in names:
        img = np.asarray(Image.open(folder / "bg_only" / f"{name}.png").convert("RGB"))
        shares.append(1 - np.all(img == GREY, axis=-1).mean())
    return np.array(shares)


def summarize(model, seed, variant, y, pred, pred_orig, breeds, rng):
    correct = (pred == y).astype(float)
    acc, acc_orig = correct.mean(), (pred_orig == y).mean()
    lo, hi = bootstrap_ci(correct, rng)
    counts = np.bincount(pred, minlength=len(breeds))
    return {
        "model": model, "seed": "" if seed is None else seed, "variant": variant, "n": len(y),
        "accuracy": acc, "ci_low": lo, "ci_high": hi,
        "macro_f1": f1_score(y, pred, average="macro", labels=range(len(breeds)), zero_division=0),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="training seeds (CNNs only)")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "background_test")
    parser.add_argument("--out", type=Path, default=COMPARISON_DIR)
    args = parser.parse_args()

    if not (args.data / "metadata.csv").exists():
        raise SystemExit(f"{args.data} not found. Run make_background_test.py first.")
    names, y = read_metadata(args.data)
    breeds = test_split()[2]
    bg_share = background_share(args.data, names)
    bg_bin = np.digitize(bg_share, BG_BINS) - 1
    print(f"{len(names)} test images. Background left in bg_only: "
          f"median {np.median(bg_share):.0%}, {(bg_share < 0.10).sum()} images under 10 %.")
    print(f"Chance level: {1 / len(breeds):.1%}")

    summary, predictions = [], []
    for model_name in args.models:
        model_rows = []
        for seed in ([None] if model_name in SVM_MODELS else args.seeds):
            name = run_name(model_name, seed)
            if not checkpoint_path(model_name, seed).exists():
                print(f"\nSkipping {name}: no checkpoint")
                continue
            print(f"\n{name}")
            model = load_model(model_name, seed)
            preds = {v: predict(model_name, model, [args.data / v / f"{n}.png" for n in names])
                     for v in VARIANTS}

            pred_jpg = predict(model_name, model, [PET_DIR / "images" / f"{n}.jpg" for n in names])
            print(f"  check: original PNG vs JPG give the same prediction for "
                  f"{(pred_jpg == preds['original']).mean():.1%} of images")

            rng = np.random.default_rng(SPLIT_SEED)
            rows = [summarize(model_name, seed, v, y, preds[v], preds["original"], breeds, rng) for v in VARIANTS]
            for b in range(len(BG_BINS) - 1):
                m = bg_bin == b
                if m.any():
                    label = f"bg_only [{BG_BINS[b]:.0%}-{min(BG_BINS[b + 1], 1):.0%} bg]"
                    rows.append(summarize(model_name, seed, label, y[m], preds["bg_only"][m],
                                          preds["original"][m], breeds, rng))
            for r in rows:
                print_row(r)
            model_rows += rows

            for i, n in enumerate(names):
                predictions.append({"model": name, "name": n, "true": breeds[y[i]],
                                    "bg_share": round(float(bg_share[i]), 4),
                                    **{f"pred_{v}": breeds[preds[v][i]] for v in VARIANTS}})

        # Mean, std and median over the seeds, per variant
        summary += model_rows
        n_seeds = len({r["seed"] for r in model_rows})
        if n_seeds > 1:
            print(f"\n{model_name}, mean +/- std over {n_seeds} seeds:")
            for variant in dict.fromkeys(r["variant"] for r in model_rows):
                stats = seed_summary([r for r in model_rows if r["variant"] == variant],
                                     ["accuracy", "macro_f1", "rel_change", "changed_answers"])
                print(f"  {variant:<22} acc={stats[0]['accuracy']:6.1%} +/- {stats[1]['accuracy']:4.1%}  "
                      f"changed={stats[0]['changed_answers']:5.1%}")
                summary += stats

    if not summary:
        raise SystemExit("\nNo trained models found, nothing to evaluate.")
    write_csv(args.out / "experiment2_summary.csv", summary)
    write_csv(args.out / "experiment2_predictions.csv", predictions)


if __name__ == "__main__":
    main()
