"""Experiment 1: error analysis on the official test split, for one model (SVM or CNN).

Accuracy with 95 % CI and macro-F1, precision and recall per breed, the breeds with highest
recall and the most common confusions, with example figures.

Examples:
    python -m src.experiments.experiment1_errors --model svm_hog_color
    python -m src.experiments.experiment1_errors --model resnet50 --seed 0
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from src.data import FIG_DIR, RESULTS_DIR, SPLIT_SEED
from src.experiments.common_evaluation import (ALL_MODELS, LABELS, bootstrap_ci, load_model, predict,
                                              run_name, test_split, write_csv)

N_TOP = 6          # most common confusions
N_VARIED = 12      # confusions with a different predicted breed in each row
ROWS_PER_FIG = 6   # rows per figure, so each figure fits on one page
PER_PAIR = 2       # misclassified images shown per confusion
N_BEST = 6         # breeds with highest recall shown with correct examples
PER_CORRECT = 3    # correctly classified images shown per breed


def show(ax, path):
    ax.imshow(Image.open(path).convert("RGB"))
    ax.axis("off")


def plot_pairs(paths, names, y, pred, pairs, filename, title):
    rng = np.random.default_rng(SPLIT_SEED)
    fig, axes = plt.subplots(len(pairs), PER_PAIR + 1, squeeze=False,
                             figsize=(9, 3.3 * len(pairs)))
    for row, (t, p) in enumerate(pairs):
        wrong = np.flatnonzero((y == t) & (pred == p))
        chosen = rng.choice(wrong, min(PER_PAIR, len(wrong)), replace=False)
        for col in range(PER_PAIR):
            axes[row, col].axis("off")
            if col < len(chosen):
                show(axes[row, col], paths[chosen[col]])
                axes[row, col].set_title(f"True: {names[t]}\nPredicted: {names[p]}", fontsize=8)
        show(axes[row, -1], paths[rng.choice(np.flatnonzero(y == p))])
        axes[row, -1].set_title(f"Example of\n{names[p]}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / filename}")


def per_class_stats(y, pred, n_classes):
    """Per breed: test images, times predicted, correct, recall and precision."""
    cm = confusion_matrix(y, pred, labels=range(n_classes))
    correct = np.diag(cm)
    n_test = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = correct / np.maximum(n_test, 1)
    precision = correct / np.maximum(predicted, 1)
    return n_test, predicted, correct, recall, precision


def save_per_class(names, stats, path):
    n_test, predicted, correct, recall, precision = stats
    rows = [{"breed": names[i], "n_test": n_test[i], "predicted": predicted[i], "correct": correct[i],
             "recall": f"{recall[i]:.4f}", "precision": f"{precision[i]:.4f}"}
            for i in np.argsort(recall)[::-1]]
    write_csv(path, rows)


def plot_correct(paths, names, y, pred, classes, stats, filename, title):
    n_test, predicted, correct, recall, precision = stats
    rng = np.random.default_rng(SPLIT_SEED)
    fig, axes = plt.subplots(len(classes), PER_CORRECT, squeeze=False,
                             figsize=(9, 3.3 * len(classes)))
    for row, c in enumerate(classes):
        hits = np.flatnonzero((y == c) & (pred == c))
        chosen = rng.choice(hits, min(PER_CORRECT, len(hits)), replace=False)
        for col in range(PER_CORRECT):
            axes[row, col].axis("off")
            if col < len(chosen):
                show(axes[row, col], paths[chosen[col]])
        axes[row, PER_CORRECT // 2].set_title(
            f"{names[c]}\nrecall {recall[c]:.2f} ({correct[c]}/{n_test[c]}), "
            f"precision {precision[c]:.2f} ({correct[c]}/{predicted[c]})",
            fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--seed", type=int, default=0, help="training seed (CNNs only)")
    args = parser.parse_args()

    name = run_name(args.model, args.seed)
    label = LABELS[args.model]
    model = load_model(args.model, args.seed)
    paths, y, names = test_split()
    pred = predict(args.model, model, paths)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Overall result
    correct = (pred == y).astype(float)
    lo, hi = bootstrap_ci(correct, np.random.default_rng(SPLIT_SEED))
    f1 = f1_score(y, pred, average="macro", labels=range(len(names)), zero_division=0)
    print(f"\nTest split: n={len(y)}  acc={correct.mean():.1%} [{lo:.1%}, {hi:.1%}]  macro-F1={f1:.3f}")
    write_csv(RESULTS_DIR / f"{name}_test_summary.csv", [{
        "model": name, "n": len(y), "accuracy": correct.mean(), "ci_low": lo, "ci_high": hi, "macro_f1": f1,
    }])

    # Correct predictions: per-class table and examples from the best breeds
    stats = per_class_stats(y, pred, len(names))
    save_per_class(names, stats, RESULTS_DIR / f"{name}_per_class.csv")
    recall = stats[3]
    best = [int(c) for c in np.argsort(recall)[::-1][:N_BEST]]
    print("\nBreeds with highest recall:")
    for c in best:
        print(f"  {names[c]}: recall={recall[c]:.2f}, precision={stats[4][c]:.2f}, "
              f"predicted {stats[1][c]} times, {stats[2][c]} correct of {stats[0][c]}")
    plot_correct(paths, names, y, pred, best, stats, f"{name}_correct_examples.png",
                 f"Breeds with highest recall, {label} (test)")

    # Wrong predictions: the most common confusions
    counts = confusion_matrix(y, pred, labels=range(len(names)))
    np.fill_diagonal(counts, 0)
    all_pairs = [np.unravel_index(i, counts.shape) for i in np.argsort(counts, axis=None)[::-1]]
    all_pairs = [(t, p) for t, p in all_pairs if counts[t, p] > 0]

    varied, used = [], set()
    for t, p in all_pairs:
        if p not in used:
            varied.append((t, p))
            used.add(p)
        if len(varied) == N_VARIED:
            break

    groups = [("Most common confusions", f"{name}_error_examples", all_pairs[:N_TOP]),
              ("Confusions with different predicted breeds", f"{name}_error_examples_varied", varied)]
    for title, filename, pairs in groups:
        print(f"\n{title}:")
        for t, p in pairs:
            print(f"  {names[t]} -> {names[p]}: {counts[t, p]} images")
        for start in range(0, len(pairs), ROWS_PER_FIG):
            part = start // ROWS_PER_FIG + 1
            plot_pairs(paths, names, y, pred, pairs[start:start + ROWS_PER_FIG],
                       f"{filename}_{part}.png", f"{title} ({part}), {label} (test)")


if __name__ == "__main__":
    main()
