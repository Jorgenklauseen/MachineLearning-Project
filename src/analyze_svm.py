import argparse
import csv
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from data import SPLIT_SEED, load_test
from train_svm import load_images, model_path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS_DIR / "figures"
N_TOP = 6          # most common confusions
N_VARIED = 12      # confusions with a different predicted breed in each row
ROWS_PER_FIG = 6   # rows per figure, so each figure fits on one page
PER_PAIR = 2       # misclassified images shown per confusion
N_BEST = 6         # breeds with highest recall shown with correct examples
PER_CORRECT = 3    # correctly classified images shown per breed


def plot_pairs(test, y, pred, pairs, filename, title):
    names = test.classes
    rng = np.random.default_rng(SPLIT_SEED)
    fig, axes = plt.subplots(len(pairs), PER_PAIR + 1, squeeze=False,
                             figsize=(9, 3.3 * len(pairs)))
    for row, (t, p) in enumerate(pairs):
        wrong = np.flatnonzero((y == t) & (pred == p))
        chosen = rng.choice(wrong, min(PER_PAIR, len(wrong)), replace=False)
        for col in range(PER_PAIR):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(test[int(chosen[col])][0])
                ax.set_title(f"True: {names[t]}\nPredicted: {names[p]}", fontsize=8)
        ax = axes[row, -1]
        ax.imshow(test[int(rng.choice(np.flatnonzero(y == p)))][0])
        ax.axis("off")
        ax.set_title(f"Example of\n{names[p]}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
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
    order = np.argsort(recall)[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["breed", "n_test", "predicted", "correct", "recall", "precision"])
        for i in order:
            writer.writerow([names[i], n_test[i], predicted[i], correct[i],
                             f"{recall[i]:.4f}", f"{precision[i]:.4f}"])
    print(f"Saved {path}")


def plot_correct(test, y, pred, classes, stats, filename, title):
    names = test.classes
    n_test, predicted, correct, recall, precision = stats
    rng = np.random.default_rng(SPLIT_SEED)
    fig, axes = plt.subplots(len(classes), PER_CORRECT, squeeze=False,
                             figsize=(9, 3.3 * len(classes)))
    for row, c in enumerate(classes):
        hits = np.flatnonzero((y == c) & (pred == c))
        chosen = rng.choice(hits, min(PER_CORRECT, len(hits)), replace=False)
        for col in range(PER_CORRECT):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(test[int(chosen[col])][0])
        axes[row, PER_CORRECT // 2].set_title(
            f"{names[c]}\nrecall {recall[c]:.2f} ({correct[c]}/{n_test[c]}), "
            f"precision {precision[c]:.2f} ({correct[c]}/{predicted[c]})",
            fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    print(f"Saved {FIG_DIR / filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", action="store_true",
                        help="analyze the HOG + color histogram model")
    color = parser.parse_args().color
    suffix = "_color" if color else ""
    variant = "HOG + color hist + SVM" if color else "HOG + SVM"

    model = joblib.load(model_path(color))
    test = load_test()
    names = test.classes
    X, y = load_images(test, range(len(test)), color)
    pred = model.predict(X)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Correct predictions: per-class table and examples from the best breeds
    stats = per_class_stats(y, pred, len(names))
    save_per_class(names, stats, RESULTS_DIR / f"svm{suffix}_per_class.csv")
    recall = stats[3]
    best = [int(c) for c in np.argsort(recall)[::-1][:N_BEST]]
    print("\nBreeds with highest recall:")
    for c in best:
        print(f"  {names[c]}: recall={recall[c]:.2f}, precision={stats[4][c]:.2f}, "
              f"predicted {stats[1][c]} times, {stats[2][c]} correct of {stats[0][c]}")
    plot_correct(test, y, pred, best, stats, f"svm{suffix}_correct_examples.png",
                 f"Breeds with highest recall, {variant} (test)")

    counts = confusion_matrix(y, pred)
    np.fill_diagonal(counts, 0)
    all_pairs = [np.unravel_index(i, counts.shape)
                 for i in np.argsort(counts, axis=None)[::-1]]

    varied, used = [], set()
    for t, p in all_pairs:
        if p not in used:
            varied.append((t, p))
            used.add(p)
        if len(varied) == N_VARIED:
            break

    groups = [("Most common confusions", f"svm{suffix}_error_examples", all_pairs[:N_TOP]),
              ("Confusions with different predicted breeds",
               f"svm{suffix}_error_examples_varied", varied)]

    for title, name, pairs in groups:
        print(f"\n{title}:")
        for t, p in pairs:
            print(f"  {names[t]} -> {names[p]}: {counts[t, p]} images")

        for start in range(0, len(pairs), ROWS_PER_FIG):
            part = start // ROWS_PER_FIG + 1
            plot_pairs(test, y, pred, pairs[start:start + ROWS_PER_FIG],
                       f"{name}_{part}.png", f"{title} ({part}), {variant} (test)")


if __name__ == "__main__":
    main()