import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from data import SPLIT_SEED, load_test
from train_svm import load_images, model_path

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
N_TOP = 6          # most common confusions
N_VARIED = 12      # confusions with a different predicted breed in each row
ROWS_PER_FIG = 6   # rows per figure, so each figure fits on one page
PER_PAIR = 2       # misclassified images shown per confusion


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