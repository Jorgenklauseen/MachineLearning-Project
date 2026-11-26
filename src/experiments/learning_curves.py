"""Learning curves: loss and accuracy per epoch, for training and validation.

Reads the history files that train_cnn.py writes (results/<model>/training/<model>_seed<n>.csv) and
saves one figure per model. With several seeds, the line is the mean over the seeds and the
shaded band shows the lowest and highest seed.

Examples:
    python -m src.experiments.learning_curves --models resnet50 pet_cnn --seeds 0 1 2
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from src.data import fig_dir, history_dir
from src.experiments.common_evaluation import CNN_MODELS, LABELS

# Training and validation: two colours, and dashed vs. solid so they also differ in black and white
SPLITS = {"train": ("Training", "#2a78d6", "--"), "val": ("Validation", "#eb6834", "-")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def load_histories(model, seeds):
    histories = []
    for seed in seeds:
        path = history_dir(model) / f"{model}_seed{seed}.csv"
        if path.exists():
            histories.append(pd.read_csv(path))
        else:
            print(f"  {path.name} not found, skipped")
    if histories:  # all seeds should have the same number of epochs; cut to the shortest just in case
        n = min(len(h) for h in histories)
        histories = [h.iloc[:n] for h in histories]
    return histories


def plot_metric(ax, histories, metric, title, percent):
    epochs = histories[0]["epoch"]
    for split, (name, color, style) in SPLITS.items():
        column = f"{split}_{metric}"
        if column not in histories[0]:
            continue  # history from before train_accuracy/val_loss were logged
        values = np.stack([h[column].to_numpy() for h in histories])
        ax.plot(epochs, values.mean(axis=0), color=color, linestyle=style, linewidth=2, label=name)
        if len(histories) > 1:
            ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=0.15, linewidth=0)

    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Epoch", color=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=CNN_MODELS, default=["resnet50", "pet_cnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    for model in args.models:
        histories = load_histories(model, args.seeds)
        if not histories:
            print(f"Skipping {model}: no training history. Run train_cnn.py first.")
            continue

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        plot_metric(ax_loss, histories, "loss", "Loss", percent=False)
        plot_metric(ax_acc, histories, "accuracy", "Accuracy", percent=True)

        subtitle = (f"Mean of {len(histories)} seeds, shaded band = lowest to highest seed"
                    if len(histories) > 1 else "1 seed")
        fig.suptitle(f"{LABELS[model]}: learning curves", color=INK, fontsize=12, x=0.01, y=0.99, ha="left")
        fig.text(0.01, 0.925, subtitle, color=MUTED, fontsize=9, ha="left")
        ax_acc.legend(loc="lower right", frameon=False, labelcolor=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out = fig_dir(model) / f"learning_curves_{model}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
