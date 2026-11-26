"""Learning curves: loss and accuracy per epoch for training and validation.

Reads the history files from train_cnn.py and saves one figure per model. With several seeds,
the line is the mean and the shaded band goes from the lowest to the highest seed.

Example:
    python -m src.experiments.learning_curves --models resnet50 pet_cnn --seeds 0 1 2
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import fig_dir, history_dir
from src.experiments.common_evaluation import CNN_MODELS, LABELS


def plot(ax, histories, metric):
    epochs = histories[0]["epoch"]
    for split, name, style in [("train", "Training", "--"), ("val", "Validation", "-")]:
        values = pd.concat([h[f"{split}_{metric}"] for h in histories], axis=1)
        line, = ax.plot(epochs, values.mean(axis=1), style, label=name)
        if len(histories) > 1:
            ax.fill_between(epochs, values.min(axis=1), values.max(axis=1), color=line.get_color(), alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=CNN_MODELS, default=["resnet50", "pet_cnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    for model in args.models:
        paths = [history_dir(model) / f"{model}_seed{seed}.csv" for seed in args.seeds]
        histories = [pd.read_csv(path) for path in paths if path.exists()]
        if not histories:
            print(f"No training history for {model}, skipped")
            continue

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        plot(ax_loss, histories, "loss")
        plot(ax_acc, histories, "accuracy")
        ax_acc.legend()
        fig.suptitle(f"{LABELS[model]}, {len(histories)} seed(s)")

        out = fig_dir(model) / f"learning_curves_{model}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
