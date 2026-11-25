"""Learning curves from results/history/*.json (written by train_cnn.py): training and validation
loss per epoch, one panel per model, one line per seed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from summarize_results import AXIS, COLOURS, GRID, INK, LABELS, MODELS, QUIET

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    runs = {}
    for path in sorted((RESULTS_DIR / "history").glob("*.json")):
        log = json.loads(path.read_text())
        if not log.get("debug"):
            runs.setdefault(log["model"], []).append(log)
    models = [m for m in MODELS if m in runs]
    if not models:
        raise SystemExit("No training histories found. Run train_cnn.py first.")

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        for log in runs[model]:
            epochs = [h["epoch"] for h in log["history"]]
            ax.plot(epochs, [h["train_loss"] for h in log["history"]], color=COLOURS[model], linewidth=1.5)
            ax.plot(epochs, [h["val_loss"] for h in log["history"]], color=COLOURS[model],
                    linewidth=1.5, linestyle="--")
            ax.axvline(log["best_epoch"], color=GRID, linewidth=1, zorder=0)
        ax.set_title(f"{LABELS[model]} ({len(runs[model])} seeds)", fontsize=9, color=INK)
        ax.set_xlabel("Epoch", color=QUIET)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Cross-entropy loss", color=QUIET)
        ax.tick_params(colors=QUIET, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(AXIS)
        ax.plot([], [], color=QUIET, linewidth=1.5, label="training")
        ax.plot([], [], color=QUIET, linewidth=1.5, linestyle="--", label="validation")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    path = RESULTS_DIR / "figures" / "learning_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    for model in models:
        for log in runs[model]:
            print(f"  {model} seed {log['seed']}: best epoch {log['best_epoch']}, "
                  f"val accuracy {log['best_val_accuracy']:.3f}, {log['minutes']} min on {log['device']}")


if __name__ == "__main__":
    main()
