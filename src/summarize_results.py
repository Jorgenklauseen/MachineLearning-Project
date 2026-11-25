"""Combine the experiment 2 predictions of all models into the main results.

Reads results/experiment2_svm{tag}_predictions.csv and results/experiment2_cnn{tag}_predictions.csv
and writes, for each tag (grey fill "" and noise fill "_noise"):
  main_table{tag}.csv / .tex   accuracy per model and test version, and the relative accuracy
                               R = accuracy on the variant / accuracy on the original images
  model_comparison{tag}.csv    paired differences in R between models
  figures/accuracy_by_variant{tag}.png
CNN numbers are averaged over seeds. Confidence intervals: 95 % bootstrap over test images
(for CNNs, each image's correctness is first averaged over the seeds). If
results/gradcam_per_image.csv exists, it also links Grad-CAM to the accuracy changes."""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from data import SPLIT_SEED
from evaluate_svm_background import CHANCE, N_BOOT, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["svm_hog", "svm_hog_color", "pet_cnn", "resnet50_untrained", "resnet50"]   # fixed order
LABELS = {"svm_hog": "SVM (HOG)", "svm_hog_color": "SVM (HOG + colour)",
          "pet_cnn": "CNN from scratch", "resnet50_untrained": "ResNet-50 from scratch",
          "resnet50": "ResNet-50 pretrained"}
VARIANT_LABELS = {"original": "Original", "pet_only": "Animal only", "bg_only": "Background only"}
# Colour follows the model, whichever models are present (validated categorical palette)
COLOURS = {"svm_hog": "#257de0", "svm_hog_color": "#f47230", "pet_cnn": "#00b894",
           "resnet50": "#f2a600", "resnet50_untrained": "#ee7fac"}
INK, QUIET, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"


def load_correctness(results, tag):
    """{model: {variant: array (seeds, images)}} and the image names, aligned across models."""
    rows = []
    for kind in ["svm", "cnn"]:
        path = results / f"experiment2_{kind}{tag}_predictions.csv"
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
    if not rows:
        return {}, []
    names = sorted({r["name"] for r in rows})
    position = {n: i for i, n in enumerate(names)}
    data = {}
    for model in [m for m in MODELS if any(r["model"] == m for r in rows)]:
        model_rows = [r for r in rows if r["model"] == model]
        seeds = sorted({int(r.get("seed") or 0) for r in model_rows})
        data[model] = {v: np.full((len(seeds), len(names)), np.nan) for v in VARIANTS}
        data[model]["seeds"] = seeds
        for r in model_rows:
            s = seeds.index(int(r.get("seed") or 0))
            for v in VARIANTS:
                data[model][v][s, position[r["name"]]] = r[f"pred_{v}"] == r["true"]
    return data, names


def bootstrap_indices(n, rng):
    return rng.integers(0, n, size=(N_BOOT, n))


def summarize_model(correct, idx):
    """Accuracy and R with seed spread and bootstrap CIs, for one model."""
    out = {"seeds": correct["original"].shape[0]}
    per_seed = {v: correct[v].mean(axis=1) for v in VARIANTS}            # accuracy per seed
    per_image = {v: correct[v].mean(axis=0) for v in VARIANTS}           # averaged over seeds
    boot = {v: per_image[v][idx].mean(axis=1) for v in VARIANTS}
    for v in VARIANTS:
        out[f"acc_{v}"] = per_image[v].mean()
        out[f"acc_{v}_seed_std"] = per_seed[v].std(ddof=1) if len(per_seed[v]) > 1 else np.nan
        out[f"acc_{v}_ci_low"], out[f"acc_{v}_ci_high"] = np.percentile(boot[v], [2.5, 97.5])
    for v in ["pet_only", "bg_only"]:
        out[f"R_{v}"] = per_image[v].mean() / per_image["original"].mean()
        r_seed = per_seed[v] / per_seed["original"]
        out[f"R_{v}_seed_std"] = r_seed.std(ddof=1) if len(r_seed) > 1 else np.nan
        out[f"R_{v}_ci_low"], out[f"R_{v}_ci_high"] = np.percentile(boot[v] / boot["original"], [2.5, 97.5])
    out["bg_only_above_chance"] = bool(out["acc_bg_only_ci_low"] > CHANCE)
    return out, boot


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print(f"Saved {path}")


def latex_table(path, table, tag):
    def acc(r, v):
        std = r[f"acc_{v}_seed_std"]
        return f"{100 * r[f'acc_{v}']:.1f}" + ("" if np.isnan(std) else f" $\\pm$ {100 * std:.1f}")

    def ratio(r, v):
        return f"{r[f'R_{v}']:.2f} [{r[f'R_{v}_ci_low']:.2f}, {r[f'R_{v}_ci_high']:.2f}]"

    fill = "noise" if "noise" in tag else "grey"
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{2}{c}{Relative accuracy $R$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Model & Original & Animal only & Background only & Animal only & Background only \\",
             r"\midrule"]
    for r in table:
        lines.append(f"{LABELS[r['model']]} & {acc(r, 'original')} & {acc(r, 'pet_only')} & "
                     f"{acc(r, 'bg_only')} & {ratio(r, 'pet_only')} & {ratio(r, 'bg_only')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Test accuracy on the original, animal-only and background-only images "
              rf"({fill} fill). CNN accuracies are means $\pm$ standard deviation over seeds. "
              rf"$R$ is the accuracy on the variant divided by the accuracy on the original images, "
              rf"with 95\,\% bootstrap confidence intervals over test images. Chance level is "
              rf"{100 * CHANCE:.1f}\,\%.}}",
              rf"\label{{tab:main_results{tag}}}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def plot(path, table, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(table)
    width = 0.8 / n
    x = np.arange(len(VARIANTS))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, r in enumerate(table):
        heights = [100 * r[f"acc_{v}"] for v in VARIANTS]
        low = [100 * (r[f"acc_{v}"] - r[f"acc_{v}_ci_low"]) for v in VARIANTS]
        high = [100 * (r[f"acc_{v}_ci_high"] - r[f"acc_{v}"]) for v in VARIANTS]
        ax.bar(x - 0.4 + width * (i + 0.5), heights, width, color=COLOURS[r["model"]],
               edgecolor="white", linewidth=1.5, label=LABELS[r["model"]], zorder=2)
        ax.errorbar(x - 0.4 + width * (i + 0.5), heights, yerr=[low, high], fmt="none",
                    ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    ax.axhline(100 * CHANCE, color=QUIET, linestyle="--", linewidth=1, zorder=1)
    ax.text(1.01, 100 * CHANCE, f"chance\n({100 * CHANCE:.1f} %)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8, color=QUIET, clip_on=False)
    ax.set_xticks(x, [VARIANT_LABELS[v] for v in VARIANTS], color=INK)
    ax.set_ylabel("Test accuracy (%)", color=QUIET)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=QUIET, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(frameon=False, ncol=n if n <= 4 else 3, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              fontsize=8, labelcolor=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def gradcam_link(results, correct, names):
    """Among images a CNN gets right on the original: mean delta_m when it is still right on
    the animal-only image versus when it is not (Grad-CAM for the same seed)."""
    path = results / "gradcam_per_image.csv"
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        cam = list(csv.DictReader(f))
    position = {n: i for i, n in enumerate(names)}
    rows = []
    for model in sorted({r["model"] for r in cam}):
        if model not in correct:
            continue
        seeds = sorted({int(r["seed"]) for r in cam if r["model"] == model} & set(correct[model]["seeds"]))
        for seed in seeds:
            s = correct[model]["seeds"].index(seed)   # row of this seed in the prediction arrays
            groups = {"still correct on animal only": [], "wrong on animal only": []}
            for r in cam:
                if r["model"] != model or int(r["seed"]) != seed or r["delta_m"] in ("", "nan"):
                    continue
                i = position.get(r["name"])
                if i is None or correct[model]["original"][s, i] != 1:
                    continue
                key = "still correct on animal only" if correct[model]["pet_only"][s, i] == 1 \
                    else "wrong on animal only"
                groups[key].append(float(r["delta_m"]))
            for group, values in groups.items():
                rows.append({"model": model, "seed": seed, "images": group, "n": len(values),
                             "mean_delta_m": float(np.mean(values)) if values else np.nan})
    if rows:
        for r in rows:
            print(f"  {r['model']} seed {r['seed']}, {r['images']:<29} n={r['n']:<5} "
                  f"mean delta_m={r['mean_delta_m']:+.3f}")
        write_csv(results / "gradcam_vs_accuracy.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tags", nargs="+", default=["", "_noise"])
    args = parser.parse_args()

    for tag in args.tags:
        correct, names = load_correctness(args.results, tag)
        if not correct:
            print(f"No predictions for tag '{tag}', skipping")
            continue
        print(f"\n=== {'grey' if not tag else tag.strip('_')} fill: {len(names)} test images ===")
        idx = bootstrap_indices(len(names), np.random.default_rng(SPLIT_SEED))
        table, boots = [], {}
        for model in correct:
            summary, boots[model] = summarize_model(correct[model], idx)
            table.append({"model": model, **summary})
            print(f"  {LABELS[model]:<24} seeds={summary['seeds']}  "
                  + "  ".join(f"{VARIANT_LABELS[v]}={summary[f'acc_{v}']:.1%}" for v in VARIANTS)
                  + f"  R_animal={summary['R_pet_only']:.2f}  R_background={summary['R_bg_only']:.2f}"
                  + f"  background above chance: {summary['bg_only_above_chance']}")
        write_csv(args.results / f"main_table{tag}.csv", table)
        latex_table(args.results / f"main_table{tag}.tex", table, tag)
        plot(args.results / "figures" / f"accuracy_by_variant{tag}.png", table, tag)

        comparisons = []   # same bootstrap resamples for both models, so the difference is paired
        for a, b in combinations(correct, 2):
            for v in ["pet_only", "bg_only"]:
                diff = boots[a][v] / boots[a]["original"] - boots[b][v] / boots[b]["original"]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                point = next(r for r in table if r["model"] == a)[f"R_{v}"] - \
                    next(r for r in table if r["model"] == b)[f"R_{v}"]
                comparisons.append({"model_a": a, "model_b": b, "variant": v, "R_a_minus_R_b": point,
                                    "ci_low": lo, "ci_high": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
        write_csv(args.results / f"model_comparison{tag}.csv", comparisons)

        if not tag:
            gradcam_link(args.results, correct, names)


if __name__ == "__main__":
    main()
