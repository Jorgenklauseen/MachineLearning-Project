"""Grad-CAM (Selvaraju et al., 2017) for the two CNNs on the original test images.

For each image and checkpoint:
  cam_share   = share of the Grad-CAM map that lies on the animal (trimap != 2)
  animal_area = share of the image that is animal
  delta_m     = cam_share - animal_area
delta_m > 0 means the map favours the animal more than its size alone would; a map spread
evenly over the image gives 0. The map explains the predicted class unless --target true.

Grad-CAM is implemented here directly from the paper (Eq. 1 and 2) instead of with a library,
so the map is not rescaled per image before the shares are computed."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data import DATA_DIR, SPLIT_SEED
from evaluate_cnn_background import ARCHITECTURE, find_checkpoints, load_model
from evaluate_svm_background import N_BOOT, PET_DIR, read_metadata
from train_cnn import BATCH_SIZE, eval_transform

ROOT = Path(__file__).resolve().parents[1]

# Layer whose activations and gradients Grad-CAM uses: the last convolutional stage.
TARGET_LAYER = {
    "pet_cnn": lambda m: m.features[-1][2],   # ReLU of the last block, before pooling: 28x28 at 224 px
    "resnet50": lambda m: m.layer4[-1],       # last bottleneck block: 7x7 at 224 px
    "resnet50_untrained": lambda m: m.layer4[-1],
}


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.activations = output
        output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(self, images, classes=None):
        """Returns (maps at input size, explained class, predicted class) for a batch."""
        logits = self.model(images)
        predicted = logits.argmax(dim=1)
        target = predicted if classes is None else classes
        self.model.zero_grad()
        logits.gather(1, target[:, None]).sum().backward()   # each image only affects its own score
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)            # alpha_k^c, Eq. 1
        maps = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # Eq. 2
        maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return maps[:, 0].detach().cpu(), target.cpu(), predicted.cpu()


def same_geometry(image, nearest=False):
    """Apply only the resize and crop steps of eval_transform, so a mask or a displayed image
    lines up pixel for pixel with what the model saw."""
    for t in eval_transform.transforms:
        if isinstance(t, transforms.Resize):
            image = transforms.Resize(t.size, interpolation=InterpolationMode.NEAREST if nearest
                                      else t.interpolation)(image)
        elif isinstance(t, transforms.CenterCrop):
            image = t(image)
    return image


def animal_mask(name):
    """Animal pixels (trimap 1 or 3), resized and cropped the same way as the image."""
    trimap = Image.open(PET_DIR / "annotations" / "trimaps" / f"{name}.png")
    return torch.from_numpy(np.asarray(same_geometry(trimap, nearest=True)) != 2)


def run(model_name, path, names, y, device, use_true_class, keep=()):
    """Per-image rows, and the maps of the images in `keep` (for the figure)."""
    model = load_model(model_name, path, device)
    cam = GradCAM(model, TARGET_LAYER[model_name](model))
    rows, maps = [], {}
    for start in range(0, len(names), BATCH_SIZE):
        batch = names[start:start + BATCH_SIZE]
        images = torch.stack([eval_transform(Image.open(PET_DIR / "images" / f"{n}.jpg").convert("RGB"))
                              for n in batch]).to(device)
        classes = torch.as_tensor(y[start:start + BATCH_SIZE], device=device) if use_true_class else None
        batch_maps, _, predicted = cam(images, classes)
        for i, name in enumerate(batch):
            m, mask = batch_maps[i], animal_mask(name)
            total = m.sum().item()
            share = (m * mask).sum().item() / total if total > 0 else float("nan")
            area = mask.float().mean().item()
            rows.append({"name": name, "true": int(y[start + i]), "pred": int(predicted[i]),
                         "correct": int(predicted[i] == y[start + i]), "animal_area": round(area, 4),
                         "cam_share": round(share, 4), "delta_m": round(share - area, 4)})
            if name in keep:
                maps[name] = m.numpy()
    return rows, maps


def bootstrap_mean_ci(values, rng):
    values = values[~np.isnan(values)]
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    return values.mean(), *np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


def save_figure(path, names, breeds, y, results, chosen):
    """Image with the animal outlined, and each model's Grad-CAM map with the same outline.
    Each map has its own colour scale (darker = more weight); the shares do not depend on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, quiet = "#191919", "#666666"
    blues = LinearSegmentedColormap.from_list(
        "blues", ["#ffffff", "#cce3fd", "#84b9f3", "#257de0", "#1757ac", "#0e3d7e"])
    fig, axes = plt.subplots(len(chosen), 1 + len(results), squeeze=False,
                             figsize=(2.6 * (1 + len(results)), 2.7 * len(chosen)))
    for row, i in zip(axes, chosen):
        name = names[i]
        mask = animal_mask(name).numpy()
        image = same_geometry(Image.open(PET_DIR / "images" / f"{name}.jpg").convert("RGB"))
        row[0].imshow(image)
        row[0].set_title(f"{breeds[int(y[i])]}", fontsize=8, color=ink)
        for ax, (model_name, (rows, maps)) in zip(row[1:], results.items()):
            r = next(r for r in rows if r["name"] == name)
            ax.imshow(maps[name], cmap=blues)
            ax.set_title(f"{model_name}: {breeds[r['pred']]}\n"
                         f"on animal {r['cam_share']:.0%}, area {r['animal_area']:.0%}",
                         fontsize=8, color=ink)
        for ax in row:
            ax.contour(mask, levels=[0.5], colors=quiet, linewidths=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "background_test",
                        help="folder with metadata.csv; decides which test images are used")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--models", nargs="+", default=["pet_cnn", "resnet50"], choices=sorted(TARGET_LAYER))
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted",
                        help="which class each map explains")
    parser.add_argument("--figure-images", type=int, default=6,
                        help="example images in the figure (seed 0 checkpoints); 0 = no figure")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, y, breeds = read_metadata(args.data)
    checkpoints = find_checkpoints(args.checkpoints, args.models)
    if not checkpoints:
        raise SystemExit(f"No checkpoints for {args.models} in {args.checkpoints}.")
    print(f"{len(names)} test images, device {device}, maps explain the {args.target} class")

    # Example images for the figure: drawn at random with a fixed seed, shown for each model's first seed
    chosen = sorted(np.random.default_rng(SPLIT_SEED).choice(
        len(names), size=min(args.figure_images, len(names)), replace=False))
    all_rows, summary, for_figure = [], [], {}
    for model_name, seed, path in checkpoints:
        first_seed = seed == min(s for m, s, _ in checkpoints if m == model_name)
        rows, maps = run(model_name, path, names, y, device, args.target == "true",
                         keep={names[i] for i in chosen} if first_seed else ())
        all_rows += [{"model": model_name, "seed": seed, **r} for r in rows]
        if first_seed:
            for_figure[model_name] = (rows, maps)

        rng = np.random.default_rng(SPLIT_SEED)
        delta = np.array([r["delta_m"] for r in rows], dtype=float)
        correct = np.array([r["correct"] for r in rows], dtype=bool)
        for group, keep in [("all", np.ones_like(correct)), ("correct", correct), ("wrong", ~correct)]:
            if keep.sum() == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(delta[keep], rng)
            summary.append({"model": model_name, "seed": seed, "images": group, "n": int(keep.sum()),
                            "delta_m": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                            "cam_share": round(np.nanmean([r["cam_share"] for r, k in zip(rows, keep) if k]), 4),
                            "animal_area": round(np.mean([r["animal_area"] for r, k in zip(rows, keep) if k]), 4),
                            "empty_maps": int(np.isnan(delta[keep]).sum())})
        for s in summary[-3:]:
            print(f"  {model_name} seed {seed} [{s['images']:<7}] n={s['n']:<5} delta_m={s['delta_m']:+.3f} "
                  f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  on animal {s['cam_share']:.1%}, "
                  f"area {s['animal_area']:.1%}, empty maps {s['empty_maps']}")

    suffix = "" if args.target == "predicted" else "_true_class"
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [(f"gradcam{suffix}_per_image.csv", all_rows), (f"gradcam{suffix}_summary.csv", summary)]:
        with open(args.out / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.out / filename}")

    if args.figure_images:
        save_figure(args.out / "figures" / f"gradcam{suffix}_examples.png", names, breeds, y,
                    for_figure, chosen)


if __name__ == "__main__":
    main()
