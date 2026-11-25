"""Grad-CAM heatmaps: which parts of a test image the model bases its prediction on.

Examples:
    python src/gradcam.py --model resnet50
    python src/gradcam.py --model pet_cnn --wrong     # only misclassified images
    python src/gradcam.py --variant bg_only           # images from make_background_test.py
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from data import DATA_DIR, SPLIT_SEED
from models import build_model
from train_cnn import CHECKPOINT_DIR, IMAGE_SIZE, ROOT, eval_transform, load_checkpoint

# Same resize and crop as eval_transform, but without normalisation, for display
view_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(IMAGE_SIZE)])
BG_DIR = DATA_DIR / "background_test"


def image_list(variant, test):
    """Image paths and labels: the official test split, or a folder made by make_background_test.py."""
    if variant == "test":
        return test._images, test._labels
    with open(BG_DIR / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [BG_DIR / variant / f"{r['name']}.png" for r in rows], [int(r["class_id"]) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=["test", "original", "pet_only", "bg_only"], default="test")
    parser.add_argument("--wrong", action="store_true", help="show only misclassified images")
    parser.add_argument("--n", type=int, default=8, help="number of images")
    args = parser.parse_args()

    run_name = f"{args.model}_seed{args.seed}"
    state, _ = load_checkpoint(CHECKPOINT_DIR / f"{run_name}.pt")
    model = build_model(args.model)
    model.load_state_dict(state)
    model.eval()

    # Last convolutional layer: layer4 in ResNet, the last conv block in PetCNN
    layer = model.layer4[-1] if args.model.startswith("resnet50") else model.features[-1]

    test = OxfordIIITPet(DATA_DIR, split="test", target_types="category")
    paths, labels = image_list(args.variant, test)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(paths))

    cols = 3 if args.wrong else 2
    fig, axes = plt.subplots(args.n, cols, figsize=(3 * cols, 3 * args.n), squeeze=False)
    row = 0
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for i in order:
            img, true = Image.open(paths[i]).convert("RGB"), int(labels[i])
            x = eval_transform(img).unsqueeze(0)
            with torch.no_grad():
                pred = int(model(x).argmax())
            if args.wrong and pred == true:
                continue

            rgb = np.asarray(view_transform(img), dtype=np.float32) / 255
            panels = [(rgb, f"True: {test.classes[true]}"),
                      (show_cam_on_image(rgb, cam(x, [ClassifierOutputTarget(pred)])[0], use_rgb=True),
                       f"Predicted: {test.classes[pred]}")]
            if args.wrong:
                panels.append((show_cam_on_image(rgb, cam(x, [ClassifierOutputTarget(true)])[0], use_rgb=True),
                               f"Heatmap for true: {test.classes[true]}"))

            for ax, (image, title) in zip(axes[row], panels):
                ax.imshow(image)
                ax.set_title(title, fontsize=8, color="black" if pred == true else "red")
                ax.axis("off")
            row += 1
            if row == args.n:
                break

    fig.suptitle(f"Grad-CAM, {run_name}, {args.variant}" + (" (misclassified)" if args.wrong else ""))
    fig.tight_layout()
    out = ROOT / "results" / "figures" / f"gradcam_{run_name}_{args.variant}{'_wrong' if args.wrong else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
