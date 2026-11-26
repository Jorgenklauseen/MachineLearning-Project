"""Grad-CAM attention on the animal: how much of each heatmap lies on the pet.

For every test image with a pet in its trimap (the same 3,662 images as experiment 2), the Grad-CAM
heatmap for the predicted breed is compared with the pet mask from the trimap:

    share = sum(cam * mask) / sum(cam)     share of the activation that lies on the pet
    area  = mean(mask)                     share of the image the pet covers; a flat heatmap gets share = area
    peak_on_pet                            whether the strongest point of the heatmap lies on the pet

The mask is the pet plus the trimap's border zone, cropped exactly like the model input
(resize to 256x256, centre crop 224). Heatmaps come from the last conv block (ResNet-50: 7x7,
PetCNN: 14x14) and are upsampled to 224x224, so they are coarse and spill over the pet's edges.

Each model is evaluated with one training seed: by default the seed with median test accuracy,
the same one the figures in experiment 1 use. If experiment 2 has been run, each image is also linked
to whether its prediction changed when the background was removed (pet_only).

If the output files already exist, only the rows for the models evaluated now are replaced.

Examples:
    python -m src.experiments.gradcam_share
    python -m src.experiments.gradcam_share --models pet_cnn_v2
"""
import argparse
import csv

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data import COMPARISON_DIR, DATA_DIR
from src.experiments.common_evaluation import BATCH_SIZE, DEVICE, load_model, run_name, test_split, write_csv
from src.experiments.experiment2_background import PET_DIR, keep_other_models, read_metadata
from src.train.train_cnn import IMAGE_SIZE, eval_transform

BG_DIR = DATA_DIR / "background_test"
SEEDS = {"resnet50": 1, "pet_cnn": 2, "pet_cnn_v2": 2}  # median test accuracy over seeds 0-2, as in experiment 1
# The trimap gets the same resize and crop as the image, with nearest neighbour so the labels stay 1, 2, 3
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE),
])
TRIMAP_BACKGROUND = 2  # trimap labels: 1 = pet, 2 = background, 3 = border zone around the pet


class ImagesWithMasks(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = Image.open(PET_DIR / "images" / f"{name}.jpg").convert("RGB")
        trimap = np.asarray(mask_transform(Image.open(PET_DIR / "annotations" / "trimaps" / f"{name}.png")))
        return eval_transform(image), torch.from_numpy((trimap != TRIMAP_BACKGROUND).astype(np.float32))


def attention_per_image(model_name, model, names):
    """Predicted class, share on the pet, pet area and peak-on-pet for every image."""
    # Last convolutional block, the same layer as gradcam.py uses
    layer = model.layer4[-1] if model_name.startswith("resnet50") else model.features[-1]
    loader = DataLoader(ImagesWithMasks(names), batch_size=BATCH_SIZE, num_workers=2)
    preds, shares, areas, peaks = [], [], [], []
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for batch, (x, mask) in enumerate(loader):
            heat = cam(input_tensor=x.to(DEVICE), targets=None)  # targets=None: the predicted class
            mask = mask.numpy()
            preds.append(cam.outputs.argmax(dim=1).cpu().numpy())
            shares.append((heat * mask).sum(axis=(1, 2)) / np.maximum(heat.sum(axis=(1, 2)), 1e-12))
            areas.append(mask.mean(axis=(1, 2)))
            peaks.append(mask.reshape(len(mask), -1)[np.arange(len(mask)), heat.reshape(len(heat), -1).argmax(axis=1)])
            if (batch + 1) % 20 == 0:
                print(f"  {(batch + 1) * BATCH_SIZE}/{len(names)} images")
    return [np.concatenate(v) for v in (preds, shares, areas, peaks)]


def read_pet_only_changes():
    """{(run, image name): True if the pet_only prediction differs from the original} from experiment 2."""
    path = COMPARISON_DIR / "experiment2_predictions.csv"
    if not path.exists():
        print(f"{path.name} not found: run experiment2_background.py to link attention to accuracy changes")
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["model"], r["name"]): r["pred_pet_only"] != r["pred_original"] for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(SEEDS), default=list(SEEDS))
    args = parser.parse_args()

    if not (BG_DIR / "metadata.csv").exists():
        raise SystemExit(f"{BG_DIR} not found. Run make_background_test.py first (it lists the images with a trimap).")
    names, y = read_metadata(BG_DIR)
    breeds = test_split()[2]
    changes = read_pet_only_changes()
    print(f"{len(names)} test images, device {DEVICE}")

    summary, per_image = [], []
    for model_name in args.models:
        name = run_name(model_name, SEEDS[model_name])
        print(f"\n{name}")
        pred, share, area, peak = attention_per_image(model_name, load_model(model_name, SEEDS[model_name]), names)
        correct = pred == y

        row = {"model": model_name, "seed": SEEDS[model_name], "n": len(names), "accuracy": correct.mean(),
               "share": share.mean(), "share_median": np.median(share), "area": area.mean(),
               "share_over_area": share.mean() / area.mean(), "peak_on_pet": peak.mean(),
               "share_correct": share[correct].mean(), "share_wrong": share[~correct].mean()}
        print(f"  accuracy {row['accuracy']:.1%} | share on pet {row['share']:.1%} (median {row['share_median']:.1%}) "
              f"vs pet area {row['area']:.1%} -> ratio {row['share_over_area']:.2f} | peak on pet {row['peak_on_pet']:.1%}")
        print(f"  share when correct {row['share_correct']:.1%}, when wrong {row['share_wrong']:.1%}")

        changed = None
        if any(run == name for run, _ in changes):
            changed = np.array([changes.get((name, n), False) for n in names])
            row.update({"n_changed_pet_only": int(changed.sum()),
                        "share_unchanged_pet_only": share[~changed].mean(),
                        "share_changed_pet_only": share[changed].mean(),
                        "corr_share_changed": np.corrcoef(share, changed)[0, 1]})
            print(f"  share when pet_only kept the prediction {row['share_unchanged_pet_only']:.1%}, "
                  f"changed it {row['share_changed_pet_only']:.1%} (n={row['n_changed_pet_only']}), "
                  f"correlation {row['corr_share_changed']:+.3f}")
        else:
            print(f"  {name} is not in experiment2_predictions.csv: run experiment 2 for it to link attention "
                  f"to accuracy changes")
        summary.append({k: float(v) if isinstance(v, np.floating) else v for k, v in row.items()})

        for i, n in enumerate(names):
            per_image.append({"model": name, "name": n, "true": breeds[y[i]], "pred": breeds[pred[i]],
                              "correct": int(correct[i]), "share": float(share[i]), "area": float(area[i]),
                              "peak_on_pet": int(peak[i]),
                              "changed_pet_only": "" if changed is None else int(changed[i])})

    # Rows for models not evaluated now are kept, so one model can be added without running the others
    for filename, rows in [("gradcam_share_summary.csv", summary), ("gradcam_share_per_image.csv", per_image)]:
        write_csv(COMPARISON_DIR / filename, keep_other_models(COMPARISON_DIR / filename, rows, set(args.models)))


if __name__ == "__main__":
    main()
