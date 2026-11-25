"""Shared code for the experiments: load a trained model (SVM or CNN) and predict on image files.

Every model is used the same way, so experiment1_errors.py and experiment2_background.py do not need to
know which kind of model they evaluate:
    model = load_model("resnet50", seed=0)
    pred = predict("resnet50", model, paths)
"""
import csv

import joblib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet

from src.data import CHECKPOINT_DIR, DATA_DIR
from src.models.cnn import build_model
from src.models.svm import SVM_IMAGE_SIZE
from src.train.train_cnn import BATCH_SIZE, eval_transform, load_checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT = 2000

SVM_MODELS = {"svm_hog": False, "svm_hog_color": True}  # name -> uses colour
CNN_MODELS = ["resnet50", "resnet50_untrained", "pet_cnn"]
ALL_MODELS = [*SVM_MODELS, *CNN_MODELS]
LABELS = {
    "svm_hog": "HOG + SVM",
    "svm_hog_color": "HOG + color hist + SVM",
    "resnet50": "ResNet-50 (pretrained)",
    "resnet50_untrained": "ResNet-50 (from scratch)",
    "pet_cnn": "PetCNN",
}


# ---------- Models ----------

def run_name(model, seed):
    """SVMs have one checkpoint each, CNNs one per training seed."""
    return model if model in SVM_MODELS else f"{model}_seed{seed}"


def checkpoint_path(model, seed):
    return CHECKPOINT_DIR / (f"{model}.joblib" if model in SVM_MODELS else f"{model}_seed{seed}.pt")


def load_model(model, seed=0):
    path = checkpoint_path(model, seed)
    if model in SVM_MODELS:
        return joblib.load(path)
    state, meta = load_checkpoint(path, DEVICE)
    if meta:
        print(f"{run_name(model, seed)}: best epoch {meta['epoch']}, val_accuracy {meta['val_accuracy']:.4f}")
    else:
        print(f"{run_name(model, seed)}: old checkpoint without metadata")
    net = build_model(model)
    net.load_state_dict(state)
    return net.to(DEVICE).eval()


# ---------- Prediction ----------

class ImageFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_transform(Image.open(self.paths[i]).convert("RGB"))


def svm_images(paths, color):
    """Same preprocessing as load_images in train_svm.py: 128x128, greyscale or RGB."""
    mode = "RGB" if color else "L"
    return np.array([
        np.asarray(Image.open(p).convert("RGB").convert(mode).resize((SVM_IMAGE_SIZE, SVM_IMAGE_SIZE)))
        for p in paths
    ])


@torch.no_grad()
def predict(model_name, model, paths):
    """Predicted class ids for a list of image files."""
    if model_name in SVM_MODELS:
        return model.predict(svm_images(paths, SVM_MODELS[model_name]))
    loader = DataLoader(ImageFiles(paths), batch_size=BATCH_SIZE, num_workers=2)
    return torch.cat([model(x.to(DEVICE)).argmax(dim=1).cpu() for x in loader]).numpy()


# ---------- Data and helpers ----------

def test_split():
    """The official test split: image paths, labels and breed names."""
    test = OxfordIIITPet(DATA_DIR, split="test", target_types="category")
    return test._images, np.array(test._labels), test.classes


def bootstrap_ci(correct, rng):
    """95 % confidence interval for accuracy, by resampling the test images."""
    idx = rng.integers(0, len(correct), size=(N_BOOT, len(correct)))
    return np.percentile(correct[idx].mean(axis=1), [2.5, 97.5])


def seed_summary(rows, metrics, keep=("model", "variant", "n")):
    """One row per seed for the same model -> three rows with mean, std and median over the seeds.

    Columns in `metrics` are aggregated, columns in `keep` are copied, the rest are left empty.
    """
    out = []
    for stat, fn in [("mean", np.mean), ("std", lambda v: np.std(v, ddof=1)), ("median", np.median)]:
        row = {k: (rows[0][k] if k in keep else "") for k in rows[0]}
        row["seed"] = stat
        for m in metrics:
            row[m] = float(fn([r[m] for r in rows]))
        out.append(row)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print(f"Saved {path}")
