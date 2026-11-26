from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

# Project folders, used by all scripts
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"  # results that cover all models (experiment 2)
TARGETS = ["category", "segmentation"]
SPLIT_SEED = 42

# Results are stored per model family: both SVMs in results/svm/, both ResNet-50s in results/resnet50/
RESULT_GROUPS = {"svm_hog": "svm", "svm_hog_color": "svm", "resnet50_untrained": "resnet50"}


def model_dir(model):
    """results/<family>/ for a model, e.g. results/pet_cnn/."""
    return RESULTS_DIR / RESULT_GROUPS.get(model, model)


def fig_dir(model):
    return model_dir(model) / "figures"


def history_dir(model):
    """Where train_cnn.py writes the training history (loss and accuracy per epoch)."""
    return model_dir(model) / "training"


def load_trainval():
    dataset = OxfordIIITPet(DATA_DIR, split="trainval", target_types=TARGETS)
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2, # 80 / 20 split
        shuffle=True,
        stratify=dataset._labels,  
        random_state=SPLIT_SEED,
    )
    return dataset, train_idx, val_idx

def load_test():
    dataset = OxfordIIITPet(DATA_DIR, split="test", target_types=TARGETS)
    return dataset

