from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

# Project folders, used by all scripts
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TARGETS = ["category", "segmentation"]
SPLIT_SEED = 42

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

