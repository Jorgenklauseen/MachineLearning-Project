from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGETS = ["category", "segmentation"]
SPLIT_SEED = 42

def load_trainval():
    dataset = OxfordIIITPet(DATA_DIR, split="trainval", target_types=TARGETS)
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        shuffle=True,
        stratify=dataset._labels,  
        random_state=SPLIT_SEED,
    )
    return dataset, train_idx, val_idx

def load_test():
    dataset = OxfordIIITPet(DATA_DIR, split="test", target_types=TARGETS)
    return dataset

