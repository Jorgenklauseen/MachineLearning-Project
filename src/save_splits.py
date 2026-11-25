"""Save the split definitions as CSV files in splits/ (image name, class id, breed).
The train/validation split comes from data.load_trainval (stratified 80/20, seed SPLIT_SEED);
the test split is the official one."""
import csv
from pathlib import Path

from data import SPLIT_SEED, load_test, load_trainval

SPLIT_DIR = Path(__file__).resolve().parents[1] / "splits"


def rows(dataset, indices):
    for i in indices:
        label = dataset._labels[i]
        yield {"name": Path(dataset._images[i]).stem, "class_id": label, "breed": dataset.classes[label]}


def main():
    dataset, train_idx, val_idx = load_trainval()
    test = load_test()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, ds, idx in [("train", dataset, train_idx), ("val", dataset, val_idx),
                           ("test", test, range(len(test)))]:
        path = SPLIT_DIR / f"{split}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "class_id", "breed"])
            writer.writeheader()
            writer.writerows(rows(ds, idx))
        print(f"{split}: {len(idx)} images -> {path}")
    print(f"Split seed: {SPLIT_SEED}")


if __name__ == "__main__":
    main()
