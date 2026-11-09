from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import SPLIT_SEED, load_trainval

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
C = 1

def hog_features(image):
    gray = np.asarray(image.convert("L").resize((128, 128)))
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")


def extract_features(dataset, indices):
    X, y = [], []
    for idx in indices:
        image, (label, _) = dataset[idx]
        X.append(hog_features(image))
        y.append(label)
    return np.array(X), np.array(y)


def build_model():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale"))


def train_svm():
    dataset, train_idx, val_idx = load_trainval()
    X_train, y_train = extract_features(dataset, train_idx)
    X_val, y_val = extract_features(dataset, val_idx)
    print("Feature shape:", X_train.shape)

    model = build_model().fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    f1 = f1_score(y_val, model.predict(X_val), average="macro")
    print(f"C={C}: accuracy={acc:.4f}, macro-F1={f1:.4f}")

    rng = np.random.default_rng(SPLIT_SEED)
    y_shuffled = rng.permutation(y_train)
    control = build_model().fit(X_train, y_shuffled).score(X_val, y_val)
    print(f"Shuffled-label control: {control:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHECKPOINT_DIR / "svm_hog.joblib")


if __name__ == "__main__":
    print("Training SVM with HOG features...")
    train_svm()