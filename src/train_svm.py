import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import SPLIT_SEED, load_test, load_trainval
from models import SVM_IMAGE_SIZE, build_svm

ROOT = Path(__file__).resolve().parents[1]


def model_path(color=False):
    return ROOT / "checkpoints" / ("svm_hog_color.joblib" if color else "svm_hog.joblib")


def results_file(color=False):
    return ROOT / "results" / ("svm_gridsearch_color.csv" if color else "svm_gridsearch.csv")


PPC_VALUES = [16, 24, 32, 42]
COLOR_BINS = [16]                  # bins per HSV channel (only with --color)
C_RBF = [1, 10]
GAMMA_FACTORS = [1, 2, 4]          
C_LINEAR = [0.001, 0.01, 0.1, 1]
K = 5
N_JOBS = 4


def load_images(dataset, indices, color=False):
    """128x128 images, as the SVM pipeline expects: grayscale, or RGB with color=True."""
    mode = "RGB" if color else "L"
    X, y = [], []
    for idx in indices:
        image, (label, _) = dataset[idx]
        X.append(np.asarray(image.convert(mode).resize((SVM_IMAGE_SIZE, SVM_IMAGE_SIZE))))
        y.append(label)
    return np.array(X), np.array(y)


def n_features(ppc, color=False, bins=None):
    features = build_svm(color).named_steps["features"]
    features.set_params(hog__ppc=ppc, **({"color__bins": bins} if color else {}))
    shape = (1, SVM_IMAGE_SIZE, SVM_IMAGE_SIZE) + ((3,) if color else ())
    return features.transform(np.zeros(shape)).shape[1]


def make_param_grid(color=False):
    grid = []
    for ppc in PPC_VALUES:
        for bins in (COLOR_BINS if color else [None]):
            base = {"features__hog__ppc": [ppc]}
            if color:
                base["features__color__bins"] = [bins]
            n = n_features(ppc, color, bins)
            grid.append({**base, "svc__kernel": ["rbf"], "svc__C": C_RBF,
                         "svc__gamma": [f / n for f in GAMMA_FACTORS]})
            grid.append({**base, "svc__kernel": ["linear"], "svc__C": C_LINEAR})
    return grid


def save_results(search, color=False):
    r = search.cv_results_
    rows = []
    for i, p in enumerate(r["params"]):
        ppc, bins = p["features__hog__ppc"], p.get("features__color__bins")
        n = n_features(ppc, color, bins)
        rows.append(dict(
            ppc=ppc, bins=bins if color else "", n_features=n,
            kernel=p["svc__kernel"], C=p["svc__C"],
            gamma_factor=round(p["svc__gamma"] * n) if "svc__gamma" in p else "",
            mean_accuracy=r["mean_test_score"][i], std_accuracy=r["std_test_score"][i],
        ))
    rows.sort(key=lambda row: row["mean_accuracy"], reverse=True)

    out = results_file(color)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nTop 5 (5-fold on training data):")
    for row in rows[:5]:
        gamma = f", gamma×{row['gamma_factor']}" if row["gamma_factor"] != "" else ""
        bins = f", bins={row['bins']}" if row["bins"] != "" else ""
        print(f"  ppc={row['ppc']}{bins}, {row['kernel']}, C={row['C']}{gamma}: "
              f"{row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", action="store_true",
                        help="HOG + HSV color histogram instead of HOG on grayscale only")
    color = parser.parse_args().color
    print(f"Variant: {'HOG + color histogram' if color else 'HOG (grayscale)'}")

    dataset, train_idx, val_idx = load_trainval()
    X_train, y_train = load_images(dataset, train_idx, color)
    X_val, y_val = load_images(dataset, val_idx, color)

    # 1. Choose settings with 5-fold CV on the training data. The test set is not used here.
    cv = StratifiedKFold(K, shuffle=True, random_state=SPLIT_SEED)
    search = GridSearchCV(build_svm(color), make_param_grid(color), cv=cv,
                          scoring="accuracy", n_jobs=N_JOBS, verbose=2)
    search.fit(X_train, y_train)   # finally refits the best setup on the full training split
    save_results(search, color)

    model = search.best_estimator_
    print(f"\nSelected:{search.best_params_}")
    print(f"Validation: accuracy={model.score(X_val, y_val):.4f}")

    path = model_path(color)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    # 2. The test set, once
    test_dataset = load_test()
    X_test, y_test = load_images(test_dataset, range(len(test_dataset)), color)
    y_pred = model.predict(X_test)
    print(f"Test: accuracy={(y_pred == y_test).mean():.4f}, "
          f"macro-F1={f1_score(y_test, y_pred, average='macro'):.4f}")

    # 3. Sanity check: shuffled labels should give about 1/37 = 0.027
    y_shuffled = np.random.default_rng(SPLIT_SEED).permutation(y_train)
    control = clone(model).fit(X_train, y_shuffled).score(X_test, y_test)
    print(f"Control with shuffled labels (test): {control:.4f}")


if __name__ == "__main__":
    main()