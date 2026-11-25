import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SVM_IMAGE_SIZE = 128
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])  # same weighting as PIL convert("L")


class HOG(BaseEstimator, TransformerMixin):
    def __init__(self, ppc=24):
        self.ppc = ppc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            hog(img @ GRAY_WEIGHTS if img.ndim == 3 else img,
                orientations=9, pixels_per_cell=(self.ppc, self.ppc),
                cells_per_block=(2, 2), block_norm="L2-Hys")
            for img in X
        ])


class ColorHist(BaseEstimator, TransformerMixin):
    def __init__(self, bins=16):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for img in X:
            hsv = rgb2hsv(img).reshape(-1, 3)
            feats.append(np.concatenate([
                np.histogram(hsv[:, c], bins=self.bins, range=(0, 1))[0] / len(hsv)
                for c in range(3)
            ]))
        return np.array(feats)


def build_svm(use_color=False):
    """use_color=False: HOG only (grayscale). use_color=True: HOG + HSV histogram (RGB input)."""
    features = FeatureUnion([
        ("hog", HOG()),
        ("color", ColorHist() if use_color else "drop"),
    ])
    return Pipeline([
        ("features", features),
        ("scaler", StandardScaler()),
        ("svc", SVC()),
    ])
