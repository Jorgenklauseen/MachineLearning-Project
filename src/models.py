import numpy as np
import torch.nn as nn
from skimage.color import rgb2hsv
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import ResNet50_Weights, resnet50

NUM_CLASSES = 37
SVM_IMAGE_SIZE = 128


# ---------- SVM-baseline ----------

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

# ---------- Neural Network ----------

def build_resnet50(pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class PetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(name):
    if name == "resnet50":
        return build_resnet50(pretrained=True)
    if name == "resnet50_untrained":
        return build_resnet50(pretrained=False)
    if name == "pet_cnn":
        return PetCNN()
    raise ValueError(f"Unknown model: {name}")

