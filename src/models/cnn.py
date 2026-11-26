import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

NUM_CLASSES = 37


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


def double_conv_block(in_channels, out_channels):
    """Two 3x3 convolutions before each pooling (VGG style), so each block sees a larger area.

    No conv bias: BatchNorm subtracts the mean right after, so the bias would do nothing.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class PetCNNv2(nn.Module):
    """Larger PetCNN: five blocks with two convolutions each.

    Receptive field about 156 px of 224 (PetCNN: about 46 px), about 2.4 M parameters
    (PetCNN: about 0.4 M). The feature map before pooling is 7x7x256.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            double_conv_block(3, 32),     # 224 -> 112
            double_conv_block(32, 64),    # 112 -> 56
            double_conv_block(64, 128),   # 56 -> 28
            double_conv_block(128, 256),  # 28 -> 14
            double_conv_block(256, 256),  # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
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
    if name == "pet_cnn_v2":
        return PetCNNv2()
    raise ValueError(f"Unknown model: {name}")
