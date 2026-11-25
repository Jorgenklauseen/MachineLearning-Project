from torchvision.datasets import OxfordIIITPet

from src.data import DATA_DIR

train = OxfordIIITPet(DATA_DIR, split="trainval", target_types=["category", "segmentation"])
test = OxfordIIITPet(DATA_DIR, split="test", target_types=["category", "segmentation"])

print(len(train), len(test), len(train.classes))
image, (breed, mask) = train[0]
print(image.size, mask.size, breed)
