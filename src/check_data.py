from torchvision.datasets import OxfordIIITPet

train = OxfordIIITPet("data", split="trainval", target_types=["category", "segmentation"])
test = OxfordIIITPet("data", split="test", target_types=["category", "segmentation"])

print(len(train), len(test), len(train.classes))
bilde, (rase, maske) = train[0]
print(bilde.size, maske.size, rase)
