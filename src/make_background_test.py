import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data import DATA_DIR

PET_DIR = DATA_DIR / "oxford-iiit-pet"
VARIANTS = ["original", "pet_only", "bg_only"]
GREY = np.array([124, 116, 104], dtype=np.uint8)

BOX_MARGIN = 4
MASK_DILATE = 4
FEATHER = 3


def read_test_split():
    rows = []
    for line in (PET_DIR / "annotations" / "test.txt").read_text().splitlines():
        name, class_id, species, _ = line.split()
        rows.append({
            "name": name,
            "breed": name.rsplit("_", 1)[0],
            "class_id": int(class_id) - 1,
            "species": "cat" if species == "1" else "dog",
        })
    return rows


def load(name):
    image = np.asarray(Image.open(PET_DIR / "images" / f"{name}.jpg").convert("RGB"))
    trimap = np.asarray(Image.open(PET_DIR / "annotations" / "trimaps" / f"{name}.png"))
    return image, trimap


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def pet_alpha(trimap):
    pet = (trimap == 1).astype(np.uint8)
    r = max(int(0.05 * min(trimap.shape)), 1)
    padded = cv2.copyMakeBorder(pet, r, r, r, r, cv2.BORDER_REPLICATE)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, disk(r))[r:-r, r:-r]

    _, labels = cv2.connectedComponents(1 - closed, connectivity=4)
    border_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    inside = (closed == 1) | ~np.isin(labels, border_labels)

    core = (trimap == 1) | (inside & (trimap == 3))
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1 - dist / FEATHER, 0, 1)
    alpha[trimap == 2] = 0
    return alpha


def make_fill(image, kind, rng):
    """What replaces the removed pixels: constant grey, or uniform random noise."""
    if kind == "grey":
        return np.broadcast_to(GREY, image.shape)
    return rng.integers(0, 256, size=image.shape, dtype=np.uint8)


def make_pet_only(image, trimap, fill):
    a = pet_alpha(trimap)[..., None].astype(np.float32)
    out = a * image + (1 - a) * fill.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def make_bg_only(image, trimap, shape, fill):
    """Returns the image and the mask of hidden pixels."""
    pet = (trimap != 2).astype(np.uint8)
    if shape == "box":
        x, y, w, h = cv2.boundingRect(pet)
        hide = np.zeros(pet.shape, dtype=bool)
        hide[max(y - BOX_MARGIN, 0):y + h + BOX_MARGIN, max(x - BOX_MARGIN, 0):x + w + BOX_MARGIN] = True
    else:
        hide = cv2.dilate(pet, disk(MASK_DILATE)).astype(bool)
    out = image.copy()
    out[hide] = fill[hide]
    return out, hide


def save_preview(path, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 3, figsize=(7.8, 2.4 * len(examples)), squeeze=False)
    for row, (name, images) in zip(axes, examples):
        for ax, image, title in zip(row, images, VARIANTS):
            ax.imshow(image)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "background_test")
    parser.add_argument("--bg-shape", choices=["box", "mask"], default="box")
    parser.add_argument("--fill", choices=["grey", "noise"], default="grey",
                        help="what replaces removed pixels; use a separate --out for each fill")
    parser.add_argument("--limit", type=int, help="lag bare N tilfeldig valgte bilder")
    parser.add_argument("--seed", type=int, default=42, help="seed for utvalget med --limit")
    parser.add_argument("--preview", type=Path, help="lagre et rutenett med eksempler hit")
    parser.add_argument("--preview-rows", type=int, default=6)
    args = parser.parse_args()

    rows, empty = [], []
    for row in read_test_split():
        (rows if (load(row["name"])[1] == 1).any() else empty).append(row)
    print(f"Utelater {len(empty)} bilder uten dyr i trimapen:", ", ".join(r["name"] for r in empty))

    if args.limit:
        pick = np.random.default_rng(args.seed).choice(len(rows), size=min(args.limit, len(rows)), replace=False)
        rows = [rows[i] for i in sorted(pick)]

    for variant in VARIANTS:
        (args.out / variant).mkdir(parents=True, exist_ok=True)

    examples = []
    for i, row in enumerate(rows, 1):
        image, trimap = load(row["name"])
        fill = make_fill(image, args.fill, np.random.default_rng([args.seed, i]))  # same noise every run
        bg_only, hide = make_bg_only(image, trimap, args.bg_shape, fill)
        row["bg_share"] = round(float(1 - hide.mean()), 4)   # share of the image left in bg_only
        images = [image, make_pet_only(image, trimap, fill), bg_only]
        for variant, img in zip(VARIANTS, images):
            Image.fromarray(img).save(args.out / variant / f"{row['name']}.png")
        if args.preview and len(examples) < args.preview_rows:
            examples.append((row["name"], images))
        if i % 200 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} bilder")

    with open(args.out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "breed", "class_id", "species", "bg_share"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "settings.txt").write_text(
        f"bg_shape={args.bg_shape}\nfill={args.fill}\nlimit={args.limit}\nseed={args.seed}\n"
        f"excluded={','.join(r['name'] for r in empty)}\n", encoding="utf-8")

    if args.preview:
        save_preview(args.preview, examples)
        print("Forhåndsvisning:", args.preview)
    print("Ferdig:", args.out)


if __name__ == "__main__":
    main()
