"""Dataset helpers.

Expected layout (label 0 = real, 1 = fake):

    data/
      train/{real,fake}/*.jpg        digital images (recapture is simulated)
      val/{real,fake}/*.jpg
      test_digital/{real,fake}/*.jpg
      test_physical/{real,fake}/*.jpg   REAL camera photos of printed posters

Rules that matter more than model choice:
  * Never put the same source image (or its recapture) in two splits.
  * Real and fake must go through the same capture pipeline (same printers,
    papers, phones, lighting), or the model will learn the medium.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABELS = {"real": 0, "fake": 1}


def list_images(root: str | Path, split: str):
    """Return (paths, labels) for root/split/{real,fake}."""
    paths, labels = [], []
    for cls, lab in LABELS.items():
        d = Path(root) / split / cls
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(lab)
    if not paths:
        raise FileNotFoundError(f"No images found under {Path(root) / split}/(real|fake)")
    return paths, np.array(labels, dtype=np.int64)


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB, honouring EXIF orientation (phone photos)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)
