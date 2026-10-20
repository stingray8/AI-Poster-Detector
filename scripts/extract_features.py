"""Tile each image, run it through a frozen backbone, save features to .npz.

Examples
  # clean digital val set
  python scripts/extract_features.py --data-root data --split val --backbone clip-l14 --out feats/val_clean.npz

  # train set with simulated recapture (2 random recaptures per image, plus the clean one)
  python scripts/extract_features.py --data-root data --split train --recapture-sim --copies 2 --keep-clean \
      --backbone clip-l14 --out feats/train_sim.npz

  # real phone photos of printed posters: detect + rectify the poster first
  python scripts/extract_features.py --data-root data --split test_physical --rectify \
      --backbone clip-l14 --out feats/test_physical.npz
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from posterdetect.data import list_images, read_rgb
from posterdetect.features import FeatureExtractor, tile_image
from posterdetect.recapture import RecaptureConfig, RecaptureSimulator
from posterdetect.rectify import rectify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--backbone", default="clip-l14", choices=["clip-l14", "dinov2-b", "fusion"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rectify", action="store_true", help="detect+warp the poster (for real phone photos)")
    ap.add_argument("--recapture-sim", action="store_true")
    ap.add_argument("--copies", type=int, default=1, help="simulated recaptures per image")
    ap.add_argument("--keep-clean", action="store_true", help="also keep the un-degraded version")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--long-side", type=int, default=896)
    ap.add_argument("--max-tiles", type=int, default=16)
    ap.add_argument("--pre-resize", type=int, default=1536, help="cap long side before simulation (speed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    a = ap.parse_args()

    paths, labels = list_images(a.data_root, a.split)
    if a.max_images:
        paths, labels = paths[:a.max_images], labels[:a.max_images]
    ext = FeatureExtractor(a.backbone)
    sim = RecaptureSimulator(RecaptureConfig(strength=a.strength), seed=a.seed) if a.recapture_sim else None

    feats, sample_idx, s_labels, s_group, s_sim, s_paths = [], [], [], [], [], []
    n_rect_fail = 0
    for gi, (p, lab) in enumerate(tqdm(list(zip(paths, labels)))):
        try:
            img = read_rgb(p)
        except Exception as e:  # corrupt file
            print("skip", p, e)
            continue
        if a.rectify:
            img, found = rectify(img)
            n_rect_fail += (not found)
        m = max(img.shape[:2])
        if m > a.pre_resize:
            img = cv2.resize(img, None, fx=a.pre_resize / m, fy=a.pre_resize / m, interpolation=cv2.INTER_AREA)

        versions = []
        if sim is None or a.keep_clean:
            versions.append((img, 0))
        if sim is not None:
            versions += [(sim(img), 1) for _ in range(a.copies)]

        for v, is_sim in versions:
            tiles = tile_image(v, a.tile, a.long_side, a.max_tiles)
            f = ext.encode(tiles)
            sid = len(s_labels)
            feats.append(f)
            sample_idx.append(np.full(len(f), sid, dtype=np.int64))
            s_labels.append(lab); s_group.append(gi); s_sim.append(is_sim); s_paths.append(p)

    if a.rectify:
        print(f"poster not found (used full image) in {n_rect_fail}/{len(paths)} photos")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, feats=np.concatenate(feats).astype(np.float16), sample_idx=np.concatenate(sample_idx),
        labels=np.array(s_labels), group=np.array(s_group), is_sim=np.array(s_sim), paths=np.array(s_paths),
        backbone=a.backbone, tile=a.tile, long_side=a.long_side, max_tiles=a.max_tiles,
    )
    print("saved", a.out, "samples:", len(s_labels), "tiles:", sum(len(f) for f in feats))


if __name__ == "__main__":
    main()
