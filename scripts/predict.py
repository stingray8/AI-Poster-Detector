"""Classify a single photo.

  python scripts/predict.py photo.jpg --model models/baseline.joblib [--no-rectify]
"""
import argparse

import joblib
import numpy as np

from posterdetect.data import read_rgb
from posterdetect.features import FeatureExtractor, tile_image
from posterdetect.rectify import rectify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-rectify", action="store_true")
    ap.add_argument("--low", type=float, default=0.35, help="below this P(fake) -> 'likely real'")
    ap.add_argument("--high", type=float, default=0.65, help="above this P(fake) -> 'likely AI'")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    clf, meta = bundle["model"], bundle["meta"]
    img = read_rgb(a.image)
    if not a.no_rectify:
        img, found = rectify(img)
        print("poster detected & rectified" if found else "poster not found; using full image")

    tiles = tile_image(img, meta["tile"], meta["long_side"], meta["max_tiles"])
    f = FeatureExtractor(meta["backbone"]).encode(tiles)
    logits = clf.decision_function(f)
    p = 1 / (1 + np.exp(-logits.mean()))
    verdict = "likely AI-generated" if p > a.high else "likely real" if p < a.low else "uncertain"
    print(f"P(fake) = {p:.3f}  ({verdict})   tiles={len(tiles)}  tile-logit std={logits.std():.2f}")
    print("NOTE: uncalibrated baseline; fit temperature scaling on validation data before trusting P.")


if __name__ == "__main__":
    main()
