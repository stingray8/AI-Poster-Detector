"""Write a grid of simulated recaptures so you can eyeball them against real
phone photos of prints. Tune RecaptureConfig until they look plausible.

  python scripts/preview_recapture.py --image some.jpg --out preview.jpg --n 5 --strength 1.0
"""
import argparse

import cv2
import numpy as np

from posterdetect.data import read_rgb
from posterdetect.recapture import RecaptureConfig, RecaptureSimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="preview.jpg")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    img = read_rgb(a.image)
    s = 1024 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sim = RecaptureSimulator(RecaptureConfig(strength=a.strength), seed=a.seed)
    tiles = [img] + [sim(img) for _ in range(a.n)]
    cols = min(3, len(tiles))
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(img)] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    grid = np.vstack(rows)
    cv2.imwrite(a.out, grid[..., ::-1])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
