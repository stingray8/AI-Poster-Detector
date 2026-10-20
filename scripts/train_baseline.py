"""Fit a linear probe (UnivFD-style) on frozen features and report metrics.

  python scripts/train_baseline.py --train feats/train_sim.npz \
      --eval val_clean=feats/val_clean.npz --eval physical=feats/test_physical.npz \
      --out models/clip_linear.joblib

Tiles are the training unit; image score = mean tile logit.
Pick the decision threshold on VAL data at a fixed FPR, not on the physical test set.
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from posterdetect.metrics import aggregate_by_sample, summarize


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def tile_labels(z):
    return z["labels"][z["sample_idx"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="name=path.npz")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--out", default="models/baseline.joblib")
    a = ap.parse_args()

    tr = [load(p) for p in a.train]
    X = np.concatenate([z["feats"].astype(np.float32) for z in tr])
    y = np.concatenate([tile_labels(z) for z in tr])
    meta = {k: tr[0][k].item() for k in ("backbone", "tile", "long_side", "max_tiles")}
    print(f"train tiles: {len(y)}  fake frac: {y.mean():.3f}  dim: {X.shape[1]}")

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced"))
    clf.fit(X, y)

    print(f"\n{'set':<18}{'n':>6}{'auc':>8}{'ap':>8}{'tpr@1%':>9}{'tpr@5%':>9}{'acc':>8}")
    for spec in a.eval:
        name, path = spec.split("=", 1)
        z = load(path)
        tile_scores = clf.decision_function(z["feats"].astype(np.float32))
        n = len(z["labels"])
        scores = aggregate_by_sample(z["sample_idx"], tile_scores, n)
        m = summarize(z["labels"], scores)
        print(f"{name:<18}{m['n']:>6}{m['auc']:>8.3f}{m['ap']:>8.3f}{m['tpr@1%fpr']:>9.3f}{m['tpr@5%fpr']:>9.3f}{m['acc']:>8.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    joblib.dump({"model": clf, "meta": meta}, a.out)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
