# posterdetect

Baseline for detecting AI-generated artwork from **camera photos of posters/prints**.

Pipeline: `photo -> find poster & rectify -> tile at native res -> frozen CLIP/DINOv2 features -> linear probe -> mean tile logit`

## Install
```bash
pip install -e ".[torch,dev]"     # or: pip install -r requirements.txt && export PYTHONPATH=.
pytest -q tests                   # CPU-only tests (rectify, recapture sim, tiling, metrics)
```

## Data layout
```
data/
  train/{real,fake}/            digital images; recapture simulated at feature-extraction time
  val/{real,fake}/              digital; used to pick threshold / C
  test_digital/{real,fake}/
  test_physical/{real,fake}/    REAL phone photos of printed posters (the test that matters)
```
Hard rules: no source image in two splits; real and fake go through the *same* print/capture
pipeline; hold out whole generators and whole phones for testing.

## Quickstart
```bash
# 0. sanity-check the simulator against real phone photos of prints
python scripts/preview_recapture.py --image poster.jpg --out preview.jpg --strength 1.0

# 1. features
python scripts/extract_features.py --data-root data --split train --recapture-sim --copies 2 --keep-clean --out feats/train_sim.npz
python scripts/extract_features.py --data-root data --split train --out feats/train_clean.npz
python scripts/extract_features.py --data-root data --split val --out feats/val_clean.npz
python scripts/extract_features.py --data-root data --split val --recapture-sim --out feats/val_sim.npz
python scripts/extract_features.py --data-root data --split test_physical --rectify --out feats/test_physical.npz

# 2. train + evaluate
python scripts/train_baseline.py --train feats/train_sim.npz \
   --eval val_clean=feats/val_clean.npz --eval val_sim=feats/val_sim.npz --eval physical=feats/test_physical.npz

# 3. one photo
python scripts/predict.py photo.jpg --model models/baseline.joblib
```

## Experiments to run first (in this order)
| # | Train on | Expectation / what it tells you |
|---|----------|--------------------------------|
| 1 | clean | high on clean val, big drop on `val_sim` and `physical` -> confirms the recapture gap |
| 2 | clean + simulated recapture | gap should shrink; if `physical` stays poor the simulator is unrealistic |
| 3 | #2 with `--backbone dinov2-b` / `fusion` | which features survive recapture |
| 4 | sweep `--long-side` 448/896/1344 | how much texture the tiles need |
| 5 | leave-one-generator-out | real generalization to unseen generators |
| 6 | add hard negatives (human-made vector/Photoshop posters, AI-upscaled real) | false-positive rate on real designs |

## Next steps (not implemented)
- LoRA fine-tuning of the backbone (C2P-CLIP / mixture-of-LoRA style), attention-MIL pooling over tiles.
- Paired consistency loss between a clean image and its recapture (invariance to capture channel).
- Temperature scaling on val; abstain band instead of hard threshold.
- OCR-based text-plausibility feature for text-heavy posters.
- Replace `find_poster_quad` with a segmentation model if the classical detector fails on your photos.
- Evaluate on RRDataset and TextFake as external benchmarks.
# AI-Poster-Detector
