"""Tiling + frozen foundation-model feature extraction.

`tile_image` is pure numpy/OpenCV. `FeatureExtractor` lazily imports torch,
open_clip and transformers so the rest of the package works without them.

Backbones:  clip-l14 | dinov2-b | fusion (concat of both)
"""
from __future__ import annotations

import cv2
import numpy as np

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def tile_image(img: np.ndarray, tile: int = 224, long_side: int = 896, max_tiles: int = 16,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Resize so the long side == `long_side`, then cut a grid of `tile`x`tile`
    crops (covering the remainder too). If there are more than `max_tiles`,
    subsample: evenly (deterministic) when rng is None, randomly otherwise.
    Returns (n, tile, tile, 3) uint8.

    `long_side` is a key hyperparameter: it sets how much of the print/camera
    texture each tile sees. Sweep it (e.g. 448 / 896 / 1344).
    """
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s != 1:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
    h, w = img.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        h, w = img.shape[:2]
    ys = list(range(0, h - tile + 1, tile))
    xs = list(range(0, w - tile + 1, tile))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    coords = [(y, x) for y in ys for x in xs]
    if len(coords) > max_tiles:
        if rng is None:
            idx = np.linspace(0, len(coords) - 1, max_tiles).round().astype(int)
        else:
            idx = np.sort(rng.choice(len(coords), max_tiles, replace=False))
        coords = [coords[i] for i in idx]
    return np.ascontiguousarray(np.stack([img[y:y + tile, x:x + tile] for y, x in coords]))


class FeatureExtractor:
    def __init__(self, backbone: str = "clip-l14", device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        names = ["clip-l14", "dinov2-b"] if backbone == "fusion" else [backbone]
        self.encoders = [self._load(n) for n in names]

    def _load(self, name: str):
        torch = self.torch
        if name == "clip-l14":
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            model = model.eval().to(self.device)
            return (lambda x: model.encode_image(x)), CLIP_MEAN, CLIP_STD
        if name == "dinov2-b":
            from transformers import AutoModel

            model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(self.device)

            def fn(x):
                h = model(pixel_values=x).last_hidden_state
                return torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1)

            return fn, IN_MEAN, IN_STD
        raise ValueError(f"unknown backbone {name!r}")

    def encode(self, tiles: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """(n, 224, 224, 3) uint8 -> (n, D) float32."""
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(tiles), batch_size):
                x = torch.from_numpy(tiles[i:i + batch_size]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
                feats = []
                for fn, mean, std in self.encoders:
                    m = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
                    s = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
                    with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device.startswith("cuda")):
                        f = fn((x - m) / s)
                    feats.append(f.float())
                outs.append(torch.cat(feats, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)
