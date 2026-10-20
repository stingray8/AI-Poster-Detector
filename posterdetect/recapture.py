"""Simulate "print it, photograph it" on a digital image.

Training on clean digital images and testing on phone photos of prints fails
badly (see RRDataset). This module approximates the main degradations so you
can generate recaptured versions of millions of images cheaply. It is an
approximation: always validate on REAL physical captures.

Order of operations roughly follows the physical chain:
  print (halftone, ink/paper colour) -> camera pose -> sensor sampling ->
  lens blur -> moire -> vignette / glare -> sensor noise -> JPEG.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RecaptureConfig:
    strength: float = 1.0          # global multiplier on degradation amplitudes
    p_halftone: float = 0.4
    p_color: float = 0.95
    p_perspective: float = 0.6
    p_rescale: float = 0.9
    p_blur: float = 0.85
    p_moire: float = 0.25
    p_vignette: float = 0.5
    p_glare: float = 0.35
    p_noise: float = 0.85
    jpeg_quality: tuple = (35, 95)
    p_double_jpeg: float = 0.2


class RecaptureSimulator:
    def __init__(self, config: RecaptureConfig | None = None, seed: int | None = None):
        self.cfg = config or RecaptureConfig()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ api
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """uint8 RGB (H, W, 3) -> uint8 RGB, same size."""
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        c = self.cfg
        x = img.astype(np.float32) / 255.0
        steps = [
            (c.p_halftone, self._halftone), (c.p_color, self._color),
            (c.p_perspective, self._perspective), (c.p_rescale, self._rescale),
            (c.p_blur, self._blur), (c.p_moire, self._moire),
            (c.p_vignette, self._vignette), (c.p_glare, self._glare),
            (c.p_noise, self._noise),
        ]
        for p, fn in steps:
            if self.rng.random() < p:
                x = fn(x)
        out = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
        out = self._jpeg(out)
        if self.rng.random() < c.p_double_jpeg:
            out = self._jpeg(out)
        return out

    # ------------------------------------------------------------ operations
    @staticmethod
    def _grid(h, w):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return xx, yy

    def _halftone(self, x):
        r, c = self.rng, self.cfg
        h, w = x.shape[:2]
        xx, yy = self._grid(h, w)
        period = max(2.5, max(h, w) / r.uniform(160, 450))
        ink = 1.0 - x
        out = np.empty_like(ink)
        angles = np.array([15.0, 75.0, 0.0]) + r.uniform(-10, 10)  # C, M, Y screens
        for ch in range(3):
            a = np.deg2rad(angles[ch])
            u = xx * np.cos(a) + yy * np.sin(a)
            v = -xx * np.sin(a) + yy * np.cos(a)
            screen = 0.25 * (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period)) + 0.5
            k = r.uniform(8, 25)
            out[..., ch] = 1.0 / (1.0 + np.exp(-k * (ink[..., ch] - screen)))
        alpha = float(np.clip(r.uniform(0.4, 1.0) * c.strength, 0, 1))
        return (1 - alpha) * x + alpha * (1.0 - out)

    def _color(self, x):
        r, c = self.rng, self.cfg
        s = c.strength
        x = x * (1 + s * r.uniform(-0.15, 0.15, size=3)).astype(np.float32)       # white balance
        x = np.clip(x, 0, 1) ** float(np.exp(s * r.uniform(-0.25, 0.25)))          # gamma
        gray = x.mean(axis=2, keepdims=True)
        x = gray + float(1 + s * r.uniform(-0.3, 0.1)) * (x - gray)                # saturation
        lo, hi = s * r.uniform(0.0, 0.08), 1 - s * r.uniform(0.0, 0.1)             # paper black/white point
        return lo + np.clip(x, 0, 1) * (hi - lo)

    def _perspective(self, x):
        r, c = self.rng, self.cfg
        h, w = x.shape[:2]
        j = r.uniform(0.005, 0.05) * c.strength * min(h, w)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + r.uniform(-j, j, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _rescale(self, x):
        r = self.rng
        h, w = x.shape[:2]
        f = r.uniform(0.3, 0.95)  # camera samples the print at lower density
        small = cv2.resize(x, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        interp = cv2.INTER_LINEAR if r.random() < 0.5 else cv2.INTER_CUBIC
        return cv2.resize(small, (w, h), interpolation=interp)

    def _blur(self, x):
        r, c = self.rng, self.cfg
        if r.random() < 0.7:
            return cv2.GaussianBlur(x, (0, 0), float(r.uniform(0.3, 1.6) * c.strength))
        L = int(r.integers(3, 10))
        k = np.zeros((L, L), np.float32)
        ang = r.uniform(0, np.pi)
        ctr = (L - 1) / 2
        dx, dy = np.cos(ang) * ctr, np.sin(ang) * ctr
        cv2.line(k, (int(round(ctr - dx)), int(round(ctr - dy))), (int(round(ctr + dx)), int(round(ctr + dy))), 1.0, 1)
        k /= max(k.sum(), 1e-6)
        return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    def _moire(self, x):
        r, c = self.rng, self.cfg
        h, w = x.shape[:2]
        xx, yy = self._grid(h, w)
        p1 = r.uniform(3, 9)
        p2 = p1 * r.uniform(0.9, 1.1)
        t1 = r.uniform(0, np.pi)
        t2 = t1 + r.uniform(-0.1, 0.1)
        g1 = np.sin(2 * np.pi * (xx * np.cos(t1) + yy * np.sin(t1)) / p1)
        g2 = np.sin(2 * np.pi * (xx * np.cos(t2) + yy * np.sin(t2)) / p2)
        amp = r.uniform(0.01, 0.05) * c.strength
        return x * (1 + amp * (g1 + g2))[..., None]

    def _vignette(self, x):
        r, c = self.rng, self.cfg
        h, w = x.shape[:2]
        xx, yy = self._grid(h, w)
        cx, cy = w / 2 + r.uniform(-0.1, 0.1) * w, h / 2 + r.uniform(-0.1, 0.1) * h
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        return x * (1 - r.uniform(0.05, 0.35) * c.strength * d2)[..., None]

    def _glare(self, x):
        r, c = self.rng, self.cfg
        h, w = x.shape[:2]
        xx, yy = self._grid(h, w)
        cx, cy = r.uniform(0, w), r.uniform(0, h)
        sx, sy = r.uniform(0.1, 0.4) * w, r.uniform(0.1, 0.4) * h
        blob = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        return x + (blob * r.uniform(0.05, 0.35) * c.strength)[..., None]

    def _noise(self, x):
        r, c = self.rng, self.cfg
        a = r.uniform(0.0005, 0.01) * c.strength   # shot-noise gain
        b = r.uniform(1e-5, 5e-4) * c.strength     # read-noise floor
        sigma = np.sqrt(np.clip(a * np.clip(x, 0, 1) + b, 0, None))
        return x + r.standard_normal(x.shape).astype(np.float32) * sigma.astype(np.float32)

    def _jpeg(self, img):
        lo, hi = self.cfg.jpeg_quality
        q = int(self.rng.integers(lo, hi + 1))
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
        assert ok
        return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1])
