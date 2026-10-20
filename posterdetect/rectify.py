"""Find the poster in a phone photo and warp it to a fronto-parallel rectangle."""
from __future__ import annotations

import cv2
import numpy as np


def order_points(pts) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]  # y - x
    return np.stack([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]]).astype(np.float32)


def find_poster_quad(img_rgb: np.ndarray, min_area_frac: float = 0.15, max_area_frac: float = 0.97,
                     work_size: int = 900):
    """Return the 4 corners (ordered, original-image coordinates) of the largest
    convex quadrilateral that plausibly is the poster, or None.

    Classical CV on purpose: it is a fast, debuggable baseline. If it fails on
    your photos, swap in a segmentation model (e.g. SAM) here and keep the API.
    """
    h, w = img_rgb.shape[:2]
    scale = min(1.0, work_size / max(h, w))
    small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_rgb
    sh, sw = small.shape[:2]
    area_total = float(sh * sw)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [edges, otsu, 255 - otsu]

    best = None
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(c)
            if area < min_area_frac * area_total or area > max_area_frac * area_total:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if best is None or area > best[0]:
                    best = (area, approx.reshape(4, 2))
    if best is None:
        return None
    return order_points(best[1] / scale)


def warp_to_rect(img_rgb: np.ndarray, quad, long_side: int = 1536) -> np.ndarray:
    """Perspective-warp `quad` to an upright rectangle. Never upsamples beyond
    native resolution (upsampling would invent fake detail)."""
    q = order_points(quad)
    W = max(np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[1] - q[0]))
    H = max(np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[0] - q[3]))
    s = min(1.0, long_side / max(W, H))
    W, H = max(8, int(round(W * s))), max(8, int(round(H * s)))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img_rgb, M, (W, H), flags=cv2.INTER_CUBIC)


def rectify(img_rgb: np.ndarray, crop_margin: float = 0.02, long_side: int = 1536):
    """Detect + warp + trim the border. Returns (image, found).
    If no poster is found, returns the input unchanged and found=False."""
    quad = find_poster_quad(img_rgb)
    if quad is None:
        return img_rgb, False
    out = warp_to_rect(img_rgb, quad, long_side=long_side)
    if crop_margin > 0:
        h, w = out.shape[:2]
        my, mx = int(h * crop_margin), int(w * crop_margin)
        if h - 2 * my > 16 and w - 2 * mx > 16:
            out = out[my:h - my, mx:w - mx]
    return np.ascontiguousarray(out), True
