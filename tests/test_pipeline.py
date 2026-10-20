import cv2
import numpy as np

from posterdetect.features import tile_image
from posterdetect.metrics import aggregate_by_sample, summarize
from posterdetect.recapture import RecaptureSimulator
from posterdetect.rectify import find_poster_quad, order_points, rectify


def _scene():
    rng = np.random.default_rng(0)
    poster = cv2.GaussianBlur((rng.random((300, 200, 3)) * 255).astype(np.uint8), (0, 0), 3)
    cv2.rectangle(poster, (20, 20), (180, 120), (255, 50, 50), -1)
    bg = np.full((600, 800, 3), (90, 110, 90), np.uint8)
    src = np.float32([[0, 0], [199, 0], [199, 299], [0, 299]])
    dst = np.float32([[250, 100], [520, 130], [505, 470], [235, 440]])
    M = cv2.getPerspectiveTransform(src, dst)
    scene = cv2.warpPerspective(poster, M, (800, 600), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    return scene, dst


def test_order_points():
    pts = [[10, 10], [0, 10], [0, 0], [10, 0]]
    assert np.allclose(order_points(pts), [[0, 0], [10, 0], [10, 10], [0, 10]])


def test_find_quad_corners():
    scene, dst = _scene()
    quad = find_poster_quad(scene)
    assert quad is not None
    assert np.abs(quad - dst).max() < 20


def test_rectify_aspect():
    scene, _ = _scene()
    out, found = rectify(scene)
    assert found
    h, w = out.shape[:2]
    assert 0.5 < w / h < 0.9


def test_rectify_fallback():
    flat = np.full((200, 300, 3), 128, np.uint8)
    out, found = rectify(flat)
    assert not found and out.shape == flat.shape


def test_recapture():
    img = (np.random.default_rng(1).random((256, 384, 3)) * 255).astype(np.uint8)
    sim = RecaptureSimulator(seed=0)
    outs = [sim(img) for _ in range(20)]
    for o in outs:
        assert o.shape == img.shape and o.dtype == np.uint8
    assert np.mean([np.abs(o.astype(int) - img).mean() for o in outs]) > 1.0
    # deterministic given seed
    assert np.array_equal(RecaptureSimulator(seed=3)(img), RecaptureSimulator(seed=3)(img))


def test_tiles():
    img = np.zeros((700, 500, 3), np.uint8)
    t = tile_image(img, tile=224, long_side=896, max_tiles=16)
    assert t.shape[1:] == (224, 224, 3) and 1 <= len(t) <= 16
    small = np.zeros((50, 60, 3), np.uint8)
    assert tile_image(small, long_side=100).shape[1:] == (224, 224, 3)


def test_metrics():
    idx = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    s = np.array([2., 2., -2., -2., 1., 1., -1., -1.])
    agg = aggregate_by_sample(idx, s, 4)
    m = summarize(np.array([1, 0, 1, 0]), agg)
    assert m["auc"] == 1.0
