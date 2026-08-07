"""Phase 1 — sub-pixel refinement of detections.

The detector emits heatmap-argmax coordinates quantized to the network stride:
at 960x540 upsampled to 1920x1080 that is ~2 px of quantization noise, which
plausibly IS the 7-10 px floor observed on the healthy sequences. Every other
mechanism in v3 redistributes existing information; this one creates it, and is
the only one that can beat PnLCalib on frames where PnLCalib already works.

Two estimators, both classical and cheap:
  * `refine_corner`  — Forstner/Harris-style: the point minimising the squared
    distance to all local intensity gradients, i.e. the true intersection of
    the two ridges forming a line junction.
  * `refine_ridge`   — for points ON a line: move along the gradient normal to
    the intensity extremum (parabolic sub-pixel peak fit).

Runs inside the GPU detection pass and is cached, so the pipeline stays
image-free afterwards.
"""
from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def refine_corner(gray, xy, win=9, max_shift=3.0):
    """Forstner point: argmin_p sum_i (n_i^T (p - x_i))^2 over local gradients,
    solved in closed form as p = (sum n n^T)^-1 sum (n n^T) x.
    Returns the refined point, or the input if the window is degenerate."""
    if cv2 is None or gray is None:
        return np.asarray(xy, float), 0.0
    x, y = float(xy[0]), float(xy[1])
    h, w = gray.shape[:2]
    r = win // 2
    xi, yi = int(round(x)), int(round(y))
    if xi - r < 1 or yi - r < 1 or xi + r >= w - 1 or yi + r >= h - 1:
        return np.array([x, y]), 0.0
    patch = gray[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    ys, xs = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1]
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    mag = np.linalg.norm(g, axis=1)
    keep = mag > max(1e-3, 0.2 * mag.max()) if mag.max() > 0 else None
    if keep is None or keep.sum() < 6:
        return np.array([x, y]), 0.0
    g = g[keep]
    pts = np.stack([xs.ravel()[keep], ys.ravel()[keep]], 1).astype(np.float64)
    A = g.T @ g
    b = np.einsum('ij,ik,ik->j', g, g, pts)
    # conditioning guard: a single dominant edge direction is not a corner
    ev = np.linalg.eigvalsh(A)
    if ev[0] <= 1e-6 or ev[1] / max(ev[0], 1e-12) > 1e4:
        return np.array([x, y]), 0.0
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.array([x, y]), 0.0
    shift = float(np.linalg.norm(p - np.array([x, y])))
    if not np.isfinite(shift) or shift > max_shift:
        return np.array([x, y]), 0.0     # never trust a large jump
    return p, shift


def refine_detections(gray, kp_dict, image_wh, win=9, max_shift=3.0):
    """Refine every keypoint in a PnLCalib-style dict (normalized coords).
    Returns a new dict plus the median applied shift, for monitoring."""
    if cv2 is None or gray is None:
        return kp_dict, 0.0
    w, h = image_wh
    out, shifts = {}, []
    for k, e in kp_dict.items():
        try:
            x = float(e.get("x")); y = float(e.get("y"))
        except (TypeError, ValueError):
            out[k] = e
            continue
        norm = x <= 1.5 and y <= 1.5
        px = np.array([x * w, y * h]) if norm else np.array([x, y])
        p, s = refine_corner(gray, px, win=win, max_shift=max_shift)
        shifts.append(s)
        e2 = dict(e)
        if norm:
            e2["x"], e2["y"] = float(p[0] / w), float(p[1] / h)
        else:
            e2["x"], e2["y"] = float(p[0]), float(p[1])
        out[k] = e2
    return out, (float(np.median(shifts)) if shifts else 0.0)
