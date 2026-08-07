"""Layer 3 — BroadTrack-style temporal tracking (paper Sec. 3.2).

Per frame t, starting from kappa_{t-1}, minimize
    L = L_F + L_OF + omega * L_T
where L_F is the marking (keypoint/line/conic) term reused from refine.py
with Cauchy robust loss, L_OF the optical-flow term (correspondences on the
field surface, back-projected to Z=0 with kappa_{t-1}, reprojected with
kappa_t; player boxes masked out when provided), and L_T the tripod term
|delta - ||O* - T||| with O* the closest point of the optical axis to T.

Confidence s = Jaccard(detected-marking mask, projected-template mask);
when s < 0.5 the tracker reinitializes from the single-frame calibrator
(PnLCalib heuristic_voting -> upgraded refinement), as BroadTrack does.
"""
from __future__ import annotations
import numpy as np

from .geometry import (Camera, LINES_WORLD, CIRCLES_WORLD, param_bounds,
                       clip_to_bounds, camera_is_plausible)
from .refine import Residuals, refine_camera
from scipy.optimize import least_squares

try:
    import cv2
except Exception:                       # geometry-only environments / tests
    cv2 = None



# ----------------------------------------------------------------------------- safe drawing
_CV_LIMIT = 1 << 20     # clamp coordinates; cv2 clips internally, but huge or
                        # non-finite values from a degenerate camera must never
                        # reach it (and some cv2/numpy builds reject np scalars)

def _pt(p):
    """Coerce one 2D point to a plain-Python int tuple, or None if unusable."""
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (int(max(-_CV_LIMIT, min(_CV_LIMIT, x))),
            int(max(-_CV_LIMIT, min(_CV_LIMIT, y))))


def _safe_line(m, a, b, color, thickness):
    pa, pb = _pt(a), _pt(b)
    if pa is not None and pb is not None:
        cv2.line(m, pa, pb, color, thickness)


def _int32_poly(prj):
    """Finite rows only, int32, clamped — the dtype cv2.polylines expects."""
    prj = np.asarray(prj, float)
    keep = np.isfinite(prj).all(axis=1)
    prj = np.clip(prj[keep], -_CV_LIMIT, _CV_LIMIT)
    return prj.astype(np.int32)


# ----------------------------------------------------------------------------- confidence
def _template_mask(cam: Camera, shape, thickness=5):
    m = np.zeros(shape, np.uint8)
    for A, B in LINES_WORLD:
        prj, front = cam.project(np.stack([A, B]))
        if front.all():
            _safe_line(m, prj[0], prj[1], 255, thickness)
    for (cx, cy, r) in CIRCLES_WORLD:
        ang = np.linspace(0, 2 * np.pi, 90)
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang),
                        np.zeros_like(ang)], axis=1)
        prj, front = cam.project(pts)
        prj = _int32_poly(prj[front])
        if len(prj) > 2:
            cv2.polylines(m, [prj], False, 255, thickness)
    return m


def _detection_mask(kp_uv, line_obs, shape, dilate=7):
    m = np.zeros(shape, np.uint8)
    for uv in np.atleast_2d(kp_uv):
        p = _pt(uv)
        if p is not None and 0 <= p[0] < shape[1] and 0 <= p[1] < shape[0]:
            cv2.circle(m, p, 2, 255, -1)
    for (_li, pts, _c) in line_obs:
        _safe_line(m, pts[0], pts[1], 255, 3)
    if dilate > 0:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def jaccard_confidence(cam, kp_uv, line_obs, image_wh, dilate=7):
    """Adaptation of BroadTrack Sec. 3.2.4 to sparse detections. BroadTrack
    intersects a DENSE segmentation of markings with the projected template;
    with sparse keypoint/line detections a raw IoU is structurally low even
    for a perfect camera (undetected elements dominate the union). We
    therefore score the precision of detections against the template,
    s = |D intersect T| / |D|: high iff the detections lie ON the projected
    template, low for a wrong camera — preserving the semantics of the
    s < 0.5 reinitialization rule."""
    if cv2 is None:
        return 1.0
    shape = (image_wh[1], image_wh[0])
    a = _detection_mask(kp_uv, line_obs, shape, dilate)
    b = cv2.dilate(_template_mask(cam, shape), np.ones((dilate, dilate), np.uint8))
    da = (a > 0)
    inter = np.logical_and(da, b > 0).sum()
    return inter / da.sum() if da.sum() else 0.0


# ----------------------------------------------------------------------------- optical flow
def flow_from_pairs(cam_prev: Camera, pair, cfg):
    """Turn cached pixel-space correspondences into (X_world, uv_t) using the
    previous camera. Identical math to the live path — only the source of the
    correspondences differs — so cached and live runs agree exactly."""
    if pair is None:
        return np.zeros((0, 3)), np.zeros((0, 2))
    p0, p1 = pair
    if len(p0) < cfg.of_min_points:
        return np.zeros((0, 3)), np.zeros((0, 2))
    Xw, valid = cam_prev.backproject_ground(p0)
    # keep only correspondences that land on the pitch surface
    inside = valid & (np.abs(Xw[:, 0]) < 60.0) & (np.abs(Xw[:, 1]) < 40.0)
    return Xw[inside], np.asarray(p1)[inside]


def flow_correspondences(prev_gray, gray, cam_prev: Camera, cfg,
                         player_boxes=None):
    """Live path: Shi-Tomasi points inside the projected pitch polygon,
    tracked with LK, back-projected to Z=0 with kappa_{t-1}."""
    if cv2 is None or prev_gray is None:
        return np.zeros((0, 3)), np.zeros((0, 2))
    h, w = prev_gray.shape[:2]
    corners_w = np.array([[-52.5, -34, 0], [52.5, -34, 0],
                          [52.5, 34, 0], [-52.5, 34, 0]], float)
    prj, front = cam_prev.project(corners_w)
    mask = np.zeros((h, w), np.uint8)
    if front.all():
        cv2.fillPoly(mask, [_int32_poly(prj)], 255)
    else:
        mask[:] = 255
    if player_boxes is not None and cfg.mask_players:
        for (x1, y1, x2, y2) in player_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, cfg.of_max_points, 0.01, 12, mask=mask)
    if p0 is None or len(p0) < cfg.of_min_points:
        return np.zeros((0, 3)), np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                         winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    p0, p1 = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    if len(p0) < cfg.of_min_points:
        return np.zeros((0, 3)), np.zeros((0, 2))
    Xw, valid = cam_prev.backproject_ground(p0)
    return Xw[valid], p1[valid]


# ----------------------------------------------------------------------------- tracker
class BroadTrackLayer:
    def __init__(self, refine_cfg, tracker_cfg, image_wh):
        self.rcfg, self.tcfg = refine_cfg, tracker_cfg
        self.image_wh = image_wh
        self.cam: Camera | None = None
        self.prev_gray = None
        self.n_reinit = 0

    # -- combined per-frame objective ---------------------------------------
    def _objective(self, kp_obs, line_obs, circle_obs, of_X, of_uv):
        marking = Residuals(kp_obs, line_obs, circle_obs, self.rcfg,
                            self.image_wh, optimize_k1=self.tcfg.optimize_k1,
                            k1_fixed=self.cam.k1)
        marking.gate_lines(self.cam)
        T = np.asarray(self.tcfg.tripod_T, float)
        delta = float(self.tcfg.tripod_delta)
        use_T = self.tcfg.use_tripod
        omega = self.tcfg.omega_tripod

        from .refine import _cheirality, _BIG

        def fn(v):
            # FIXED length: marking block + (2*Nof + Nof) + (1 if tripod).
            res = [marking(v)]
            cam = Camera.from_vec(v, self.image_wh, k1_fixed=self.cam.k1)
            if len(of_X):                                   # L_OF (Eq. 4)
                prj, front, z = cam.project_full(of_X)
                d = np.where(front[:, None], prj - of_uv, 0.0)
                res.append(d.ravel())
                res.append(_cheirality(z, front))
            if use_T:                                       # L_T (Eq. 5-6)
                r3 = cam.R[2, :]
                Ostar = cam.C + np.dot(T - cam.C, r3) * r3
                res.append(np.array([omega * (delta - np.linalg.norm(Ostar - T))]))
            out = np.concatenate(res)
            return np.nan_to_num(out, nan=_BIG, posinf=_BIG, neginf=-_BIG)
        return fn

    def _reinit(self, single_frame_calibrate, frame_ctx):
        cam = single_frame_calibrate(frame_ctx)
        if cam is not None and not camera_is_plausible(cam):
            cam = None
        if cam is not None:
            if self.cam is not None:
                cam.k1 = self.cam.k1        # carry k1 over reinitialization
            self.cam = cam
            self.n_reinit += 1
        return self.cam

    def step(self, kp_obs, line_obs, circle_obs, gray=None, player_boxes=None,
             single_frame_calibrate=None, frame_ctx=None, flow_pair=None):
        """Process one frame; returns (Camera|None, confidence s)."""
        kp_idx, kp_uv, kp_conf, kp_X = kp_obs
        if self.cam is None:
            if single_frame_calibrate is None:
                return None, 0.0
            self._reinit(single_frame_calibrate, frame_ctx)
            if self.cam is None:
                return None, 0.0
        # optical flow from previous frame
        of_X, of_uv = (np.zeros((0, 3)), np.zeros((0, 2)))
        if self.tcfg.use_optical_flow:
            if flow_pair is not None:            # cached (image-free) path
                of_X, of_uv = flow_from_pairs(self.cam, flow_pair, self.tcfg)
            elif gray is not None:               # live path
                of_X, of_uv = flow_correspondences(self.prev_gray, gray,
                                                   self.cam, self.tcfg,
                                                   player_boxes)
        # Minimum-support gate: with too few geometric cues the per-frame
        # update would overfit noise and jump; hold the warm-started camera
        # instead (optical-flow correspondences count as support since they
        # constrain the same parameters).
        support = (2 * len(kp_obs[0]) + 2 * len(line_obs)
                   + sum(len(c[1]) for c in circle_obs) + len(of_X))
        if support < self.tcfg.min_support:
            s = jaccard_confidence(self.cam, kp_uv, line_obs, self.image_wh,
                                   self.tcfg.jaccard_dilate_px)
            if gray is not None:
                self.prev_gray = gray
            return self.cam, s
        # warm-started, two-stage robust LM update (soft_l1 -> Cauchy)
        fn = self._objective(kp_obs, line_obs, circle_obs, of_X, of_uv)
        x0 = clip_to_bounds(self.cam.to_vec(optimize_k1=self.tcfg.optimize_k1))
        lo, hi = param_bounds(len(x0))
        try:
            warm = least_squares(fn, x0, method="trf", loss="soft_l1",
                                 f_scale=max(5.0 * self.rcfg.robust_fscale, 20.0),
                                 max_nfev=self.rcfg.max_nfev, x_scale="jac",
                                 bounds=(lo, hi))
            sol = least_squares(fn, warm.x, method="trf",
                                loss=self.rcfg.robust_loss,
                                f_scale=self.rcfg.robust_fscale,
                                max_nfev=self.rcfg.max_nfev, x_scale="jac",
                                bounds=(lo, hi))
            cand = Camera.from_vec(sol.x, self.image_wh, k1_fixed=self.cam.k1)
            if camera_is_plausible(cand):      # never accept a runaway solution
                self.cam = cand
        except Exception:
            pass
        # confidence + reinitialization (BroadTrack: s < 0.5)
        s = jaccard_confidence(self.cam, kp_uv, line_obs, self.image_wh,
                               self.tcfg.jaccard_dilate_px)
        if s < self.tcfg.reinit_below_s and single_frame_calibrate is not None:
            self._reinit(single_frame_calibrate, frame_ctx)
            s = jaccard_confidence(self.cam, kp_uv, line_obs, self.image_wh,
                                   self.tcfg.jaccard_dilate_px)
        if gray is not None:
            self.prev_gray = gray
        return self.cam, s


# ----------------------------------------------------------------------------- tripod estimation
def estimate_tripod(cams):
    """BroadTrack Sec. 4.1: from an unconstrained pass, derive T and delta.
    T is fit as the point minimizing distance to all optical axes (least
    squares over line-to-point distances); delta is the mean residual axis
    distance."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for cam in cams:
        r3 = cam.R[2, :] / np.linalg.norm(cam.R[2, :])
        P = np.eye(3) - np.outer(r3, r3)     # projector orthogonal to the axis
        A += P
        b += P @ cam.C
    T = np.linalg.lstsq(A, b, rcond=None)[0]
    ds = []
    for cam in cams:
        r3 = cam.R[2, :] / np.linalg.norm(cam.R[2, :])
        Ostar = cam.C + np.dot(T - cam.C, r3) * r3
        ds.append(np.linalg.norm(Ostar - T))
    return T, float(np.mean(ds))
