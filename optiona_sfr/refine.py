"""Layer 2 — upgraded PnLCalib refinement.

Cost (report Sec. 3): C = alpha * sum w * d_line  +  beta * sum w * d_Sampson(circle)
                        + (1 - alpha - beta) * sum c * d_kpt
All three terms are geometric pixel distances, so they are commensurate
(the deliberate fix over Broadcast2Pitch's algebraic conic residual).
Lines are gated by a projection-error threshold on their endpoints before
entering the line set, exactly as in PnLCalib Sec. III.

The conic term evaluates the image conic C' = H^{-T} C_w H^{-1} on
*undistortion-corrected* sample pixels, so it stays valid when k1 != 0.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from .geometry import (Camera, LINES_WORLD, CIRCLES_WORLD, circle_conic,
                       conic_image, sampson_conic_distance, point_line_distance,
                       param_bounds, clip_to_bounds, camera_is_plausible)


def _undistort_pixels(cam: Camera, uv):
    """Map observed (distorted) pixels to the pinhole (k1=0) image so they can
    be compared against homography-transformed conics."""
    if abs(cam.k1) < 1e-12:
        return np.atleast_2d(uv)
    Xg, valid = cam.backproject_ground(uv)
    pin = Camera(cam.f, cam.pan, cam.tilt, cam.roll, cam.C, k1=0.0,
                 image_wh=(cam.w, cam.h))
    uv2, _ = pin.project(Xg)
    uv = np.atleast_2d(uv).copy()
    uv[valid] = uv2[valid]
    return uv


# Cost charged per world point that falls behind the camera. Without it the
# optimizer can drive every observation behind the image plane, where masked
# residuals evaluate to zero — a perfect but meaningless minimum.
CHEIRALITY_W = 500.0


_BIG = 1e6      # finite stand-in for non-finite residuals


def _cheirality(z, front):
    """Per-point penalty, ALWAYS the same length as z (zero where in front).
    Behind-camera points must cost, or 'push the pitch behind the camera'
    becomes a zero-cost optimum."""
    return CHEIRALITY_W * np.where(front, 0.0, 1.0 + np.maximum(0.0, -z))


def _cheirality_penalty(z, front):
    """Backwards-compatible helper (returns None when nothing is behind)."""
    behind = ~front
    if not np.any(behind):
        return None
    return CHEIRALITY_W * (1.0 + np.maximum(0.0, -z[behind]))


class Residuals:
    def __init__(self, kp_obs, line_obs, circle_obs, cfg, image_wh,
                 optimize_k1=False, k1_fixed=0.0):
        (self.kp_idx, self.kp_uv, self.kp_conf, self.kp_X) = kp_obs
        self.line_obs = line_obs        # list of (line_idx, pts(2,2), conf)
        self.circle_obs = circle_obs    # list of (circle_idx, uv(N,2), conf(N,))
        self.cfg = cfg
        self.image_wh = image_wh
        self.optimize_k1 = optimize_k1
        self.k1_fixed = k1_fixed
        a, b = cfg.alpha_lines, (cfg.beta_conic if cfg.use_conic_term else 0.0)
        self.wa, self.wb, self.wk = a, b, max(1.0 - a - b, 1e-3)

    def gate_lines(self, cam: Camera):
        """Keep a detected line only if both projected world endpoints are
        within line_gate_px of the detection (PnLCalib's misdetection gate)."""
        kept = []
        for (li, pts, conf) in self.line_obs:
            A, B = LINES_WORLD[li - 1]
            prj, front = cam.project(np.stack([A, B]))
            if not front.all():
                continue
            d = point_line_distance(prj, pts[0], pts[1])
            if d.max() < self.cfg.line_gate_px:
                kept.append((li, pts, conf))
        self.line_obs = kept

    def __call__(self, v):
        """FIXED-LENGTH residual vector.

        scipy's least_squares requires the residual length to be constant.
        Appending a cheirality penalty only when points happen to be behind
        the camera (and skipping lines conditionally) makes it vary, which
        raises mid-optimization; a broad `except` then silently returned the
        UNREFINED camera. Every term is therefore emitted unconditionally and
        zeroed out when inactive.

        Layout: [2*Nkp keypoint | Nkp cheirality | per line: 2 dist + 2 cheir
                 | per circle group: Ni Sampson]
        """
        cam = Camera.from_vec(v, self.image_wh, k1_fixed=self.k1_fixed)
        cw = self.cfg.use_confidence_weights
        parts = []

        # --- keypoints: 2N reprojection + N cheirality -----------------------
        if len(self.kp_idx):
            prj, front, z = cam.project_full(self.kp_X)
            d = np.where(front[:, None], prj - self.kp_uv, 0.0)
            w = self.kp_conf if cw else np.ones_like(self.kp_conf)
            parts.append((self.wk * w[:, None] * d).ravel())
            parts.append(_cheirality(z, front))

        # --- lines: 2 point-to-line + 2 cheirality, per line -----------------
        for (li, pts, conf) in self.line_obs:
            A, B = LINES_WORLD[li - 1]
            prj, front, z = cam.project_full(np.stack([A, B]))
            if front.all():
                d = point_line_distance(pts, prj[0], prj[1])
            else:
                d = np.zeros(len(pts))
            parts.append(self.wa * (conf if cw else 1.0) * d)
            parts.append(_cheirality(z, front))

        # --- circles: Sampson distance per sampled point ---------------------
        if self.wb > 0:
            H = cam.homography_ground()
            for (ci, uv, conf) in self.circle_obs:
                if len(uv) == 0:
                    continue
                try:
                    Cimg = conic_image(circle_conic(*CIRCLES_WORLD[ci]), H)
                    uvu = _undistort_pixels(cam, uv)
                    d = sampson_conic_distance(Cimg, uvu)
                except np.linalg.LinAlgError:
                    d = np.full(len(uv), _BIG)      # degenerate H: penalise
                w = conf if cw else np.ones(len(uv))
                parts.append(self.wb * np.asarray(w) * d)

        if not parts:
            return np.array([_BIG])
        out = np.concatenate(parts)
        # A degenerate camera can produce NaN/inf; scipy cannot handle those,
        # so map them to a large finite cost (rejected, not crashed).
        return np.nan_to_num(out, nan=_BIG, posinf=_BIG, neginf=-_BIG)


def build_circle_obs(kp_idx, kp_uv, kp_conf, membership):
    """Group detected keypoints that lie on field circles into per-circle
    observation sets for the conic term (Route A of the report: no retraining)."""
    groups = {}
    for i, k in enumerate(kp_idx):
        ci = membership.get(int(k))
        if ci is None:
            continue
        groups.setdefault(ci, []).append(i)
    out = []
    for ci, ids in groups.items():
        ids = np.array(ids, int)
        out.append((ci, kp_uv[ids], kp_conf[ids]))
    return out


def refine_camera(cam: Camera, kp_obs, line_obs, circle_obs, cfg,
                  optimize_k1=False, robust=True, do_no_harm=True):
    """Nonlinear least-squares refinement of the camera from all cues.

    Two-stage schedule, each choice principled:
    1) soft_l1 warm-up on points + lines ONLY, wide scale. The Sampson conic
       distance is a first-order approximation — accurate near the conic but
       misleading from a far initialization (and the center circle is
       rotationally symmetric, so alone it under-constrains orientation).
    2) Lines re-gated with the warmed camera (gating with a bad init rejects
       everything), then full cost incl. conic under the configured robust
       loss (Cauchy) for the outlier-rejecting polish.
    """
    import dataclasses
    wh = (cam.w, cam.h)
    seed = cam
    if robust:
        # drop correspondences that disagree with the trusted seed
        kp_obs, line_obs, circle_obs, n_rej = select_inliers(
            cam, kp_obs, line_obs, circle_obs,
            k=getattr(cfg, "inlier_k", 3.0),
            min_px=getattr(cfg, "inlier_min_px", 12.0))
    x0 = clip_to_bounds(cam.to_vec(optimize_k1=optimize_k1))
    lo, hi = param_bounds(len(x0))
    try:
        cfg_warm = dataclasses.replace(cfg, use_conic_term=False)
        fn1 = Residuals(kp_obs, line_obs, [], cfg_warm, wh,
                        optimize_k1=optimize_k1, k1_fixed=cam.k1)
        fn1.gate_lines(cam)
        warm = least_squares(fn1, x0, method="trf", loss="soft_l1",
                             f_scale=max(5.0 * cfg.robust_fscale, 20.0),
                             max_nfev=cfg.max_nfev, x_scale="jac",
                             bounds=(lo, hi))
        cam_w = Camera.from_vec(warm.x, wh, k1_fixed=cam.k1)
        fn2 = Residuals(kp_obs, line_obs, circle_obs, cfg, wh,
                        optimize_k1=optimize_k1, k1_fixed=cam.k1)
        fn2.gate_lines(cam_w)
        sol = least_squares(fn2, warm.x, method="trf", loss=cfg.robust_loss,
                            f_scale=cfg.robust_fscale, max_nfev=cfg.max_nfev,
                            x_scale="jac", bounds=(lo, hi))
        out = Camera.from_vec(sol.x, wh, k1_fixed=cam.k1)
        if not camera_is_plausible(out):
            return (seed if do_no_harm and camera_is_plausible(seed) else None), \
                float("inf")
        if do_no_harm:
            # TRUST REGION. The seed is a good camera (5-14 px on real data),
            # so a large move cannot be an improvement — and a cost-based test
            # alone CANNOT detect harm caused by a wrong observation, because
            # it measures fit to that same wrong observation. With 1 line in 10
            # mislabelled, "better fit" meant 13.9 px -> 488.9 px. Bounding the
            # move in geometry space caps the damage whatever is corrupt.
            # ADAPTIVE trust region: trust the seed in proportion to how well
            # it explains its own observations. A seed fitting at 3-5 px (real
            # data) may only be nudged; a badly seeded frame may be moved far.
            s_fit = score_camera(seed, kp_obs, line_obs, cfg)
            tr = max(getattr(cfg, "trust_region_px", 30.0),
                     getattr(cfg, "trust_region_scale", 5.0) *
                     (s_fit if np.isfinite(s_fit) else 0.0))
            if projected_displacement(seed, out, X=kp_obs[3]) > tr:
                return seed, float("inf")
            c_seed = robust_cost(fn2(x0), cfg.robust_loss, cfg.robust_fscale)
            c_out = robust_cost(sol.fun, cfg.robust_loss, cfg.robust_fscale)
            if not np.isfinite(c_out) or c_out > c_seed:
                return seed, c_seed
            return out, c_out
        return out, float(np.mean(np.abs(sol.fun)))
    except Exception:
        return (seed if camera_is_plausible(seed) else None), float("inf")


# ----------------------------------------------------------------------------- robust inlier selection
def select_inliers(cam, kp_obs, line_obs, circle_obs, k=3.0, min_px=12.0,
                   min_kp=4):
    """Discard observations that disagree with a TRUSTED seed camera.

    Measured on realistic zoomed geometry: ONE mislabelled keypoint out of 9
    moves the refined camera from 2.6 px to 438.9 px, and two cause outright
    abstention. A robust loss only DOWNWEIGHTS such points; with few
    observations in a narrow field of view that is not enough. Since PnLCalib's
    estimate is already good (5-14 px on real data), residuals at the seed
    identify mislabelled correspondences directly, so we drop them outright.

    Threshold: max(min_px, k * MAD) — adaptive but never tighter than the
    detector's own noise floor.
    """
    kp_idx, kp_uv, kp_conf, kp_X = kp_obs
    keep_kp = np.ones(len(kp_idx), bool)
    if len(kp_idx):
        prj, front = cam.project(kp_X)
        d = np.full(len(kp_idx), np.inf)
        d[front] = np.linalg.norm(prj[front] - kp_uv[front], axis=1)
        finite = np.isfinite(d)
        if finite.sum() >= min_kp:
            med = np.median(d[finite])
            mad = np.median(np.abs(d[finite] - med)) * 1.4826
            thr = max(min_px, med + k * max(mad, 1e-6))
            keep_kp = finite & (d <= thr)
            if keep_kp.sum() < min_kp:      # keep the best few rather than none
                order = np.argsort(np.where(finite, d, np.inf))
                keep_kp = np.zeros(len(d), bool)
                keep_kp[order[:min(min_kp, finite.sum())]] = True
    kp_out = (kp_idx[keep_kp], kp_uv[keep_kp], kp_conf[keep_kp], kp_X[keep_kp])

    keep_lines = []
    for (li, pts, conf) in line_obs:
        A, B = LINES_WORLD[li - 1]
        prj, front = cam.project(np.stack([A, B]))
        if not front.all():
            continue
        d = point_line_distance(pts, prj[0], prj[1])
        # tighter than before: a mislabelled line can sit inside a loose gate
        # in a zoomed view, and lines carry the dominant weight (alpha=0.6)
        if np.isfinite(d).all() and d.max() <= max(min_px, 1.5 * min_px):
            keep_lines.append((li, pts, conf))

    keep_circ = []
    for (ci, uv, conf) in circle_obs:
        if len(uv) == 0:
            continue
        H = cam.homography_ground()
        try:
            Cimg = conic_image(circle_conic(*CIRCLES_WORLD[ci]), H)
            d = sampson_conic_distance(Cimg, _undistort_pixels(cam, uv))
        except np.linalg.LinAlgError:
            continue
        m = np.isfinite(d) & (d <= max(min_px, 3.0 * min_px))
        if m.sum() >= 5:
            keep_circ.append((ci, uv[m], np.asarray(conf)[m]))
    return kp_out, keep_lines, keep_circ, int((~keep_kp).sum())


def projected_displacement(cam_a, cam_b, n=60, X=None):
    """Mean pixel displacement between two cameras, measured WHERE THE
    EVIDENCE IS.

    Passing the observed world points (X) matters for zoomed views: a full-
    pitch grid extrapolates far outside the field of view, where small pose
    changes are hugely amplified, so a grid-based bound is far too tight for a
    legitimate correction. Falls back to a grid when there are too few points.
    """
    if cam_a is None or cam_b is None:
        return np.inf
    if X is not None and len(np.atleast_2d(X)) >= 4:
        G = np.atleast_2d(np.asarray(X, float))
    else:
        g = np.linspace(-45, 45, n)
        G = np.column_stack([g, np.linspace(-28, 28, n), np.zeros(n)])
    pa, fa = cam_a.project(G)
    pb, fb = cam_b.project(G)
    m = fa & fb & np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    if m.sum() < 5:
        return np.inf
    return float(np.mean(np.linalg.norm(pa[m] - pb[m], axis=1)))


def robust_cost(res, loss="cauchy", f_scale=4.0):
    """The scalar scipy actually minimises. Comparing this at the seed and at
    the solution is the correct acceptance test: a MEDIAN residual is far too
    coarse (with ~10 observations it barely moves), which made the do-no-harm
    guard reject every genuine improvement."""
    r = np.asarray(res, float)
    z = (r / max(f_scale, 1e-9)) ** 2
    if loss == "cauchy":
        rho = np.log1p(z)
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = np.where(z <= 1, z, 2.0 * np.sqrt(z) - 1.0)
    else:
        rho = z
    return 0.5 * float(np.sum(rho)) * max(f_scale, 1e-9) ** 2


def score_camera(cam, kp_obs, line_obs, cfg):
    """Median geometric residual of a camera on the given observations."""
    if cam is None:
        return np.inf
    vals = []
    kp_idx, kp_uv, _, kp_X = kp_obs
    if len(kp_idx):
        prj, front = cam.project(kp_X)
        if front.any():
            vals.append(np.linalg.norm(prj[front] - kp_uv[front], axis=1))
    for (li, pts, _c) in line_obs:
        A, B = LINES_WORLD[li - 1]
        prj, front = cam.project(np.stack([A, B]))
        if front.all():
            vals.append(point_line_distance(pts, prj[0], prj[1]))
    if not vals:
        return np.inf
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.inf
