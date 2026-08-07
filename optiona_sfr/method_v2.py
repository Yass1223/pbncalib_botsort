"""Method v2 — fixed optical centre, 4 DOF per frame, temporal smoothing.

Rationale from the real-data evidence (see README v0.6.0):

* the audited camera centres barely move across a sequence
  (frame 1: (-1.35, 55.09, -9.67); frame 300: (0.01, 50.17, -9.50)) — a
  broadcast main camera is tripod-mounted, so C is physically CONSTANT;
* yet v1 estimated 8 DOF independently per frame, which admits the classic
  focal/distance degeneracy: "further away + longer lens" costs almost the
  same as "closer + shorter lens". Measured consequences were extreme —
  focal_travel 39,475 m per sequence, jitter_f 5512 px, tripod delta 48.5 m,
  and each ADDED degree of freedom made things worse (A4 +k1 worse than A3);
* a soft tripod penalty (BroadTrack's L_T) was too weak to remove it.

v2 therefore:
  1. estimates ONE centre per sequence, robustly, and holds it FIXED;
  2. solves only (pan, tilt, roll, f) per frame — 4 DOF, no degeneracy;
  3. lets PnLCalib do what it is already good at (native point-and-line
     refinement) instead of substituting our own objective;
  4. smooths the 4-DOF trajectory with outlier rejection, and interpolates
     frames where calibration failed — 27% of frames in v1 had no camera at
     all, and interpolating a smooth 4-DOF trajectory through a fixed centre
     is well posed.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from .geometry import Camera, camera_is_plausible

# 4-DOF bounds: [pan, tilt, roll, f] (+ optional k1)
V2_LO = np.array([-2 * np.pi, 0.05, -np.pi / 3, 100.0, -0.6])
V2_HI = np.array([2 * np.pi, np.pi - 0.05, np.pi / 3, 30000.0, 0.6])


# ----------------------------------------------------------------------------- fixed centre
def estimate_fixed_center(cams, weights=None, trim=0.2, verbose=True):
    """One optical centre for the whole sequence.

    Uses a trimmed geometric median (L1), which ignores the wild outliers that
    a per-frame 8-DOF fit produces while staying unbiased for the bulk.
    Returns (C, spread_m) where spread is the median absolute deviation — a
    direct measure of how badly the per-frame estimates disagreed.
    """
    P = np.array([c.C for c in cams if c is not None and np.isfinite(c.C).all()])
    if len(P) == 0:
        return None, np.inf
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)[:len(P)]
    med = np.median(P, axis=0)
    d = np.linalg.norm(P - med, axis=1)
    keep = d <= np.quantile(d, 1.0 - trim) if len(P) > 10 else np.ones(len(P), bool)
    P, w = P[keep], w[keep]
    # Weiszfeld iterations for the weighted geometric median
    C = np.average(P, axis=0, weights=w)
    for _ in range(64):
        r = np.linalg.norm(P - C, axis=1)
        r = np.maximum(r, 1e-6)
        Cn = np.sum(P * (w / r)[:, None], axis=0) / np.sum(w / r)
        if np.linalg.norm(Cn - C) < 1e-6:
            C = Cn
            break
        C = Cn
    spread = float(np.median(np.linalg.norm(P - C, axis=1)))
    if verbose:
        print(f"[v2] fixed centre C = {np.round(C, 2)} m "
              f"(from {len(P)} frames, median deviation {spread:.2f} m)")
    return C, spread


# ----------------------------------------------------------------------------- 4-DOF fit
class FixedCenterObjective:
    """Wraps a full-parameter residual so only (pan, tilt, roll, f[, k1]) vary."""

    def __init__(self, base, C, image_wh, optimize_k1=False, k1=0.0):
        self.base = base
        self.C = np.asarray(C, float)
        self.image_wh = image_wh
        self.optimize_k1 = optimize_k1
        self.k1 = k1

    def n(self):
        return 5 if self.optimize_k1 else 4

    def to_vec(self, cam):
        v = [cam.pan, cam.tilt, cam.roll, cam.f]
        if self.optimize_k1:
            v.append(cam.k1)
        return np.array(v, float)

    def to_cam(self, v):
        k1 = v[4] if self.optimize_k1 else self.k1
        return Camera(f=v[3], pan=v[0], tilt=v[1], roll=v[2], C=self.C,
                      k1=k1, image_wh=self.image_wh)

    def full_vec(self, v):
        k1 = v[4] if self.optimize_k1 else self.k1
        return np.array([v[0], v[1], v[2], v[3], *self.C, k1])

    def __call__(self, v):
        return self.base(self.full_vec(v))


def refine_fixed_center(cam, kp_obs, line_obs, circle_obs, cfg, C,
                        optimize_k1=False, extra=None, robust=True,
                        do_no_harm=True):
    """Per-frame 4-DOF refinement about a FIXED optical centre.

    Uses the same robust machinery as v1: mislabelled correspondences are
    rejected against the trusted seed, and a refinement that fits the inliers
    worse than the seed is discarded (one bad keypoint in nine was enough to
    turn a 2.6 px camera into a 439 px one)."""
    from .refine import (Residuals, select_inliers, robust_cost,
                         projected_displacement)
    wh = (cam.w, cam.h)
    seed = Camera(cam.f, cam.pan, cam.tilt, cam.roll, C, cam.k1, wh)
    if robust:
        kp_obs, line_obs, circle_obs, _ = select_inliers(
            seed, kp_obs, line_obs, circle_obs,
            k=getattr(cfg, "inlier_k", 3.0),
            min_px=getattr(cfg, "inlier_min_px", 12.0))
    base = Residuals(kp_obs, line_obs, circle_obs, cfg, wh,
                     optimize_k1=True, k1_fixed=cam.k1)
    base.gate_lines(Camera(cam.f, cam.pan, cam.tilt, cam.roll, C, cam.k1, wh))
    obj = FixedCenterObjective(base, C, wh, optimize_k1, cam.k1)
    if extra is not None:
        inner = obj

        def combined(v):
            return np.concatenate([inner(v), extra(inner.to_cam(v))])
        combined.to_cam = inner.to_cam
        combined.to_vec = inner.to_vec
        combined.n = inner.n
        obj = combined
    n = obj.n()
    lo, hi = V2_LO[:n].copy(), V2_HI[:n].copy()
    x0 = np.clip(obj.to_vec(cam), lo + 1e-9, hi - 1e-9)
    try:
        warm = least_squares(obj, x0, method="trf", loss="soft_l1",
                             f_scale=max(5.0 * cfg.robust_fscale, 20.0),
                             max_nfev=cfg.max_nfev, x_scale="jac",
                             bounds=(lo, hi))
        sol = least_squares(obj, warm.x, method="trf", loss=cfg.robust_loss,
                            f_scale=cfg.robust_fscale, max_nfev=cfg.max_nfev,
                            x_scale="jac", bounds=(lo, hi))
        out = obj.to_cam(sol.x)
        if not camera_is_plausible(out):
            return (seed if do_no_harm and camera_is_plausible(seed) else None), \
                float("inf")
        if do_no_harm:
            from .refine import score_camera
            s_fit = score_camera(seed, kp_obs, line_obs, cfg)
            tr = max(getattr(cfg, "trust_region_px", 30.0),
                     getattr(cfg, "trust_region_scale", 5.0) *
                     (s_fit if np.isfinite(s_fit) else 0.0))
            if projected_displacement(seed, out, X=kp_obs[3]) > tr:
                return seed, float("inf")
            c_seed = robust_cost(obj(x0), cfg.robust_loss, cfg.robust_fscale)
            c_out = robust_cost(sol.fun, cfg.robust_loss, cfg.robust_fscale)
            if not np.isfinite(c_out) or c_out > c_seed:
                return seed, c_seed
            return out, c_out
        return out, float(np.mean(np.abs(sol.fun)))
    except Exception:
        return (seed if camera_is_plausible(seed) else None), float("inf")


# ----------------------------------------------------------------------------- temporal
def _robust_reject(x, ok, k=4.0, win=15):
    """Flag samples deviating from a running median by > k * MAD."""
    x = np.asarray(x, float)
    n = len(x)
    good = ok.copy()
    half = max(win // 2, 1)
    for i in range(n):
        if not ok[i]:
            continue
        a, b = max(0, i - half), min(n, i + half + 1)
        seg = x[a:b][ok[a:b]]
        if len(seg) < 5:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) * 1.4826
        if mad > 1e-9 and abs(x[i] - med) > k * mad:
            good[i] = False
    return good


def smooth_and_interpolate(frames, pan, tilt, roll, f, ok, window=15,
                           polyorder=2, reject_k=4.0, verbose=False):
    """Robust smoothing + gap filling of the 4-DOF trajectory.

    Pan is unwrapped before smoothing so wrap-around does not create a false
    discontinuity. Outliers are rejected against a running median/MAD, then all
    missing frames (failed calibration + rejected samples) are interpolated,
    then a Savitzky-Golay filter is applied. Returns arrays plus the final
    validity mask and the interpolation mask.
    """
    from scipy.signal import savgol_filter
    frames = np.asarray(frames)
    ok = np.asarray(ok, bool).copy()
    series = {"pan": np.unwrap(np.asarray(pan, float)),
              "tilt": np.asarray(tilt, float),
              "roll": np.asarray(roll, float),
              "f": np.asarray(f, float)}
    for name, x in series.items():
        xf = np.where(ok, x, np.nan)
        fill = np.nanmedian(xf) if np.isfinite(xf).any() else 0.0
        ok = ok & _robust_reject(np.nan_to_num(xf, nan=fill), ok,
                                 k=reject_k, win=window)
    if ok.sum() < max(5, polyorder + 2):
        return series["pan"], series["tilt"], series["roll"], series["f"], ok, \
            np.zeros(len(frames), bool)
    interpolated = ~ok
    out = {}
    idx = np.arange(len(frames))
    for name, x in series.items():
        y = np.interp(idx, idx[ok], x[ok])
        w = min(window if window % 2 == 1 else window + 1, len(y) - (1 - len(y) % 2))
        if w >= polyorder + 2:
            y = savgol_filter(y, w, polyorder)
        out[name] = y
    if verbose:
        print(f"[v2] smoothing: {int(interpolated.sum())}/{len(frames)} frames "
              "interpolated")
    return (out["pan"], out["tilt"], out["roll"], out["f"],
            np.ones(len(frames), bool), interpolated)


# ----------------------------------------------------------------------------- flow (rotation-only regime)
def flow_residual_factory(C, image_wh, flow_pair, cam_prev):
    """With C FIXED, frame-to-frame ground correspondences constrain only
    rotation and zoom, which is far better conditioned than the v1 case where
    an uncertain centre made back-projection unreliable. Fixed length:
    2N reprojection + N cheirality."""
    if flow_pair is None or cam_prev is None:
        return None
    from .refine import _cheirality
    p0, p1 = flow_pair
    if len(p0) < 8:
        return None
    Xw, valid = cam_prev.backproject_ground(np.asarray(p0, float))
    inside = valid & (np.abs(Xw[:, 0]) < 60.0) & (np.abs(Xw[:, 1]) < 40.0)
    Xw, uv = Xw[inside], np.asarray(p1, float)[inside]
    if len(Xw) < 8:
        return None

    def fn(cam):
        prj, front, z = cam.project_full(Xw)
        d = np.where(front[:, None], prj - uv, 0.0)
        return np.concatenate([d.ravel(), _cheirality(z, front)])
    return fn


# ----------------------------------------------------------------------------- sequence
def run_sequence_v2(seq_dir, exp_cfg, kp_world, membership, verbose=False):
    """Two-pass v2 pipeline. Returns a DataFrame with the v1 schema so all
    existing metrics and summaries work unchanged."""
    import pandas as pd
    from pathlib import Path
    from .detection import load_cached, get_baseline_cameras
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .refine import build_circle_obs
    from .experiments import (load_pitch_annotations, reprojection_error,
                              cached_sequence_ids)

    v2 = exp_cfg.v2
    paths = exp_cfg.paths
    seq_dir = Path(seq_dir)
    seq_id = seq_dir.name
    from .experiments import frame_indices
    idxs = frame_indices(paths.cache_dir, seq_id)
    if not idxs:
        raise RuntimeError(f"No detection cache for {seq_id}")
    first = load_cached(paths.cache_dir, seq_id, idxs[0])
    image_wh = tuple(first["wh"])
    ann = load_pitch_annotations(seq_dir, getattr(paths, "labels_dir", None),
                                 image_wh=image_wh)
    # ---- pass 1: PnLCalib per frame (cached once per sequence) ----------
    cams = get_baseline_cameras(paths, seq_id, image_wh, exp_cfg.refine,
                                kp_world, membership,
                                refine_lines=v2.native_refine_lines,
                                frames=idxs, verbose=verbose)
    obs, conf = {}, {}
    for fi in idxs:
        det = load_cached(paths.cache_dir, seq_id, fi)
        if det is None:
            continue
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        circ = build_circle_obs(kp_obs[0], kp_obs[1], kp_obs[2], membership)
        obs[fi] = (kp_obs, line_obs, circ, det.get("flow"))
        conf[fi] = float(len(kp_obs[0]) + 2 * len(line_obs))

    # ---- fixed centre, with a SEQUENCE-LEVEL do-no-harm test -------------
    from .refine import score_camera
    quality = {}
    for fi in idxs:
        cam = cams.get(fi)
        if cam is None:
            continue
        kp_obs, line_obs, _c, _f = obs[fi]
        quality[fi] = score_camera(cam, kp_obs, line_obs, exp_cfg.refine)

    C, spread = (None, np.inf)
    if v2.fix_center:
        # weight by FIT QUALITY, not merely by how many features were seen:
        # a frame whose camera fits its own observations badly should not
        # drag the sequence centre.
        keys = [i for i in idxs if cams.get(i) is not None]
        good = [cams[i] for i in keys]
        w = [1.0 / (1.0 + max(quality.get(i, np.inf), 0.0)) for i in keys]
        C, spread = estimate_fixed_center(good, w, verbose=verbose)

        # Adopting a single centre is a strong assumption. Verify it on a
        # sample of frames: refit 4 DOF about C and compare the fit to the
        # per-frame cameras. If it is clearly worse, KEEP the per-frame
        # centres — measured on real data, a contaminated centre turned a
        # 10.3 px sequence into 369.7 px.
        sample = [i for i in keys[::max(1, len(keys) // 25)]][:25]
        if C is not None and sample:
            before, after = [], []
            for fi in sample:
                kp_obs, line_obs, circ, _f = obs[fi]
                base = cams[fi]
                seedC = Camera(base.f, base.pan, base.tilt, base.roll, C,
                               base.k1, image_wh)
                out, _ = refine_fixed_center(seedC, kp_obs, line_obs, circ,
                                             exp_cfg.refine, C)
                if out is None:
                    continue
                before.append(quality.get(fi, np.inf))
                after.append(score_camera(out, kp_obs, line_obs, exp_cfg.refine))
            if before:
                b, a = float(np.median(before)), float(np.median(after))
                ok = a <= max(2.0 * b, b + 5.0)
                if verbose or not ok:
                    print(f"[v2] fixed-centre check: per-frame {b:.1f} px -> "
                          f"fixed {a:.1f} px "
                          f"({'ADOPTED' if ok else 'REJECTED, keeping per-frame centres'})")
                if not ok:
                    C = None

    # ---- pass 2: 4-DOF refit --------------------------------------------
    prev = None
    for fi in idxs:
        cam = cams.get(fi)
        if C is None or not v2.refit_4dof:
            prev = cam if cam is not None else prev
            continue
        seed = cam if cam is not None else prev
        if seed is None:
            continue
        seed = Camera(seed.f, seed.pan, seed.tilt, seed.roll, C, seed.k1,
                      image_wh)
        kp_obs, line_obs, circ, flow = obs[fi]
        extra = (flow_residual_factory(C, image_wh, flow, prev)
                 if v2.use_flow else None)
        out, _ = refine_fixed_center(seed, kp_obs, line_obs, circ,
                                     exp_cfg.refine, C,
                                     optimize_k1=v2.optimize_k1,
                                     extra=extra)
        cams[fi] = out if out is not None else cam
        prev = cams[fi] if cams[fi] is not None else prev

    # ---- temporal smoothing / interpolation ------------------------------
    ok = np.array([cams.get(i) is not None for i in idxs])
    getv = lambda a: np.array([getattr(cams[i], a) if cams.get(i) is not None
                               else np.nan for i in idxs])
    pan, tilt, roll, fo = getv("pan"), getv("tilt"), getv("roll"), getv("f")
    interp = np.zeros(len(idxs), bool)
    if v2.smooth and ok.sum() > 10:
        pan, tilt, roll, fo, ok_s, interp = smooth_and_interpolate(
            idxs, np.nan_to_num(pan), np.nan_to_num(tilt),
            np.nan_to_num(roll), np.nan_to_num(fo), ok,
            window=v2.window, polyorder=v2.polyorder, verbose=verbose)
        if not v2.interpolate:
            ok_s = ok_s & ~interp
        ok = ok_s

    rows = []
    for j, fi in enumerate(idxs):
        if not ok[j]:
            rows.append(dict(frame=fi, ok=0)); continue
        base_cam = cams.get(fi)
        Cj = C if C is not None else (base_cam.C if base_cam is not None else None)
        if Cj is None:
            # interpolated frame with no per-frame camera and no shared
            # centre: nothing to anchor it to.
            rows.append(dict(frame=fi, ok=0)); continue
        cam = Camera(fo[j], pan[j], tilt[j], roll[j], Cj,
                     base_cam.k1 if base_cam is not None else 0.0, image_wh)
        row = dict(frame=fi, ok=1, s=np.nan, pan=cam.pan, tilt=cam.tilt,
                   roll=cam.roll, f=cam.f, cx=cam.C[0], cy=cam.C[1],
                   cz=cam.C[2], k1=cam.k1, interpolated=int(interp[j]))
        row["reproj_px"] = reprojection_error(cam, ann.get(fi, {}))
        rows.append(row)
    return pd.DataFrame(rows)
