"""Method v3 — so(3) trajectory, shared-centre bundle adjustment, BIC envelope.

Phase 2: one optical centre per sequence with a GAUSSIAN-CORE prior (not pure
Huber: Huber's linear tail makes large deviations cheap along the
focal<->distance direction, the exact degeneracy that produced 39,475 m of
focal travel; the prior exists to close it). Core sigma ~0.3 m where
micro-vibration lives, Huber tail beyond 3 sigma, hard box at +-2 m.

Phase 3: all rotation smoothing and interpolation in the Lie algebra so(3).
Euler interpolation is non-geodesic and our ZXZ parameterisation couples pan
and roll, so independent per-angle smoothing creates artefacts. Focal length is
smoothed in LOG space because zoom is multiplicative.

Phase 4: BIC selects between candidate cameras per frame on the inlier set,
with sigma from robust MAD (residuals are heavy-tailed, not Gaussian) and k as
effective DOF.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from .geometry import Camera, ptr_to_R, R_to_ptr


# ----------------------------------------------------------------------------- so(3)
def so3_log(R):
    """SO(3) -> so(3) (rotation vector), numerically safe near 0 and pi."""
    R = np.asarray(R, float)
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = np.arccos(c)
    if th < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - th < 1e-6:                       # near pi: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        v = np.sqrt(np.maximum(np.diag(A), 0.0))
        i = int(np.argmax(v))
        v = v * np.sign(A[i]) if v[i] > 0 else v
        v = v / max(np.linalg.norm(v), 1e-12)
        return v * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * np.sin(th)))


def so3_exp(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def slerp_rotations(R0, R1, t):
    """Geodesic interpolation on SO(3) — the manifold-correct SLERP."""
    return R0 @ so3_exp(t * so3_log(R0.T @ R1))


def smooth_rotations_so3(Rs, ok, window=9, polyorder=2):
    """Smooth and interpolate a rotation trajectory in the Lie algebra.

    Rotations are lifted to so(3) relative to a running reference (the nearest
    valid rotation), smoothed there, and mapped back. Gaps are filled by
    geodesic interpolation, which is SLERP driven by the local angular
    velocity.
    """
    from scipy.signal import savgol_filter
    n = len(Rs)
    ok = np.asarray(ok, bool)
    if ok.sum() < 3:
        return list(Rs), np.zeros(n, bool)
    idx = np.arange(n)
    valid = idx[ok]
    ref = Rs[valid[0]]
    # lift to a continuous 3-vector field relative to the reference
    v = np.zeros((n, 3))
    for i in valid:
        v[i] = so3_log(ref.T @ Rs[i])
    # interpolate the algebra coordinates over gaps (geodesic in the small)
    filled = np.zeros((n, 3))
    for d in range(3):
        filled[:, d] = np.interp(idx, valid, v[valid, d])
    w = min(window if window % 2 == 1 else window + 1, n - (1 - n % 2))
    if w >= polyorder + 2:
        filled = savgol_filter(filled, w, polyorder, axis=0)
    out = [ref @ so3_exp(filled[i]) for i in range(n)]
    return out, ~ok


def smooth_log_focal(f, ok, window=9, polyorder=2):
    """Zoom is multiplicative, so smooth log f, not f."""
    from scipy.signal import savgol_filter
    n = len(f)
    ok = np.asarray(ok, bool) & np.isfinite(f) & (np.asarray(f) > 0)
    if ok.sum() < 3:
        return np.asarray(f, float)
    idx = np.arange(n)
    lf = np.interp(idx, idx[ok], np.log(np.asarray(f, float)[ok]))
    w = min(window if window % 2 == 1 else window + 1, n - (1 - n % 2))
    if w >= polyorder + 2:
        lf = savgol_filter(lf, w, polyorder)
    return np.exp(lf)


# ----------------------------------------------------------------------------- centre prior
def center_prior_residual(C, C0, sigma=0.3, huber_k=3.0, box=2.0):
    """Gaussian core, Huber tail, hard box.

    Quadratic within `huber_k * sigma` (micro-vibration regime), then linear so
    a genuinely supported shift is not fought, and the box makes the tail
    unusable as an escape route along the focal/distance degeneracy.
    """
    d = (np.asarray(C, float) - np.asarray(C0, float)) / sigma
    r = np.linalg.norm(d)
    if r <= huber_k:
        out = d
    else:
        out = d * np.sqrt(huber_k * (2.0 * r - huber_k)) / max(r, 1e-12)
    over = max(0.0, r * sigma - box)
    if over > 0:
        out = out * (1.0 + 50.0 * over)     # steep wall at the box
    return out


# ----------------------------------------------------------------------------- bundle adjustment
def bundle_adjust(frames, obs, cams, C0, cfg, sigma_C=0.3, max_frames=60,
                  verbose=False):
    """Joint optimisation of ONE shared centre and per-frame (pan, tilt, roll, f).

    Every frame constrains the centre for every other frame — information a
    per-frame estimator structurally cannot access, and the principled source
    of an accuracy gain rather than a tie.
    """
    from .refine import Residuals
    avail = [f for f in frames if cams.get(f) is not None]
    if len(avail) < 5:
        return None, cams
    # Evenly subsample: the shared centre is over-determined by a few dozen
    # well-spread frames, and cost grows with the parameter count.
    step = max(1, len(avail) // max_frames)
    sel = avail[::step][:max_frames]
    image_wh = (cams[sel[0]].w, cams[sel[0]].h)
    res_fns = []
    for f in sel:
        kp_obs, line_obs, circ = obs[f][0], obs[f][1], obs[f][2]
        R = Residuals(kp_obs, line_obs, circ, cfg, image_wh,
                      optimize_k1=True, k1_fixed=cams[f].k1)
        R.gate_lines(Camera(cams[f].f, cams[f].pan, cams[f].tilt, cams[f].roll,
                            C0, cams[f].k1, image_wh))
        res_fns.append(R)
    x0 = np.concatenate([np.asarray(C0, float)] +
                        [[cams[f].pan, cams[f].tilt, cams[f].roll, cams[f].f]
                         for f in sel])
    n = len(sel)

    def fn(x):
        C = x[:3]
        parts = [10.0 * center_prior_residual(C, C0, sigma_C)]
        for i, f in enumerate(sel):
            p = x[3 + 4 * i: 7 + 4 * i]
            v = np.array([p[0], p[1], p[2], p[3], C[0], C[1], C[2],
                          cams[f].k1])
            parts.append(res_fns[i](v))
        return np.concatenate(parts)

    lo = np.concatenate([np.asarray(C0) - 2.0,
                         np.tile([-2 * np.pi, 0.05, -np.pi / 3, 100.0], n)])
    hi = np.concatenate([np.asarray(C0) + 2.0,
                         np.tile([2 * np.pi, np.pi - 0.05, np.pi / 3, 30000.0], n)])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

    # SPARSITY. The Jacobian is block-arrow: each frame's residuals depend only
    # on the shared centre and on that frame's own 4 parameters. Declaring it
    # turns ~(3 + 4n) function evaluations per iteration into a handful — with
    # n = 120 that was ~487 evaluations of 120 residual blocks each, which is
    # what exhausted a Kaggle session mid-grid.
    from scipy.sparse import lil_matrix
    sizes = [len(f_(x0[3 + 4 * i: 7 + 4 * i].tolist() + list(C0) +
                   [cams[sel[i]].k1]))
             if False else len(res_fns[i](np.array(
                 [cams[sel[i]].pan, cams[sel[i]].tilt, cams[sel[i]].roll,
                  cams[sel[i]].f, *C0, cams[sel[i]].k1])))
             for i, f_ in enumerate(res_fns)]
    m = 3 + int(np.sum(sizes))
    S = lil_matrix((m, len(x0)), dtype=int)
    S[:3, :3] = 1
    row = 3
    for i, sz in enumerate(sizes):
        S[row:row + sz, :3] = 1                       # shared centre
        S[row:row + sz, 3 + 4 * i: 7 + 4 * i] = 1     # this frame only
        row += sz
    try:
        sol = least_squares(fn, x0, method="trf", loss=cfg.robust_loss,
                            f_scale=cfg.robust_fscale, max_nfev=40,
                            x_scale="jac", bounds=(lo, hi),
                            jac_sparsity=S.tocsr())
    except Exception:
        return None, cams
    C = sol.x[:3]
    out = dict(cams)
    for i, f in enumerate(sel):
        p = sol.x[3 + 4 * i: 7 + 4 * i]
        out[f] = Camera(p[3], p[0], p[1], p[2], C, cams[f].k1, image_wh)
    if verbose:
        print(f"[v3] bundle: C {np.round(C0, 2)} -> {np.round(C, 2)} "
              f"({len(sel)} frames)")
    return C, out


# ----------------------------------------------------------------------------- BIC envelope
def bic(residuals, k, sigma=None):
    """BIC on the inlier set. sigma from robust MAD, since residuals are
    heavy-tailed rather than Gaussian; k is EFFECTIVE degrees of freedom."""
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return np.inf
    if sigma is None:
        sigma = max(np.median(np.abs(r - np.median(r))) * 1.4826, 1e-3)
    rss = float(np.sum((r / sigma) ** 2))
    return n * np.log(max(rss / n, 1e-12)) + k * np.log(n)


def bic_select(candidates, kp_obs, line_obs, cfg, k_map, margin=2.0,
               verbose=False):
    """Pick the candidate camera with the lowest BIC, requiring a margin over
    the baseline so noise cannot flip the choice. `candidates` is an ordered
    dict-like [(name, camera)] whose FIRST entry is the baseline."""
    from .refine import score_camera
    from .geometry import LINES_WORLD, point_line_distance
    def resid(cam):
        if cam is None:
            return None
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
            return None
        return np.concatenate(vals)

    base_name, base_cam = candidates[0]
    rb = resid(base_cam)
    if rb is None:
        for nm, c in candidates[1:]:
            if c is not None:
                return nm, c
        return base_name, base_cam
    sigma = max(np.median(np.abs(rb - np.median(rb))) * 1.4826, 1e-3)
    best_name, best_cam = base_name, base_cam
    best = bic(rb, k_map.get(base_name, 8), sigma)
    for nm, cam in candidates[1:]:
        r = resid(cam)
        if r is None:
            continue
        v = bic(r, k_map.get(nm, 4), sigma)
        if v < best - margin:
            best_name, best_cam, best = nm, cam, v
    if verbose:
        print(f"[bic] selected {best_name}")
    return best_name, best_cam


# ----------------------------------------------------------------------------- Mahalanobis repair
def reprojection_covariances(cam, X, cfg, sigma_det=2.0, cond_cap=1e3):
    """Per-point reprojection covariance Sigma_i = J_i Sigma_theta J_i^T + sigma_d^2 I.

    Sigma_theta ~ sigma^2 (J^T J)^-1 from the camera Jacobian. Reprojection
    uncertainty is strongly anisotropic at grazing angles — exactly our zoomed
    touchline geometry — so Euclidean matching is the wrong metric.

    Guard: in zoomed views J^T J is ill-conditioned, and an over-elongated
    ellipse makes WRONG matches look cheap along the elongated axis. The
    condition number is capped and eigenvalues floored at the detector noise
    level; on failure we fall back to isotropic (Euclidean) covariance.
    """
    X = np.atleast_2d(np.asarray(X, float))
    v0 = cam.to_vec(optimize_k1=False)
    eps = np.array([1e-5, 1e-5, 1e-5, 1.0, 1e-3, 1e-3, 1e-3])
    J = np.zeros((len(X), 2, 7))
    base, front, _ = cam.project_full(X)
    for j in range(7):
        vp = v0.copy(); vp[j] += eps[j]
        pj, _, _ = Camera.from_vec(vp, (cam.w, cam.h)).project_full(X)
        J[:, :, j] = (pj - base) / eps[j]
    Jf = J[front].reshape(-1, 7)
    if len(Jf) < 8:
        return None, front
    A = Jf.T @ Jf
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, w.max() / cond_cap if w.max() > 0 else 1e-9)
    Sigma_theta = (V * (1.0 / w)) @ V.T * (sigma_det ** 2)
    Sig = np.einsum('nij,jk,nlk->nil', J, Sigma_theta, J) + \
        (sigma_det ** 2) * np.eye(2)[None, :, :]
    return Sig, front


def mahalanobis_repair(cam, kp_uv, kp_conf, kp_world, cfg, sigma_det=2.0,
                       chi2_gate=9.21, verbose=False):
    """Re-assign detections to their most likely world element under the
    trusted camera, using Mahalanobis rather than Euclidean distance.

    Detections our index map got wrong (10% on real data) are currently
    DISCARDED; recovering them enlarges the constraint pool beyond what the
    baseline uses. chi2_gate = 9.21 is the 99% quantile for 2 DOF.
    """
    keys = np.array(sorted(kp_world))
    Xall = np.array([kp_world[k] for k in keys])
    Sig, front = reprojection_covariances(cam, Xall, cfg, sigma_det)
    prj, fr = cam.project(Xall)
    vis = np.where(fr)[0]
    if len(vis) == 0:
        return None
    assign, dists = [], []
    for i, uv in enumerate(np.atleast_2d(kp_uv)):
        best, bd = None, np.inf
        for j in vis:
            d = prj[j] - uv
            if Sig is None:
                m2 = float(d @ d) / (sigma_det ** 2)
            else:
                try:
                    m2 = float(d @ np.linalg.solve(Sig[j], d))
                except np.linalg.LinAlgError:
                    m2 = float(d @ d) / (sigma_det ** 2)
            if m2 < bd:
                best, bd = int(keys[j]), m2
        if best is not None and bd <= chi2_gate:
            assign.append(best); dists.append(bd)
        else:
            assign.append(None); dists.append(bd)
    n_ok = sum(a is not None for a in assign)
    if verbose:
        print(f"[repair] {n_ok}/{len(assign)} detections assigned within the "
              f"99% chi2 gate")
    return assign, np.array(dists)


# ----------------------------------------------------------------------------- sequence
def run_sequence_v3(seq_dir, exp_cfg, kp_world, membership, verbose=False):
    """v3 pipeline: parity -> repair -> bundle -> so(3) smoothing -> BIC."""
    import pandas as pd
    from pathlib import Path
    from .detection import load_cached, get_baseline_cameras
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .refine import build_circle_obs, refine_camera
    from .method_v2 import estimate_fixed_center
    from .parity import flip_pitch_half
    from .experiments import load_pitch_annotations, reprojection_error

    v3 = exp_cfg.v3
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
    base = get_baseline_cameras(paths, seq_id, image_wh, exp_cfg.refine,
                                kp_world, membership, refine_lines=True,
                                frames=idxs, verbose=verbose)
    obs = {}
    for fi in idxs:
        det = load_cached(paths.cache_dir, seq_id, fi)
        if det is None:
            continue
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        circ = build_circle_obs(kp_obs[0], kp_obs[1], kp_obs[2], membership)
        obs[fi] = (kp_obs, line_obs, circ, det.get("flow"))

    # --- Phase 0: parity (majority vote over the sequence) ----------------
    if v3.parity_check:
        from .refine import score_camera
        votes = 0, 0
        agree = dis = 0
        ref = None
        for fi in idxs[::37]:
            c = base.get(fi)
            if c is None:
                continue
            if ref is None:
                ref = c
                continue
            # a frame whose centre is on the opposite half is a parity outlier
            if np.dot(c.C[:2], ref.C[:2]) < 0:
                dis += 1
            else:
                agree += 1
        if dis > agree and ref is not None:
            for fi in idxs:                    # flip the minority half
                c = base.get(fi)
                if c is not None and np.dot(c.C[:2], ref.C[:2]) >= 0:
                    base[fi] = flip_pitch_half(c)
            if verbose:
                print(f"[v3] parity: flipped {agree} frames to the majority half")

    # --- Phase 2: shared centre + bundle adjustment -----------------------
    C = None
    if v3.shared_center:
        good = [base[i] for i in idxs if base.get(i) is not None]
        C, _ = estimate_fixed_center(good, None, verbose=verbose)
        if C is not None and v3.bundle_adjust:
            Cb, cams_b = bundle_adjust(idxs, obs, base, C, exp_cfg.refine,
                                       sigma_C=v3.sigma_center, verbose=verbose)
            if Cb is not None:
                C, base_bundled = Cb, cams_b
            else:
                base_bundled = base
        else:
            base_bundled = base
    else:
        base_bundled = base

    # --- Phase 3: so(3) + log-focal smoothing -----------------------------
    ok = np.array([base_bundled.get(i) is not None for i in idxs])
    Rs = [base_bundled[i].R if base_bundled.get(i) is not None else np.eye(3)
          for i in idxs]
    fs = np.array([base_bundled[i].f if base_bundled.get(i) is not None else np.nan
                   for i in idxs])
    interp = np.zeros(len(idxs), bool)
    if v3.smooth and ok.sum() > 5:
        Rs, interp = smooth_rotations_so3(Rs, ok, window=v3.window)
        fs = smooth_log_focal(np.nan_to_num(fs, nan=np.nanmedian(fs)), ok,
                              window=v3.window)
        if v3.interpolate:
            ok = np.ones(len(idxs), bool)

    # --- Phase 4: BIC envelope -------------------------------------------
    rows = []
    k_map = {"baseline": 8, "v3": 4}
    for j, fi in enumerate(idxs):
        if not ok[j]:
            rows.append(dict(frame=fi, ok=0)); continue
        pan, tilt, roll = R_to_ptr(Rs[j])
        Cj = C if C is not None else (base_bundled[fi].C
                                      if base_bundled.get(fi) is not None else None)
        if Cj is None:
            rows.append(dict(frame=fi, ok=0)); continue
        cand = Camera(fs[j], pan, tilt, roll, Cj,
                      base_bundled[fi].k1 if base_bundled.get(fi) is not None else 0.0,
                      image_wh)
        cam = cand
        chosen = "v3"
        if v3.bic_envelope and base.get(fi) is not None:
            kp_obs, line_obs = obs[fi][0], obs[fi][1]
            chosen, cam = bic_select([("baseline", base[fi]), ("v3", cand)],
                                     kp_obs, line_obs, exp_cfg.refine, k_map,
                                     margin=v3.bic_margin)
        row = dict(frame=fi, ok=1, s=np.nan, pan=cam.pan, tilt=cam.tilt,
                   roll=cam.roll, f=cam.f, cx=cam.C[0], cy=cam.C[1],
                   cz=cam.C[2], k1=cam.k1, interpolated=int(interp[j]),
                   chosen=chosen)
        row["reproj_px"] = reprojection_error(cam, ann.get(fi, {}))
        rows.append(row)
    return pd.DataFrame(rows)
