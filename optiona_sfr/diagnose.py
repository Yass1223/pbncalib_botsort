"""Diagnostics that separate a WRONG METRIC from a WRONG CAMERA.

If median reprojection error is high, exactly one of these is true:
  (a) the world model / annotation-name mapping is wrong -> even a perfect
      camera scores badly, and no amount of estimation work will help;
  (b) the mapping is right and the estimation is at fault.

`fit_camera_to_annotations` settles it: it fits a camera directly to the
GROUND-TRUTH annotations. That is an upper bound on achievable accuracy for
our world model. A small residual means the model and metric are sound and the
problem is estimation; a large residual means the model or the name mapping is
wrong and every downstream number is meaningless.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

from .geometry import (Camera, param_bounds, clip_to_bounds,
                       camera_is_plausible, PITCH_L, PITCH_W)
from .experiments import (load_pitch_annotations, _world_geometry_by_name,
                          reprojection_error)


def annotation_report(seq_dir, labels_dir=None, max_frames=3,
                      image_wh=(1920, 1080)):
    """Which annotation classes exist, and are they mapped to world geometry?"""
    ann = load_pitch_annotations(Path(seq_dir), labels_dir, image_wh=image_wh,
                                 verbose=True)
    if not ann:
        print("[diag] NO annotations parsed — the JSON schema differs from "
              "what load_pitch_annotations expects. Everything downstream "
              "that uses reproj_px is meaningless until this is fixed.")
        return ann
    geom = _world_geometry_by_name()
    seen, mapped, unmapped = {}, set(), set()
    for f, d in list(ann.items())[:max_frames]:
        for cls, pts in d.items():
            seen[cls] = seen.get(cls, 0) + len(pts)
            (mapped if cls in geom else unmapped).add(cls)
    print(f"[diag] {len(ann)} annotated frames; classes in first "
          f"{max_frames} frames:")
    for cls, n in sorted(seen.items()):
        print(f"    {'OK ' if cls in geom else 'MISSING'}  {cls}  ({n} pts)")
    if unmapped:
        print(f"[diag] {len(unmapped)} class names have NO world geometry — "
              "they are silently skipped by the metric. Fix the name map in "
              "experiments._world_geometry_by_name before trusting results.")
    return ann


def _ann_residuals(cam, ann_frame, geom):
    res = []
    for cls, pts in ann_frame.items():
        W = geom.get(cls)
        if W is None:
            continue
        prj, front, _ = cam.project_full(W)
        prj = prj[front]
        prj = prj[np.isfinite(prj).all(axis=1)]
        if len(prj) < 2:
            res.append(np.full(len(pts), 1e4))
            continue
        d = np.linalg.norm(pts[:, None, :] - prj[None, :, :], axis=2).min(1)
        res.append(d)
    if not res:
        return np.array([1e4])
    return np.nan_to_num(np.concatenate(res), nan=1e4, posinf=1e4)


def fit_camera_to_annotations(ann_frame, image_wh=(1920, 1080), n_starts=6,
                              seed=0, geom=None, seed_cameras=()):
    """Best-achievable camera for THIS frame given our world model, fitted to
    the ground-truth annotations from several starting points.
    Returns (camera, median residual px)."""
    geom = geom if geom is not None else _world_geometry_by_name()
    rng = np.random.default_rng(seed)
    lo, hi = param_bounds(7)
    best, best_err = None, np.inf
    starts = []
    for _ in range(n_starts):
        starts.append(Camera(
            f=rng.uniform(1200, 5000),
            pan=rng.uniform(-0.6, 0.6),
            tilt=rng.uniform(1.0, 1.6),
            roll=rng.uniform(-0.1, 0.1),
            C=[rng.uniform(-20, 20), rng.uniform(-90, -35),
               rng.uniform(-25, -6)],
            image_wh=image_wh))
    # both touchlines (broadcast cameras sit on either side)
    starts.append(Camera(3000, 0.0, 1.31, 0.0, [0, -55, -14], image_wh=image_wh))
    starts.append(Camera(3000, np.pi, 1.31, 0.0, [0, 55, -14], image_wh=image_wh))
    # Real estimates as starting points. Without them a multi-start fit can
    # land in a local minimum and report a "best achievable" WORSE than an
    # actual camera — which makes the bound meaningless.
    starts.extend([c for c in seed_cameras if c is not None])
    for c0 in starts:
        x0 = clip_to_bounds(c0.to_vec(optimize_k1=False))

        def fn(v):
            cam = Camera.from_vec(v, image_wh)
            return _ann_residuals(cam, ann_frame, geom)

        try:
            sol = least_squares(fn, x0, method="trf", loss="soft_l1",
                                f_scale=20.0, max_nfev=200, x_scale="jac",
                                bounds=(lo, hi))
        except Exception:
            continue
        cam = Camera.from_vec(sol.x, image_wh)
        err = float(np.median(_ann_residuals(cam, ann_frame, geom)))
        if err < best_err and camera_is_plausible(cam):
            best, best_err = cam, err
    return best, best_err


def diagnose_sequence(seq_dir, cfg, kp_world, membership, frames=(1, 100, 300),
                      labels_dir=None):
    """Full verdict for one sequence. Prints, in order:
      1. annotation classes and whether they map to world geometry;
      2. BEST-ACHIEVABLE error (camera fitted to ground truth) -> tests the
         world model and metric;
      3. raw PnLCalib error and refined error -> tests the estimation.
    """
    from .detection import load_cached, make_single_frame_calibrator
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .refine import refine_camera, build_circle_obs

    seq_dir = Path(seq_dir)
    labels_dir = labels_dir or getattr(cfg.paths, "labels_dir", None)
    first = load_cached(cfg.paths.cache_dir, seq_dir.name, 1)
    image_wh = tuple(first["wh"])
    ann = annotation_report(seq_dir, labels_dir, image_wh=image_wh)
    if not ann:
        return
    cal_raw = make_single_frame_calibrator(cfg.paths.pnlcalib_repo, image_wh,
                                           cfg.refine, kp_world, membership,
                                           use_refine=False)
    print(f"\n[diag] {seq_dir.name}: frame | best-achievable | PnLCalib raw | "
          "+our refinement   (median px)")
    for fi in frames:
        det = load_cached(cfg.paths.cache_dir, seq_dir.name, fi)
        if det is None or fi not in ann:
            print(f"    {fi:>5} | (no cache or no annotation)")
            continue
        ctx = {"kp": det["kp"], "lines": det["lines"]}
        cam_raw = cal_raw(ctx)
        # seed with the real estimate: a "best achievable" worse than an
        # actual camera would be a failed fit, not a bound
        _, best_err = fit_camera_to_annotations(ann[fi], image_wh,
                                                seed_cameras=(cam_raw,))
        e_raw = reprojection_error(cam_raw, ann[fi]) if cam_raw else None
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        circ = build_circle_obs(kp_obs[0], kp_obs[1], kp_obs[2], membership)
        cam_ref = None
        if cam_raw is not None:
            cam_ref, _ = refine_camera(cam_raw, kp_obs, line_obs, circ,
                                       cfg.refine)
        e_ref = reprojection_error(cam_ref, ann[fi]) if cam_ref else None
        fmt = lambda e: "  n/a  " if e is None else f"{e:7.1f}"
        print(f"    {fi:>5} | {fmt(best_err)} | {fmt(e_raw)} | {fmt(e_ref)}")
    print("\n[diag] VERDICT:")
    print("  best-achievable LARGE (>50 px)  -> world model / name mapping is "
          "wrong; fix that first, ignore all other numbers.")
    print("  best-achievable SMALL, raw large -> detector-to-world index "
          "mapping or PnLCalib initialisation is the problem.")
    print("  raw small, refined larger        -> our refinement is hurting; "
          "check weights alpha/beta and the line-class ordering.")


# ----------------------------------------------------------------------------- conversion / mapping audit
def _P_from_pnlcalib(fp):
    """Projection matrix built EXACTLY as PnLCalib's inference.py does, so we
    can compare it against our Camera conversion."""
    cp = fp["cam_params"]
    fx, fy = float(cp["x_focal_length"]), float(cp["y_focal_length"])
    pp = np.asarray(cp["principal_point"], float)
    C = np.asarray(cp["position_meters"], float)
    R = np.asarray(cp["rotation_matrix"], float)
    It = np.eye(4)[:-1]
    It[:, -1] = -C
    Q = np.array([[fx, 0, pp[0]], [0, fy, pp[1]], [0, 0, 1.0]])
    return Q @ (R @ It), fx, fy, pp, C, R


def _project_P(P, X):
    Xh = np.hstack([np.atleast_2d(X), np.ones((len(np.atleast_2d(X)), 1))])
    p = (P @ Xh.T).T
    z = p[:, 2]
    ok = np.abs(z) > 1e-9
    uv = np.full((len(p), 2), np.nan)
    uv[ok] = p[ok, :2] / z[ok, None]
    return uv, ok & (z > 0)


def audit_conversion_and_mapping(seq_dir, cfg, kp_world, membership,
                                 frames=(1, 300), labels_dir=None):
    """Pinpoint WHY a PnLCalib camera scores badly. Four checks per frame:

    1. CONVERSION  — our Camera vs PnLCalib's own projection matrix. Must
       agree to <1 px, else Camera.from_pnlcalib is wrong.
    2. SELF-FIT    — does the raw camera reproject ITS OWN detected keypoints
       (via our kp_world index map) accurately? A large value here with a
       correct conversion means our keypoint INDEX MAPPING is wrong.
    3. ANNOTATION  — error against ground truth.
    4. MIRROR      — same, for the camera rotated 180 deg in pan, to test the
       left/right pitch-half ambiguity.
    """
    from .detection import load_cached
    from .pnl_adapter import keypoints_to_obs, add_repo_to_path
    seq_dir = Path(seq_dir)
    labels_dir = labels_dir or getattr(cfg.paths, "labels_dir", None)
    geom = _world_geometry_by_name()
    first = load_cached(cfg.paths.cache_dir, seq_dir.name, 1)
    image_wh = tuple(first["wh"])
    ann = load_pitch_annotations(seq_dir, labels_dir, image_wh=image_wh)
    add_repo_to_path(cfg.paths.pnlcalib_repo)
    from utils.utils_calib import FramebyFrameCalib
    engine = FramebyFrameCalib(iwidth=image_wh[0], iheight=image_wh[1],
                               denormalize=True)
    print(f"[audit] {seq_dir.name}  (image {image_wh})")
    for fi in frames:
        det = load_cached(cfg.paths.cache_dir, seq_dir.name, fi)
        if det is None:
            continue
        try:
            engine.update(det["kp"], det["lines"])
            fp = engine.heuristic_voting(refine_lines=False)
        except Exception as e:
            print(f"  frame {fi}: heuristic_voting failed: {e}")
            continue
        if fp is None:
            print(f"  frame {fi}: heuristic_voting returned None")
            continue
        P, fx, fy, pp, C, R = _P_from_pnlcalib(fp)
        cam = Camera.from_pnlcalib(fp, image_wh)
        print(f"  frame {fi}: fx={fx:.1f} fy={fy:.1f} pp={np.round(pp,1)} "
              f"C={np.round(C,2)} det(R)={np.linalg.det(R):+.3f}")
        # 1. conversion
        G = np.column_stack([np.linspace(-50, 50, 40),
                             np.linspace(-30, 30, 40), np.zeros(40)])
        uvP, okP = _project_P(P, G)
        uvC, okC, _ = cam.project_full(G)
        m = okP & okC
        dconv = (np.linalg.norm(uvP[m] - uvC[m], axis=1).max()
                 if m.any() else np.nan)
        # 2. self-fit on detected keypoints via our index map
        kp_idx, kp_uv, kp_conf, kp_X = keypoints_to_obs(det["kp"], image_wh,
                                                        kp_world)
        if len(kp_idx):
            prj, fr = cam.project(kp_X)
            e_self = float(np.median(np.linalg.norm(prj[fr] - kp_uv[fr], axis=1))) \
                if fr.any() else np.nan
        else:
            e_self = np.nan
        # 3/4. annotations, direct and mirrored
        e_ann = reprojection_error(cam, ann.get(fi, {}))
        mir = Camera(cam.f, cam.pan + np.pi, cam.tilt, cam.roll, cam.C,
                     cam.k1, image_wh)
        e_mir = reprojection_error(mir, ann.get(fi, {}))
        fmt = lambda e: "  n/a " if e is None or not np.isfinite(e) else f"{e:7.1f}"
        print(f"      conversion max diff {dconv:8.3f} px  | self-fit on "
              f"{len(kp_idx)} detected kps {fmt(e_self)} px")
        print(f"      annotation err {fmt(e_ann)} px | mirrored (pan+180) "
              f"{fmt(e_mir)} px")
    print("\n[audit] READ THIS AS:")
    print("  conversion diff > 1 px      -> Camera.from_pnlcalib is wrong.")
    print("  conversion ok, self-fit big -> our kp_world INDEX MAPPING is wrong")
    print("     (PnLCalib's camera must fit its own detections).")
    print("  self-fit ok, annotation big -> PnLCalib's estimate itself is off")
    print("     for these frames; if MIRRORED is small, it is the left/right")
    print("     pitch-half ambiguity.")


def resolve_and_install_convention(seq_dir, cfg, kp_world, membership,
                                   frames=(1, 100, 200, 300, 400),
                                   labels_dir=None):
    """Resolve the pitch sign convention from real data and install it.

    Uses real PnLCalib cameras as the tie-break, so the resulting world frame
    is the one the detected keypoints already live in. Call this once, before
    Stage 4; every reprojection number afterwards depends on it.
    """
    from .detection import load_cached, make_single_frame_calibrator
    from .experiments import set_pitch_convention
    from .pitch_model import resolve_convention
    seq_dir = Path(seq_dir)
    labels_dir = labels_dir or getattr(cfg.paths, "labels_dir", None)
    first = load_cached(cfg.paths.cache_dir, seq_dir.name, 1)
    image_wh = tuple(first["wh"])
    ann = load_pitch_annotations(seq_dir, labels_dir, image_wh=image_wh)
    if not ann:
        print("[pitch] no annotations — cannot resolve; keeping current "
              "convention")
        return None
    cal = make_single_frame_calibrator(cfg.paths.pnlcalib_repo, image_wh,
                                       cfg.refine, kp_world, membership,
                                       use_refine=False)
    ref = {}
    for fi in frames:
        det = load_cached(cfg.paths.cache_dir, seq_dir.name, fi)
        if det is None or fi not in ann:
            continue
        cam = cal({"kp": det["kp"], "lines": det["lines"]})
        if cam is not None:
            ref[fi] = cam
    sub = {f: ann[f] for f in frames if f in ann}
    rx, ty, report = resolve_convention(sub, image_wh, n_frames=len(sub),
                                        ref_cameras=ref)
    set_pitch_convention(rx, ty)
    print(f"[pitch] installed convention right_x={'+' if rx else '-'}, "
          f"top_y={'+' if ty else '-'}")
    return rx, ty, report


# ----------------------------------------------------------------------------- correspondence audit
def audit_correspondences(seq_dir, cfg, kp_world, membership,
                          frames=(1, 100, 300), labels_dir=None):
    """Verify the DETECTOR-INDEX -> WORLD-ELEMENT maps empirically.

    Uses the raw PnLCalib camera (independently validated by its own annotation
    error) as a reference: project every world line/keypoint, then ask which
    world element each DETECTION actually matches best. If our assumed index
    map is right, the best match equals the detected index. Any systematic
    disagreement is the line-class ordering (or keypoint index) assumption
    failing — the remaining unverified piece, and the prime suspect when
    refinement makes a good camera worse.
    """
    from .detection import load_cached, make_single_frame_calibrator
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .geometry import LINES_WORLD, point_line_distance
    seq_dir = Path(seq_dir)
    first = load_cached(cfg.paths.cache_dir, seq_dir.name, 1)
    image_wh = tuple(first["wh"])
    cal = make_single_frame_calibrator(cfg.paths.pnlcalib_repo, image_wh,
                                       cfg.refine, kp_world, membership,
                                       use_refine=False)
    line_hits = line_tot = kp_hits = kp_tot = 0
    line_map, kp_map = {}, {}
    for fi in frames:
        det = load_cached(cfg.paths.cache_dir, seq_dir.name, fi)
        if det is None:
            continue
        cam = cal({"kp": det["kp"], "lines": det["lines"]})
        if cam is None:
            continue
        # --- lines -------------------------------------------------------
        proj = []
        for A, B in LINES_WORLD:
            p, fr = cam.project(np.stack([A, B]))
            proj.append(p if fr.all() else None)
        for (li, pts, _c) in lines_to_obs(det["lines"], image_wh):
            best, bestd = None, np.inf
            for j, p in enumerate(proj, start=1):
                if p is None:
                    continue
                d = float(np.mean(point_line_distance(pts, p[0], p[1])))
                if d < bestd:
                    best, bestd = j, d
            if best is None:
                continue
            line_tot += 1
            line_hits += int(best == li)
            line_map.setdefault(li, []).append((best, round(bestd, 1)))
        # --- keypoints ---------------------------------------------------
        idx, uv, conf, X = keypoints_to_obs(det["kp"], image_wh, kp_world)
        if len(idx):
            allk = np.array([kp_world[k] for k in sorted(kp_world)])
            keys = np.array(sorted(kp_world))
            pk, frk = cam.project(allk)
            for i, k in enumerate(idx):
                d = np.linalg.norm(pk[frk] - uv[i], axis=1)
                if not len(d):
                    continue
                best = int(keys[frk][np.argmin(d)])
                kp_tot += 1
                kp_hits += int(best == int(k))
                kp_map.setdefault(int(k), []).append((best, round(float(d.min()), 1)))
    print(f"[corr] lines: {line_hits}/{line_tot} detections matched the world "
          f"element our index map assumes "
          f"({100.0*line_hits/max(line_tot,1):.0f}%)")
    print(f"[corr] keypoints: {kp_hits}/{kp_tot} "
          f"({100.0*kp_hits/max(kp_tot,1):.0f}%)")
    bad = {k: v for k, v in line_map.items()
           if v and sum(b == k for b, _ in v) < len(v) / 2}
    if bad:
        print("[corr] line indices whose detections consistently match a "
              "DIFFERENT world line (detected -> matched, distance):")
        for k in sorted(bad):
            print(f"    {k:>3} -> {bad[k]}")
    return line_map, kp_map


def refinement_term_ablation(seq_dir, cfg, kp_world, membership,
                             frames=(1, 100, 300), labels_dir=None):
    """Which residual term degrades a good camera? Measures annotation error
    for raw / keypoints-only / +lines / +lines+conic."""
    import dataclasses
    from .detection import load_cached, make_single_frame_calibrator
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .refine import refine_camera, build_circle_obs
    seq_dir = Path(seq_dir)
    labels_dir = labels_dir or getattr(cfg.paths, "labels_dir", None)
    first = load_cached(cfg.paths.cache_dir, seq_dir.name, 1)
    image_wh = tuple(first["wh"])
    ann = load_pitch_annotations(seq_dir, labels_dir, image_wh=image_wh)
    cal = make_single_frame_calibrator(cfg.paths.pnlcalib_repo, image_wh,
                                       cfg.refine, kp_world, membership,
                                       use_refine=False)
    print(f"[terms] {seq_dir.name}: frame |    raw |  kp only |  kp+lines | "
          "kp+lines+conic  (median px)")
    for fi in frames:
        det = load_cached(cfg.paths.cache_dir, seq_dir.name, fi)
        if det is None or fi not in ann:
            continue
        cam0 = cal({"kp": det["kp"], "lines": det["lines"]})
        if cam0 is None:
            continue
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        circ = build_circle_obs(kp_obs[0], kp_obs[1], kp_obs[2], membership)
        outs = []
        for lines, circles, cfgr in [
                ([], [], dataclasses.replace(cfg.refine, use_conic_term=False)),
                (line_obs, [], dataclasses.replace(cfg.refine, use_conic_term=False)),
                (line_obs, circ, cfg.refine)]:
            cam, _ = refine_camera(cam0, kp_obs, lines, circles, cfgr)
            outs.append(reprojection_error(cam, ann[fi]) if cam else None)
        f = lambda e: "   n/a " if e is None or not np.isfinite(e) else f"{e:7.1f}"
        e0 = reprojection_error(cam0, ann[fi])
        print(f"         {fi:>5} |{f(e0)} | {f(outs[0])} | {f(outs[1])} | "
              f"{f(outs[2])}")
    print("[terms] the first column that jumps identifies the harmful term.")
