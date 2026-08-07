"""Detection front-end + caching.

Heavy HRNet inference runs ONCE per sequence and is cached as one .npz per
frame (kp/lines dicts pickled inside). Every downstream experiment and
ablation reruns from the cache on CPU, so all methods consume byte-identical
detections — the comparability requirement of the test protocol.
"""
from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path

from .pnl_adapter import add_repo_to_path


def detect_frame(frame_bgr, m_kp, m_l, dev, cfg, repo_dir):
    """Mirror of PnLCalib inference() up to (kp_dict, lines_dict) — verified
    against the repo's inference.py."""
    import cv2, torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
    from PIL import Image
    add_repo_to_path(repo_dir)
    from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
                                     get_keypoints_from_heatmap_batch_maxpool_l,
                                     complete_keypoints, coords_to_dict)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = F.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
    if t.size(-1) != cfg.input_wh[0]:
        t = T.Resize((cfg.input_wh[1], cfg.input_wh[0]))(t)
    t = t.to(dev)
    _, _, h, w = t.size()
    with torch.no_grad():
        hm_kp = m_kp(t)
        hm_l = m_l(t)
    kp = get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1, :, :])
    ln = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :])
    kp_dict = coords_to_dict(kp, threshold=cfg.kp_threshold)
    ln_dict = coords_to_dict(ln, threshold=cfg.line_threshold)
    kp_dict, ln_dict = complete_keypoints(kp_dict[0], ln_dict[0], w=w, h=h,
                                          normalize=True)
    return kp_dict, ln_dict


def cache_path(cache_dir, seq_id, frame_idx):
    d = Path(cache_dir) / str(seq_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{frame_idx:06d}.pkl"


def compute_flow_pairs(prev_gray, gray, max_points=150, quality=0.01,
                       min_dist=12):
    """Shi-Tomasi + Lucas-Kanade correspondences between consecutive frames,
    in PIXEL space (camera-independent). Cached at detection time so the
    optimization stages need no images: the tracker back-projects p0 with its
    own kappa_{t-1} later, which is exactly what the live path does."""
    import cv2
    if prev_gray is None or gray is None:
        return None
    p0 = cv2.goodFeaturesToTrack(prev_gray, max_points, quality, min_dist)
    if p0 is None or len(p0) < 6:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                         winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 6:
        return None
    return (p0.reshape(-1, 2)[ok].astype("float32"),
            p1.reshape(-1, 2)[ok].astype("float32"))


def cache_sequence(frames_iter, seq_id, models, cfg, repo_dir, cache_dir,
                   overwrite=False, compute_flow=True, subpixel=True):
    """frames_iter yields (frame_idx, frame_bgr). Skips already-cached frames
    so interrupted Kaggle sessions resume for free. Stores detections AND
    optical-flow pairs, which makes images disposable afterwards."""
    import cv2
    m_kp, m_l, dev = models
    n = 0
    prev_gray = None
    for idx, frame in frames_iter:
        p = cache_path(cache_dir, seq_id, idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if compute_flow else None
        if p.exists() and not overwrite:
            prev_gray = gray
            continue
        kp_dict, ln_dict = detect_frame(frame, m_kp, m_l, dev, cfg, repo_dir)
        sub_shift = 0.0
        if subpixel:
            # Detector coordinates are quantized to the heatmap stride
            # (~2-8 px at 1080p) — refine against image gradients HERE, while
            # pixels are available, and cache the result so the pipeline stays
            # image-free afterwards.
            from .subpixel import refine_detections
            g = gray if gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_dict, sub_shift = refine_detections(
                g, kp_dict, (frame.shape[1], frame.shape[0]))
        flow = compute_flow_pairs(prev_gray, gray) if compute_flow else None
        with open(p, "wb") as f:
            pickle.dump({"kp": kp_dict, "lines": ln_dict,
                         "wh": (frame.shape[1], frame.shape[0]),
                         "flow": flow, "subpixel_shift": sub_shift}, f)
        prev_gray = gray
        n += 1
    return n


def load_cached(cache_dir, seq_id, frame_idx):
    p = cache_path(cache_dir, seq_id, frame_idx)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- single-frame calibration
def make_single_frame_calibrator(repo_dir, image_wh, refine_cfg, kp_world,
                                 membership, use_refine=True,
                                 refine_lines=False):
    """Returns f(frame_ctx)->Camera|None where frame_ctx = {'kp':..,'lines':..}.
    Runs PnLCalib FramebyFrameCalib.heuristic_voting for the initial estimate,
    then our upgraded refinement (Layer 2)."""
    add_repo_to_path(repo_dir)
    from utils.utils_calib import FramebyFrameCalib
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .refine import refine_camera, build_circle_obs
    from .geometry import Camera

    cam_engine = FramebyFrameCalib(iwidth=image_wh[0], iheight=image_wh[1],
                                   denormalize=True)

    def calibrate(ctx):
        try:
            cam_engine.update(ctx["kp"], ctx["lines"])
            fp = cam_engine.heuristic_voting(refine_lines=refine_lines)
        except Exception:
            fp = None
        if fp is None:
            return None
        cam = Camera.from_pnlcalib(fp, image_wh)
        from .geometry import camera_is_plausible
        if not camera_is_plausible(cam):
            return None
        if not use_refine:
            return cam
        kp_obs = keypoints_to_obs(ctx["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(ctx["lines"], image_wh)
        circle_obs = build_circle_obs(kp_obs[0], kp_obs[1], kp_obs[2],
                                      membership)
        cam, _ = refine_camera(cam, kp_obs, line_obs, circle_obs, refine_cfg)
        return cam

    return calibrate


# ----------------------------------------------------------------------------- per-frame variant selection
def _self_fit(cam, det, image_wh, kp_world, membership, refine_cfg):
    """How well a camera explains ITS OWN detections (median px).

    Ground-truth-free, so it is a legitimate selection criterion: PnLCalib's
    camera must fit the keypoints and lines it was derived from.
    """
    if cam is None or det is None:
        return float("inf")
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .refine import score_camera
    kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
    line_obs = lines_to_obs(det["lines"], image_wh)
    if len(kp_obs[0]) < 4:
        return float("inf")
    return score_camera(cam, kp_obs, line_obs, refine_cfg)


def select_baseline_variant(paths, seq_id, image_wh, refine_cfg, kp_world,
                            membership, frames=None, margin=1.10,
                            verbose=True):
    """Choose PnLCalib's `refine_lines` setting PER FRAME, by self-fit.

    Neither setting wins everywhere. Measured on SoccerNet-GSR (median
    reprojection over a whole sequence):

        sequence   refine_lines=False   refine_lines=True
        SNGS-118             13.5 px            616.4 px
        SNGS-119            902.3 px             10.5 px

    Native line refinement is catastrophic on one sequence and essential on
    another, so a fixed choice guarantees a broken sequence either way. Both
    variants are cheap (cached), and the better one is identifiable without
    ground truth by asking which camera fits its own detections. `margin`
    requires the alternative to be clearly better before switching, so noise
    cannot flip the choice.
    """
    from .experiments import frame_indices
    idxs = frames or frame_indices(paths.cache_dir, seq_id)
    A = get_baseline_cameras(paths, seq_id, image_wh, refine_cfg, kp_world,
                             membership, refine_lines=True, frames=idxs)
    B = get_baseline_cameras(paths, seq_id, image_wh, refine_cfg, kp_world,
                             membership, refine_lines=False, frames=idxs)
    out, n_true, n_false, n_none = {}, 0, 0, 0
    for fi in idxs:
        det = load_cached(paths.cache_dir, seq_id, fi)
        a, b = A.get(fi), B.get(fi)
        sa = _self_fit(a, det, image_wh, kp_world, membership, refine_cfg)
        sb = _self_fit(b, det, image_wh, kp_world, membership, refine_cfg)
        if not np.isfinite(sa) and not np.isfinite(sb):
            out[fi] = a if a is not None else b
            n_none += 1
            continue
        if sb < sa / margin:            # switch only on a clear win
            out[fi] = b; n_false += 1
        else:
            out[fi] = a; n_true += 1
    if verbose:
        print(f"[variant] {seq_id}: refine_lines=True on {n_true} frames, "
              f"False on {n_false}, undecided {n_none}")
    return out


# ----------------------------------------------------------------------------- baseline camera cache
def baseline_cache_path(cache_dir, seq_id, refine_lines):
    """Stored in a SEPARATE directory, never beside the frame pickles: a
    sibling file there is one careless glob away from breaking frame
    enumeration (it did)."""
    from pathlib import Path
    d = Path(cache_dir) / "_baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{seq_id}_rl{int(bool(refine_lines))}.pkl"


def get_baseline_cameras(paths, seq_id, image_wh, refine_cfg, kp_world,
                         membership, refine_lines=True, frames=None,
                         verbose=False):
    """PnLCalib's per-frame estimate, computed ONCE per sequence and reused.

    It is identical for every configuration in every grid, but was being
    recomputed for each — 12 configs x 3000 frames = 36,000 redundant
    heuristic_voting calls, which is what exhausted a Kaggle session before it
    produced any results. Cached as plain parameters so it survives sessions.
    """
    from pathlib import Path
    from .geometry import Camera
    p = baseline_cache_path(paths.cache_dir, seq_id, refine_lines)
    cached = {}
    if p.exists():
        try:
            with open(p, "rb") as f:
                cached = pickle.load(f)
        except Exception:
            cached = {}
    from .experiments import frame_indices
    idxs = frames or frame_indices(paths.cache_dir, seq_id)
    missing = [i for i in idxs if i not in cached]
    if missing:
        cal = make_single_frame_calibrator(paths.pnlcalib_repo, image_wh,
                                           refine_cfg, kp_world, membership,
                                           use_refine=False,
                                           refine_lines=refine_lines)
        for fi in missing:
            det = load_cached(paths.cache_dir, seq_id, fi)
            if det is None:
                cached[fi] = None
                continue
            cam = cal({"kp": det["kp"], "lines": det["lines"]})
            cached[fi] = None if cam is None else dict(
                f=cam.f, pan=cam.pan, tilt=cam.tilt, roll=cam.roll,
                C=list(map(float, cam.C)), k1=cam.k1,
                pp=list(map(float, cam.pp)))
        try:
            with open(p, "wb") as f:
                pickle.dump(cached, f)
        except Exception:
            pass
        if verbose:
            print(f"[baseline] {seq_id}: computed {len(missing)} frames "
                  f"(cached {len(cached) - len(missing)})")
    out = {}
    for fi in idxs:
        d = cached.get(fi)
        if d is None:
            out[fi] = None
            continue
        cam = Camera(d["f"], d["pan"], d["tilt"], d["roll"], d["C"], d["k1"],
                     image_wh)
        cam.pp = np.array(d["pp"], float)
        out[fi] = cam
    return out
