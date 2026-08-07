"""Adapter around the cloned PnLCalib repo.

The repo is GPL-2.0 and is cloned at runtime on Kaggle (never vendored here).
This module (a) loads its detector + FramebyFrameCalib exactly as its own
inference.py does (API verified from the repo main branch, 2026-08-02), and
(b) converts its kp/line dictionaries into plain numpy observations with
confidences and world coordinates for our refinement/tracking layers.

Keypoint world coordinates are read from PnLCalib's own utils at runtime; the
import is defensive because internal names may evolve. Circle membership of a
keypoint is decided *automatically* from its world (x, y): it belongs to a
circle iff | ||(x,y)-(cx,cy)|| - r | < tol. No manual class tables.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

from .geometry import LINES_WORLD, CIRCLES_WORLD, PITCH_L, PITCH_W


def add_repo_to_path(repo_dir):
    repo_dir = str(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


# ----------------------------------------------------------------------------- model loading
def load_models(cfg, repo_dir):
    """Exactly mirrors PnLCalib/inference.py model construction."""
    import yaml, torch
    add_repo_to_path(repo_dir)
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l

    repo_dir = Path(repo_dir)
    cfg_kp = yaml.safe_load(open(repo_dir / "config" / "hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open(repo_dir / "config" / "hrnetv2_w48_l.yaml"))
    dev = cfg.device if torch_cuda_ok(cfg.device) else "cpu"

    def _load(build, path, which):
        try:
            sd = torch.load(str(path), map_location=dev)
        except Exception as e:
            raise RuntimeError(
                f"Could not read the {which} checkpoint at {path}: {e}\n"
                "The file is corrupt or truncated (existence is not validity). "
                "Delete it and re-run Stage 0, or call "
                "kaggle_setup.setup_pnlcalib(paths, detector_cfg, "
                "force_weights=True) to force a fresh, verified download."
            ) from e
        m = build
        m.load_state_dict(sd)
        return m.to(dev).eval()

    m_kp = _load(get_cls_net(cfg_kp), cfg.weights_kp_path, "keypoint")
    m_l = _load(get_cls_net_l(cfg_l), cfg.weights_line_path, "line")
    return m_kp, m_l, dev


def torch_cuda_ok(device):
    import torch
    return device.startswith("cuda") and torch.cuda.is_available()


# ----------------------------------------------------------------------------- world templates
_KP_WORLD_CANDIDATES = [
    "keypoint_world_coords_2D", "keypoints_world_coords_2D",
    "keypoint_world_coords", "keypoints_world",
]
_KP_AUX_CANDIDATES = ["keypoint_aux_world_coords_2D", "keypoint_aux_world_coords"]


def _table_to_array(table):
    a = []
    for xy in table:
        xy = np.asarray(xy, float).ravel()
        a.append([xy[0], xy[1], xy[2] if xy.size > 2 else 0.0])
    return np.array(a, float)


def infer_and_center(arr):
    """Decide the world-frame convention ONCE for the whole table, then apply
    it uniformly.

    A per-point test ("is this point inside 0..105 x 0..68?") is WRONG: on an
    already-centered table it matches only the positive quadrant, shifting a
    quarter of the keypoints by (52.5, 34) and leaving the rest — a scrambled
    world model that produced ~1500 px systematic reprojection error on real
    data. The convention is a property of the table, not of a point.
    """
    x, y = arr[:, 0], arr[:, 1]
    uncentered = (x.min() >= -1e-6 and y.min() >= -1e-6 and
                  (x.max() > PITCH_L * 0.6 or y.max() > PITCH_W * 0.6))
    out = arr.copy()
    if uncentered:
        out[:, 0] -= PITCH_L / 2.0
        out[:, 1] -= PITCH_W / 2.0
    return out, uncentered


def load_keypoint_world(repo_dir, verbose=True):
    """Return {kp_index(1-based): np.array([x, y, z])} in PITCH-CENTERED
    coordinates, matching geometry.LINES_WORLD / CIRCLES_WORLD."""
    add_repo_to_path(repo_dir)
    import importlib
    mod = importlib.import_module("utils.utils_calib")
    table = None
    for name in _KP_WORLD_CANDIDATES:
        table = getattr(mod, name, None)
        if table is not None:
            break
    if table is None:
        raise RuntimeError(
            "Could not find the keypoint world-coordinate table in "
            "PnLCalib utils.utils_calib. Inspect that file and add the symbol "
            f"name to _KP_WORLD_CANDIDATES (tried {_KP_WORLD_CANDIDATES}).")

    arr = _table_to_array(table)
    aux = None
    for name in _KP_AUX_CANDIDATES:
        aux = getattr(mod, name, None)
        if aux is not None:
            break
    if aux is not None:
        arr = np.vstack([arr, _table_to_array(aux)])

    centered, was_uncentered = infer_and_center(arr)
    if verbose:
        print(f"[world] {len(centered)} keypoints; source frame "
              f"{'corner-origin (0..105 x 0..68) -> recentered' if was_uncentered else 'already pitch-centered'}"
              f"; x in [{centered[:,0].min():.1f}, {centered[:,0].max():.1f}], "
              f"y in [{centered[:,1].min():.1f}, {centered[:,1].max():.1f}]")
    # sanity: a valid pitch model must span roughly the real pitch
    if centered[:, 0].max() - centered[:, 0].min() < PITCH_L * 0.5:
        raise RuntimeError("keypoint world table does not span the pitch "
                           "- the coordinate convention is wrong")
    return {i + 1: centered[i] for i in range(len(centered))}


def circle_membership(kp_world, tol=0.75):
    """kp_index -> circle index (0..2) if the keypoint lies on a circle."""
    member = {}
    for idx, X in kp_world.items():
        if abs(X[2]) > 1e-6:
            continue
        for ci, (cx, cy, r) in enumerate(CIRCLES_WORLD):
            if abs(np.hypot(X[0] - cx, X[1] - cy) - r) < tol:
                member[idx] = ci
                break
    return member


# ----------------------------------------------------------------------------- detection dicts
def _entry_xy_conf(e):
    """Robustly read one detection entry: supports {'x','y'} plus a confidence
    under 'p' / 'score' / 'conf' (schema of PnLCalib's coords_to_dict may vary
    slightly across commits)."""
    x = e.get("x", e.get("x_1"))
    y = e.get("y", e.get("y_1"))
    c = e.get("p", e.get("score", e.get("conf", 1.0)))
    return float(x), float(y), float(c)


def _dict_is_normalized(vals):
    """Decide normalization for a WHOLE detection dict at once. Per-point
    heuristics are unsafe (a normalized coord slightly outside the frame would
    be misread as a pixel value), and a max-based rule is equally unsafe (one
    far-off-frame inferred point flips the whole dict). Majority vote: the
    dict is normalized iff most coordinates are small."""
    if not vals:
        return True
    small = sum(1 for v in vals if abs(v) <= 1.25)
    return small >= 0.5 * len(vals)


def keypoints_to_obs(kp_dict, image_wh, kp_world):
    """-> (idx, uv_px(N,2), conf(N,), Xw(N,3)) for keypoints with known world."""
    w, h = image_wh
    raw = []
    for k, e in kp_dict.items():
        k = int(k)
        if k not in kp_world:
            continue
        x, y, c = _entry_xy_conf(e)
        raw.append((k, x, y, c))
    norm = _dict_is_normalized([v for _, x, y, _ in raw for v in (x, y)])
    idxs, uvs, confs, Xs = [], [], [], []
    for k, x, y, c in raw:
        if norm:
            x, y = x * w, y * h
        if not (-0.25 * w <= x <= 1.25 * w and -0.25 * h <= y <= 1.25 * h):
            continue            # discard far-off-frame inferred points
        idxs.append(k); uvs.append([x, y]); confs.append(c); Xs.append(kp_world[k])
    if not idxs:
        return np.zeros(0, int), np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3))
    return (np.array(idxs), np.array(uvs, float),
            np.array(confs, float), np.array(Xs, float))


def lines_to_obs(lines_dict, image_wh):
    """-> list of (line_idx(1-based), endpoints_px(2,2), conf). Accepts either
    endpoint-dict entries or lists of two point-dicts."""
    w, h = image_wh
    out = []
    for k, e in lines_dict.items():
        k = int(k)
        pts, conf = [], 1.0
        if isinstance(e, dict) and "x_1" in e:
            pts = [[e["x_1"], e["y_1"]], [e["x_2"], e["y_2"]]]
            conf = float(e.get("p", e.get("score", 1.0)))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            cs = []
            for pe in e[:2]:
                x, y, c = _entry_xy_conf(pe)
                pts.append([x, y]); cs.append(c)
            conf = float(np.mean(cs))
        else:
            continue
        pts = np.array(pts, float)
        if _dict_is_normalized(list(np.abs(pts).ravel())):
            pts[:, 0] *= w; pts[:, 1] *= h
        if 1 <= k <= len(LINES_WORLD):
            out.append((k, pts, conf))
    return out


def load_world(repo_dir, tol=0.75):
    """One-call convenience: (kp_world, circle_membership). Use this at the
    top of any stage so cells are self-sufficient and resume-safe."""
    kp_world = load_keypoint_world(repo_dir)
    return kp_world, circle_membership(kp_world, tol=tol)
