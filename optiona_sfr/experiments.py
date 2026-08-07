"""Sequence runner, GSR data access, metrics, ablation driver.

Metrics implemented here: mean/median reprojection error against the GSR
point-along-line annotations, completeness rate, and temporal-coherence
diagnostics (std of frame-to-frame deltas in pan/tilt/f, total focal-point
travel — the BroadTrack Fig. 3 diagnostics). The official JaC evaluator is
invoked from the cloned PnLCalib repo's `sn_calibration` package so that
challenge numbers are produced by the reference implementation, not by a
reimplementation.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from .geometry import Camera, LINES_WORLD, CIRCLES_WORLD, PITCH_L, PITCH_W
from .pnl_adapter import keypoints_to_obs, lines_to_obs
from .refine import build_circle_obs
from .tracker import BroadTrackLayer
from .detection import load_cached, make_single_frame_calibrator


# ----------------------------------------------------------------------------- GSR dataset access
def list_gsr_sequences(gsr_root, split="test"):
    """SoccerNet-GSR layout: <root>/<split>/<SNGS-xxx>/{img1/*.jpg,
    Labels-GameState.json}. Only directories that actually contain an img1/
    folder count as sequences; on failure the error shows what IS on disk."""
    root = Path(gsr_root) / split
    if not root.exists():
        parent = Path(gsr_root)
        found = sorted(p.name for p in parent.iterdir()) if parent.exists() \
            else "<parent missing>"
        raise FileNotFoundError(
            f"GSR split not found at {root}.\n"
            f"Contents of {parent}: {found}\n"
            "Run Stage 1 (kaggle_setup.ensure_gsr_data) — it is resume-capable "
            "and will download/unzip whatever is missing — or point "
            "Paths.gsr_root at the directory that CONTAINS train/valid/test.")
    seqs = sorted([d for d in root.iterdir()
                   if d.is_dir() and (d / "img1").is_dir()])
    if not seqs:
        present = sorted(p.name for p in root.iterdir())[:20]
        raise FileNotFoundError(
            f"No sequence directories (with img1/) under {root}. "
            f"Found instead: {present}. If these are nested one level deeper, "
            "re-run kaggle_setup.ensure_gsr_data to flatten the layout.")
    return seqs


def iter_frames(seq_dir):
    import cv2
    img_dir = seq_dir / "img1"
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for i, f in enumerate(files, start=1):
        yield i, cv2.imread(str(f))


def load_pitch_annotations(seq_dir, labels_dir=None, image_wh=None,
                           verbose=False):
    """Extract per-frame annotated pitch points {frame_idx: {class: (N,2) px}}
    from Labels-GameState.json. Falls back to the harvested copy under
    `labels_dir/<seq_name>/` — needed once the split has been released from
    scratch storage. Returns {} with a warning if the pitch field is absent
    (schema drift is reported, never silently ignored)."""
    seq_dir = Path(seq_dir)
    p = seq_dir / "Labels-GameState.json"
    if not p.exists() and labels_dir is not None:
        p = Path(labels_dir) / seq_dir.name / "Labels-GameState.json"
    if not p.exists():
        print(f"[warn] no Labels-GameState.json for {seq_dir.name} "
              f"(looked in seq dir and labels_dir={labels_dir})")
        return {}
    data = json.loads(p.read_text())
    out = {}
    for ann in data.get("annotations", []):
        # pitch annotations carry a 'lines' dict of named point chains
        lines = ann.get("lines")
        if not lines:
            continue
        img_id = str(ann.get("image_id", ""))
        fidx = int(img_id[-6:]) if img_id[-6:].isdigit() else None
        if fidx is None:
            continue
        d = {}
        for cls, pts in lines.items():
            arr = np.array([[q["x"], q["y"]] for q in pts], float)
            d[cls] = arr
        out[fidx] = d
    if not out:
        print(f"[warn] Labels-GameState.json in {seq_dir} contains no pitch "
              "line annotations under annotations[*].lines — check schema.")
        return out

    # SoccerNet-GSR stores pitch annotations NORMALIZED to [0, 1]. Read as
    # pixels they collapse onto the image corner, ~1100 px from where the
    # pitch actually projects — which is exactly the systematic error observed
    # on real data. Detect the convention from the data and denormalize.
    allpts = np.concatenate([np.concatenate(list(d.values()))
                             for d in out.values() if d])
    normalized = np.nanmax(np.abs(allpts)) <= 1.5
    if normalized:
        if image_wh is None:
            image_wh = (1920, 1080)
            if verbose:
                print("[ann] normalized coordinates detected; no image size "
                      "given, assuming 1920x1080")
        w, h = image_wh
        for d in out.values():
            for cls in d:
                d[cls] = d[cls] * np.array([w, h], float)
        if verbose:
            print(f"[ann] normalized annotations -> denormalized by {w}x{h}")
    elif verbose:
        print("[ann] annotations already in pixel coordinates "
              f"(max |coord| = {np.nanmax(np.abs(allpts)):.1f})")
    return out


# Map SoccerNet class names to world geometry. The convention (which x sign
# "right" means, which y sign "top" means) is RESOLVED FROM DATA — see
# pitch_model.resolve_convention — never guessed.
_NAME2GEOM = None
_CONVENTION = (True, True)      # (right_is_positive_x, top_is_positive_y)


def set_pitch_convention(right_is_positive_x, top_is_positive_y):
    """Install the resolved convention; invalidates the cached geometry."""
    global _CONVENTION, _NAME2GEOM
    _CONVENTION = (bool(right_is_positive_x), bool(top_is_positive_y))
    _NAME2GEOM = None
    return _CONVENTION


def _world_geometry_by_name():
    global _NAME2GEOM
    if _NAME2GEOM is None:
        from .pitch_model import build_name_map
        _NAME2GEOM = build_name_map(*_CONVENTION)
    return _NAME2GEOM


def reprojection_error(cam: Camera, ann_frame, geom=None):
    """Mean px distance from each annotated point to the nearest projected
    point of its world element (sn-calibration-style reprojection error)."""
    geom = geom if geom is not None else _world_geometry_by_name()
    errs = []
    for cls, pts in ann_frame.items():
        W = geom.get(cls)
        if W is None:
            continue
        prj, front = cam.project(W)
        prj = prj[front]
        prj = prj[np.isfinite(prj).all(axis=1)]
        if len(prj) < 2:
            continue
        d = np.linalg.norm(pts[:, None, :] - prj[None, :, :], axis=2).min(1)
        errs.append(d[np.isfinite(d)])
    if not errs:
        return None
    e = np.concatenate(errs)
    if e.size == 0 or not np.isfinite(e).all():
        return None
    return float(np.mean(e))


# ----------------------------------------------------------------------------- result invalidation
def _result_stamp(exp_cfg):
    """Fingerprint of the package version + the parts of the config that
    change results. Cached CSVs from a different fingerprint are recomputed."""
    import hashlib, dataclasses, json as _json
    from . import __version__
    payload = {
        "version": __version__,
        "refine": dataclasses.asdict(exp_cfg.refine),
        "tracker": {k: v for k, v in dataclasses.asdict(exp_cfg.tracker).items()},
        # The pitch sign convention changes every reprojection number, so it
        # MUST invalidate cached results — otherwise resolving the convention
        # and re-running silently re-reports the old numbers.
        "pitch_convention": list(_CONVENTION),
        "method": getattr(exp_cfg, "method", "v1"),
        "v2": dataclasses.asdict(exp_cfg.v2) if hasattr(exp_cfg, "v2") else {},
        "v3": dataclasses.asdict(exp_cfg.v3) if hasattr(exp_cfg, "v3") else {},
        "smooth": dataclasses.asdict(exp_cfg.smooth) if hasattr(exp_cfg, "smooth") else {},
    }
    return hashlib.sha1(
        _json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- sequence runner
def run_sequence(seq_dir, exp_cfg, kp_world, membership, use_frames=None):
    """Run one experiment config on one sequence from the detection cache.
    Writes <results>/<exp>/<seq>.csv (resume-safe) and returns the DataFrame."""
    paths = exp_cfg.paths
    seq_id = seq_dir.name
    out_dir = Path(paths.results_dir) / exp_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{seq_id}.csv"
    meta = out_dir / f"{seq_id}.meta"
    # Results are cached per (sequence, experiment) — but they MUST be
    # invalidated when the code or the config changes, otherwise a fixed
    # pipeline silently re-reports the old broken numbers.
    stamp = _result_stamp(exp_cfg)
    if out_csv.exists() and meta.exists() and meta.read_text() == stamp:
        return pd.read_csv(out_csv)

    if getattr(exp_cfg, "method", "v1") == "smooth":
        from .smooth import run_sequence_smooth
        df = run_sequence_smooth(seq_dir, exp_cfg, kp_world, membership)
        df.to_csv(out_csv, index=False)
        meta.write_text(stamp)
        return df
    if getattr(exp_cfg, "method", "v1") == "v3":
        from .method_v3 import run_sequence_v3
        df = run_sequence_v3(seq_dir, exp_cfg, kp_world, membership)
        df.to_csv(out_csv, index=False)
        meta.write_text(stamp)
        return df
    if getattr(exp_cfg, "method", "v1") == "v2":
        from .method_v2 import run_sequence_v2
        df = run_sequence_v2(seq_dir, exp_cfg, kp_world, membership)
        df.to_csv(out_csv, index=False)
        meta.write_text(stamp)
        return df
    ann = None   # loaded after image_wh is known
    # Frame list comes from the DETECTION CACHE, not from disk images: after
    # Stage 2 the pixels can be pruned (chunked staging), and every stage from
    # here on runs from cached detections + cached optical-flow pairs.
    cdir = Path(paths.cache_dir) / seq_id
    cached = frame_indices(paths.cache_dir, seq_id)
    if not cached:
        raise RuntimeError(f"No detection cache for {seq_id}; run Stage 2.")
    idxs = cached if use_frames is None else [i for i in cached if i in set(use_frames)]

    first = load_cached(paths.cache_dir, seq_id, idxs[0])
    image_wh = tuple(first["wh"])
    ann = load_pitch_annotations(seq_dir, getattr(paths, "labels_dir", None),
                                 image_wh=image_wh)

    calibrate = make_single_frame_calibrator(
        paths.pnlcalib_repo, image_wh, exp_cfg.refine, kp_world, membership)

    tracker = BroadTrackLayer(exp_cfg.refine, exp_cfg.tracker, image_wh) \
        if exp_cfg.tracker.enabled else None

    rows = []
    for fi in idxs:
        det = load_cached(paths.cache_dir, seq_id, fi)
        if det is None:
            rows.append(dict(frame=fi, ok=0)); continue
        ctx = {"kp": det["kp"], "lines": det["lines"]}
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        circle_obs = build_circle_obs(kp_obs[0], kp_obs[1], kp_obs[2],
                                      membership)
        if tracker is not None:
            cam, s = tracker.step(kp_obs, line_obs, circle_obs,
                                  flow_pair=det.get("flow"),
                                  single_frame_calibrate=calibrate,
                                  frame_ctx=ctx)
        else:
            cam, s = calibrate(ctx), np.nan
        row = dict(frame=fi, ok=int(cam is not None), s=s)
        if cam is not None:
            row.update(pan=cam.pan, tilt=cam.tilt, roll=cam.roll, f=cam.f,
                       cx=cam.C[0], cy=cam.C[1], cz=cam.C[2], k1=cam.k1)
            re = reprojection_error(cam, ann.get(fi, {}))
            row["reproj_px"] = re
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    meta.write_text(stamp)
    return df


# ----------------------------------------------------------------------------- aggregation
def summarize(exp_name, results_dir):
    files = sorted((Path(results_dir) / exp_name).glob("*.csv"))
    per_seq = []
    for f in files:
        df = pd.read_csv(f)
        ok = df["ok"] == 1
        re = df.loc[ok, "reproj_px"].dropna()
        d = dict(seq=f.stem,
                 completeness=float(ok.mean()),
                 mre=float(re.mean()) if len(re) else np.nan,
                 medre=float(re.median()) if len(re) else np.nan,
                 n_frames=len(df))
        if ok.sum() > 2:
            sub = df.loc[ok]
            for p in ["pan", "tilt", "f"]:
                d[f"jitter_{p}"] = float(np.std(np.diff(sub[p].values)))
            C = sub[["cx", "cy", "cz"]].values
            d["focal_travel_m"] = float(
                np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))
        per_seq.append(d)
    if not per_seq:
        # A config that was never run (e.g. the A-grid with RUN_A=False) has no
        # result files. Return empty frames rather than raising, so Stage 5 can
        # summarise whatever HAS been computed.
        return pd.DataFrame(), pd.DataFrame()
    per = pd.DataFrame(per_seq)
    agg = per.drop(columns=["seq"], errors="ignore") \
             .mean(numeric_only=True).to_frame(exp_name).T
    return per, agg


def frame_indices(cache_dir, seq_id):
    """Frame numbers present in a sequence cache.

    ONE place enumerates frames. Three separate `int(p.stem)` globs previously
    existed; guarding two of them still left the v2/v3 runners crashing on the
    baseline pickle (`invalid literal for int(): '_baseline_rl1'`).
    """
    d = Path(cache_dir) / str(seq_id)
    if not d.is_dir():
        return []
    return sorted(int(p.stem) for p in d.glob("*.pkl") if p.stem.isdigit())


def cached_sequence_ids(cache_dir):
    """Sequence ids present in the detection cache — the source of truth for
    every stage after caching. Works in a fresh session with released splits:
    the images are gone, but the cache (and harvested labels) persist."""
    d = Path(cache_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and not p.name.startswith("_")
                  and any(q.stem.isdigit() for q in p.glob("*.pkl")))
