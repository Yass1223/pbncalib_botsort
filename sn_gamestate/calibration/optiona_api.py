"""Option A field registration (PnLCalib + temporal smoothing) as a TrackLab
``VideoLevelModule``.

Replaces the BroadTrack calibration stage, keeping the *same* image-to-pitch code
path (``sn_calibration_baseline.Camera.unproject_point_on_planeZ0`` via a helper
copied byte-for-byte out of ``broadtrack_api``) so ``bbox_pitch`` stays
comparable for the A/B.

Torch isolation
---------------
Everything PnLCalib, torch and the HRNet checkpoints touch is behind exactly one
function: :func:`compute_cameras`. The pipeline pins torch 1.13.1 while PnLCalib
asks for 2.3.1; whether 1.13.1 can read the v1.0.0 checkpoints is answered by
``scripts/verify_pnlcalib_env.py`` on Kaggle, not here. If that answer is "no",
the fix — a subprocess boundary, or re-serialised checkpoints — lands inside that
one function and nothing else in this module moves. The contract handling, the
camera conversion, the caching and the projection below are all
torch-independent and were written and tested without waiting for it.

Why ``s`` is computed here, and computed POST-SMOOTHING
------------------------------------------------------
The chosen method is ``smooth``, and **none of the non-tracker methods produce a
confidence**: ``smooth.py:230``, ``method_v2.py:392`` and ``method_v3.py:464``
all write ``s=np.nan`` literally. Only ``method="v1"`` (the ``BroadTrackLayer``
path) emits a real ``s``. That matters because completeness here is ~1.0 by
construction — a camera that locks on, degrades and then gets carried forward
produces no exception and no missing row — so ``s`` is the *only* evidence that
a camera is live rather than stale.

**This ``s`` is ours, not optiona_sfr's.** It is computed with
``tracker.jaccard_confidence`` — the same detection-precision measure
``s = |D n T| / |D|`` that BroadTrack's ``s < 0.5`` reinitialisation rule uses —
scored against each frame's own cached detections.

It is deliberately measured **after** smoothing and interpolation, on the camera
the module actually emits, not on the pre-smoothing PnLCalib seed. Scoring the
seed would answer the wrong question: an interpolated frame has no seed at all,
and a frame whose seed was good but whose smoothed result drifted is exactly the
case this number exists to catch.

``jaccard_confidence`` returns a hard ``1.0`` when ``cv2`` is unimportable inside
``optiona_sfr.tracker``; that branch is detected and warned about explicitly,
because a silent 1.0 would make every frame look perfect.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sn_calibration_baseline.camera import Camera
from tracklab.pipeline.videolevel_module import VideoLevelModule

from sn_gamestate.calibration.optiona_convert import (
    get_bbox_pitch,
    optiona_camera_to_sncalib,
    principal_point_offset,
)

log = logging.getLogger(__name__)

#: ``pp`` further than this from the image centre is legitimate but changes the
#: image size ``from_json_parameters`` records (``image_width = 2 * pp[0]``).
PP_WARN_PX = 2.0

#: Half-extents beyond which a projected pitch position is physically impossible.
#: The pitch is 105 x 68 m (|x| <= 52.5, |y| <= 34); players legitimately stand
#: off the field of play, so these are generous — anything outside is not a
#: player near the touchline, it is a broken unprojection.
OUT_OF_PITCH_X, OUT_OF_PITCH_Y = 60.0, 45.0

#: Median-``s`` level below which a sequence is flagged. **A starting heuristic,
#: not a calibrated threshold** — 0.5 is BroadTrack's online reinitialisation
#: cut, measured on a live tracker; ours is a post-smoothing score on a different
#: pipeline. Re-derive it from the first full run's distribution.
S_HEURISTIC = 0.5


# ======================================================================================
# THE TORCH / PnLCalib BOUNDARY — the only function in this module that imports
# torch, loads a checkpoint, or calls into the cloned PnLCalib repo.
# ======================================================================================
def compute_cameras(cfg, frame_paths, seq_id, device):
    """Run detection + calibration + smoothing for one sequence.

    Returns ``(cameras, confidences, image_wh, info)`` where ``cameras`` is
    ``{frame_index: optiona_sfr.geometry.Camera | None}`` (1-based frame index,
    matching ``optiona_sfr``'s own convention) and ``confidences`` is
    ``{frame_index: float}``.

    Everything version-fragile is contained here. Callers see plain data.
    """
    import cv2
    from optiona_sfr import tracker as _tracker
    from optiona_sfr.config import DetectorCfg, RefineCfg, SmoothCfg
    from optiona_sfr.detection import (cache_sequence, get_baseline_cameras,
                                       load_cached, select_baseline_variant)
    from optiona_sfr.experiments import frame_indices
    from optiona_sfr.pnl_adapter import (keypoints_to_obs, lines_to_obs,
                                         load_models, load_world)
    from optiona_sfr.smooth import smooth_pnlcalib

    repo = Path(str(cfg.pnlcalib_repo))
    if not (repo / "inference.py").is_file():
        raise RuntimeError(
            f"PnLCalib clone not found at '{repo}'. Run scripts/setup_pnlcalib.sh "
            "first, or point cfg.pnlcalib_repo at an existing checkout.")

    det_cfg = DetectorCfg(
        kp_threshold=float(getattr(cfg, "kp_threshold", 0.3434)),
        line_threshold=float(getattr(cfg, "line_threshold", 0.7867)),
        device=str(device),
    )
    det_cfg.weights_kp_path = str(cfg.weights_kp)
    det_cfg.weights_line_path = str(cfg.weights_line)
    for attr in ("weights_kp_path", "weights_line_path"):
        if not Path(getattr(det_cfg, attr)).is_file():
            raise RuntimeError(
                f"checkpoint missing: {getattr(det_cfg, attr)} ({attr}). "
                "scripts/setup_pnlcalib.sh downloads and verifies the v1.0.0 "
                "SV_kp / SV_lines assets.")

    cache_dir = Path(str(cfg.det_cache_dir)) / seq_id
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    det_cache_root = Path(str(cfg.det_cache_dir))

    kp_world, membership = load_world(repo)

    # Detection runs once per sequence and is cached per frame, so a re-run (or a
    # sharded multi-GPU run) skips the GPU entirely.
    need = not frame_indices(det_cache_root, seq_id)
    if need or bool(getattr(cfg, "refresh_detections", False)):
        models = load_models(det_cfg, repo)

        def _frames():
            for i, p in enumerate(frame_paths, start=1):
                img = cv2.imread(str(p))
                if img is None:
                    log.warning(f"[optiona] unreadable frame, skipped: {p}")
                    continue
                yield i, img

        n = cache_sequence(_frames(), seq_id, models, det_cfg, repo,
                           det_cache_root, compute_flow=False, subpixel=True)
        log.info(f"[optiona] {seq_id}: cached detections for {n} new frame(s)")

    idxs = frame_indices(det_cache_root, seq_id)
    if not idxs:
        raise RuntimeError(f"detection cache is empty for {seq_id}")
    image_wh = tuple(load_cached(det_cache_root, seq_id, idxs[0])["wh"])

    refine_cfg = RefineCfg()
    sm = SmoothCfg()
    method = str(getattr(cfg, "method", "smooth"))

    if bool(getattr(cfg, "select_variant", True)):
        # PnLCalib's native line refinement is catastrophic on some sequences and
        # essential on others (optiona README v3.2.0: 13.5 px vs 616.4 px on one,
        # the reverse on another). Choosing per frame by self-fit is what makes
        # the fixed choice safe.
        cams = select_baseline_variant(
            _Paths(det_cache_root, repo), seq_id, image_wh, refine_cfg, kp_world,
            membership, frames=idxs, margin=float(sm.variant_margin), verbose=False)
    else:
        cams = get_baseline_cameras(
            _Paths(det_cache_root, repo), seq_id, image_wh, refine_cfg, kp_world,
            membership, refine_lines=True, frames=idxs, verbose=False)

    info = {"method": method, "n_frames": len(idxs)}
    if method == "smooth":
        cams, sinfo = smooth_pnlcalib(
            idxs, cams,
            window=int(getattr(cfg, "smooth_window", sm.window)),
            polyorder=int(getattr(cfg, "smooth_polyorder", sm.polyorder)),
            interpolate=bool(getattr(cfg, "interpolate", sm.interpolate)),
            reject_k=float(getattr(cfg, "reject_k", sm.reject_k)),
            verbose=False)
        info.update(sinfo)
    elif method == "raw":
        pass                                    # PnLCalib per-frame, unsmoothed
    else:
        raise RuntimeError(
            f"unknown method '{method}'. Supported: 'smooth' (default, the "
            "measured winner) and 'raw' (PnLCalib per-frame, no smoothing). "
            "The 'v2'/'v3' variants live in optiona_sfr's experiment driver "
            "(experiments.run_sequence), which writes result CSVs and is not an "
            "API — wiring them here is a separate change.")

    # --- confidence, computed here because the smooth path emits none ---------
    if getattr(_tracker, "cv2", None) is None:
        log.warning("[optiona] cv2 unavailable inside optiona_sfr.tracker: "
                    "jaccard_confidence returns a hard 1.0, so `s` carries NO "
                    "information for this run and min_score cannot gate.")
    conf = {}
    dilate = int(getattr(cfg, "jaccard_dilate_px", 7))
    for fi in idxs:
        cam = cams.get(fi)
        if cam is None:
            conf[fi] = 0.0
            continue
        det = load_cached(det_cache_root, seq_id, fi)
        if det is None:
            conf[fi] = float("nan")
            continue
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        try:
            conf[fi] = float(_tracker.jaccard_confidence(
                cam, kp_obs[1], line_obs, image_wh, dilate))
        except Exception as exc:                # never let scoring kill a run
            log.debug(f"[optiona] jaccard_confidence failed on {fi}: {exc}")
            conf[fi] = float("nan")
    return cams, conf, image_wh, info


class _Paths:
    """Minimal stand-in for ``optiona_sfr.config.Paths``.

    The real dataclass carries Kaggle-specific defaults (``/kaggle/working`` and
    friends) that would be actively misleading here; the two attributes the
    detection helpers actually read are these.
    """

    def __init__(self, cache_dir, pnlcalib_repo):
        self.cache_dir = cache_dir
        self.pnlcalib_repo = pnlcalib_repo


# ======================================================================================
# Module — nothing below this line depends on torch.
# ======================================================================================
class OptionACalibration(VideoLevelModule):
    """Camera calibration + image-to-pitch projection via PnLCalib + smoothing."""

    input_columns = {"detection": ["bbox_ltwh", "image_id"], "image": []}
    output_columns = {"detection": ["bbox_pitch"], "image": ["parameters"]}

    def __init__(self, cfg, device, tracking_dataset=None):
        self.cfg = cfg
        self.device = device

        self.calib_dir = Path(str(cfg.calib_dir))
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.use_cached_json = bool(getattr(cfg, "use_cached_json", True))
        self.use_prev_parameters = bool(getattr(cfg, "use_prev_parameters", True))
        self.min_score = float(getattr(cfg, "min_score", 0.0))

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _frames_dir(metadatas: pd.DataFrame) -> Path:
        """Folder holding this video's frames (SNGS ``.../img1``)."""
        return Path(str(metadatas["file_path"].iloc[0])).parent

    @staticmethod
    def _sequence_name(frames_dir: Path) -> str:
        """`.../SNGS-116/img1` -> `SNGS-116` (falls back to the folder name)."""
        return frames_dir.parent.name or frames_dir.name

    def _build_payload(self, metadatas: pd.DataFrame, seq_name: str) -> dict:
        """Calibrate the sequence and convert to the cacheable JSON payload."""
        # Frame order comes from the pipeline's own file list, sorted by name, so
        # the 1-based index handed to optiona_sfr matches %06d.jpg ordering.
        files = sorted((Path(str(p)) for p in metadatas["file_path"]),
                       key=lambda p: p.name)
        cams, conf, image_wh, info = compute_cameras(
            self.cfg, files, seq_name, self.device)

        frames, n_pp = {}, 0
        for i, path in enumerate(files, start=1):
            cam = cams.get(i)
            if cam is None:
                continue
            if principal_point_offset(cam) > PP_WARN_PX:
                n_pp += 1
            frames[path.name] = {
                "parameters": optiona_camera_to_sncalib(cam),
                "s": float(conf.get(i, float("nan"))),
            }
        if n_pp:
            log.warning(
                f"[optiona] {seq_name}: {n_pp}/{len(frames)} frames carry a "
                f"principal point more than {PP_WARN_PX} px off centre. That is "
                "PnLCalib's own estimate and is preserved deliberately, but note "
                "from_json_parameters derives image_width = 2*pp[0].")
        return {"sequence": seq_name, "image_wh": list(image_wh),
                "info": info, "frames": frames}

    def _log_s_stats(self, seq_name, s_by_order, first_lock, n_frames):
        """`s` is the only staleness signal — completeness cannot show one."""
        vals = np.array([v for v in s_by_order if v is not None and np.isfinite(v)],
                        dtype=float)
        if vals.size == 0:
            log.warning(f"[optiona] {seq_name}: no usable confidence values; "
                        "staleness cannot be assessed for this sequence.")
            return
        lock = "never" if first_lock is None else f"frame {first_lock}"
        log.info(
            f"[optiona] {seq_name}: s min={vals.min():.3f} median="
            f"{float(np.median(vals)):.3f} max={vals.max():.3f} "
            f"(n={vals.size}/{n_frames}); first lock-on {lock}; "
            f"below min_score={self.min_score}: {int((vals < self.min_score).sum())}")
        if vals.size and float(np.median(vals)) < S_HEURISTIC:
            log.warning(
                f"[optiona] {seq_name}: median s {float(np.median(vals)):.3f} is "
                f"below {S_HEURISTIC}. Completeness will still read ~1.0 — treat "
                "these pitch coordinates as suspect until checked. NOTE this "
                "threshold is a STARTING HEURISTIC, not a calibrated cut: 0.5 is "
                "borrowed from BroadTrack's ONLINE reinitialisation rule, which "
                "scores a live tracker frame-by-frame, whereas this s is computed "
                "post-smoothing on a different pipeline. The right cut for this "
                "stage has not been measured yet — do that from the first full "
                "run's distribution rather than trusting the borrowed number.")

    @staticmethod
    def _log_pitch_bounds(seq_name, bbox_pitch):
        """Pure diagnostic: count projected positions that cannot be real.

        ``get_bbox_pitch`` rejects an entry only when a coordinate is NaN. A bbox
        whose bottom edge sits ABOVE the horizon still yields a finite
        intersection with Z=0 — behind the camera — so it returns
        plausible-looking, meaningless pitch coordinates instead of ``None``.

        The helper is left byte-identical to ``broadtrack_api``'s on purpose, so
        the A/B compares like with like; that means this failure mode is shared
        by BOTH branches. It is therefore measured rather than silently absorbed:
        a cheirality check on the unprojected ray would fix it, but it must be
        applied to both paths at once or neither, and only after baseline numbers
        are recorded. This counter IS that baseline. Nothing here modifies a
        single coordinate.

        **This count is a LOWER BOUND, not a measurement.** It detects only the
        behind-camera unprojections that happen to land far outside the pitch. An
        above-horizon bbox can just as easily unproject to a coordinate *inside*
        60 x 45 m by coincidence, and that one is counted as fine. So a rising
        number is strong evidence of a problem, but **"0% outside" does not mean
        "no cheirality problem"** — it means none was caught by this test. Read it
        the way you would read completeness on this stage: absence of a signal,
        not evidence of correctness.
        """
        vals = [v for v in bbox_pitch.values() if isinstance(v, dict)]
        if not vals:
            return
        bad = sum(
            1 for v in vals
            if abs(float(v["x_bottom_middle"])) > OUT_OF_PITCH_X
            or abs(float(v["y_bottom_middle"])) > OUT_OF_PITCH_Y
        )
        pct = 100.0 * bad / len(vals)
        msg = (f"[optiona] {seq_name}: {bad}/{len(vals)} projected positions "
               f"({pct:.1f}%) outside |x|<={OUT_OF_PITCH_X} / "
               f"|y|<={OUT_OF_PITCH_Y} m")
        if bad == 0:
            log.info(msg + " — none")
        else:
            log.warning(
                msg + ". These are not players near the touchline; they are "
                "bboxes whose bottom edge is above the horizon, unprojected "
                "behind the camera. Values are left untouched for A/B parity — "
                "compare this count against the BroadTrack run, which shares the "
                "same helper and the same hole.")

    # ---------------------------------------------------------------- main
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # Engine contract: return the detections DataFrame ONLY. The image-level
        # 'parameters' column is written on `metadatas` in place, which the
        # offline engine persists (image_pred is the same object across the loop).
        if len(metadatas) == 0:
            return detections

        frames_dir = self._frames_dir(metadatas)
        seq_name = self._sequence_name(frames_dir)
        out_json = self.calib_dir / f"{seq_name}.json"

        payload = None
        if self.use_cached_json and out_json.is_file():
            try:
                payload = json.loads(out_json.read_text())
                log.info(f"[optiona] using cached calibration: {out_json}")
            except (OSError, json.JSONDecodeError) as e:
                log.warning(f"[optiona] cache unreadable ({e}); recomputing.")
                payload = None

        if payload is None:
            try:
                payload = self._build_payload(metadatas, seq_name)
            except Exception as e:
                log.error(f"[optiona] calibration failed for {seq_name}: "
                          f"{type(e).__name__}: {e}")
                return self._empty_outputs(detections, metadatas)
            try:
                out_json.write_text(json.dumps(payload))
            except OSError as e:
                log.warning(f"[optiona] could not write {out_json}: {e}")

        by_name = payload.get("frames", {}) or {}
        log.info(f"[optiona] {seq_name}: {len(by_name)} calibrated frames "
                 f"for {len(metadatas)} images")

        params_rows, bbox_pitch = {}, {}
        last_params = None
        n_carried = n_lowscore = 0
        s_seen, first_lock = [], None

        # Temporal order matters for carry-forward: iterate sorted by file name
        # (%06d.jpg sorts chronologically); do not rely on incoming row order.
        order = metadatas["file_path"].astype(str).map(lambda p: Path(p).name)
        for image_id in order.sort_values().index:
            name = order[image_id]
            entry = by_name.get(name)
            params = None
            if entry is not None:
                s = entry.get("s")
                s = float(s) if s is not None else float("nan")
                s_seen.append(s)
                # NaN means "not measurable", not "bad" — do not discard on it.
                if not np.isfinite(s) or s >= self.min_score:
                    params = entry.get("parameters") or None
                else:
                    n_lowscore += 1
            if params is not None and first_lock is None:
                first_lock = name
            if params is None and self.use_prev_parameters and last_params is not None:
                params = last_params
                n_carried += 1
            if params is None:
                params_rows[image_id] = {}
                continue
            last_params = params
            params_rows[image_id] = params

            cam = Camera()
            cam.from_json_parameters(params)
            image_dets = detections[detections.image_id == image_id]
            if len(image_dets):
                projected = image_dets.bbox.ltrb().apply(get_bbox_pitch(cam))
                bbox_pitch.update(projected.to_dict())

        if n_carried or n_lowscore:
            log.info(f"[optiona] {seq_name}: carried-forward {n_carried} frame(s), "
                     f"{n_lowscore} below min_score={self.min_score}")
        self._log_s_stats(seq_name, s_seen, first_lock, len(metadatas))
        self._log_pitch_bounds(seq_name, bbox_pitch)

        detections = detections.copy()
        col = pd.Series(bbox_pitch, dtype=object).reindex(detections.index)
        detections["bbox_pitch"] = col.where(col.notna(), None)  # NaN -> None

        params_col = pd.Series(params_rows, dtype=object).reindex(metadatas.index)
        metadatas["parameters"] = params_col.apply(
            lambda v: v if isinstance(v, dict) else {}
        )
        return detections

    @staticmethod
    def _empty_outputs(detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections = detections.copy()
        detections["bbox_pitch"] = None
        metadatas["parameters"] = pd.Series(
            [{} for _ in range(len(metadatas))], index=metadatas.index, dtype=object
        )
        return detections
