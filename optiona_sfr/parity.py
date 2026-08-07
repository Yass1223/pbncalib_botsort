"""Phase 0 — diagnose gross topological failures (e.g. SNGS-118, 612.83 px).

An error two orders of magnitude above the 7-10 px seen on healthy sequences is
categorical, not gradual. Three hypotheses, tested in order:

  H1 PITCH-HALF AMBIGUITY. The pitch has a 2-fold symmetry: rotating 180 deg
     about the world vertical and mapping C -> (-Cx, -Cy, Cz) maps the pitch
     onto itself. This is a PROPER transform (det = +1) and is exactly what
     PnLCalib's disambiguation resolves — so it is what can fail.
     NOT a reflection: reflections have det = -1, and every audited frame had
     det(R) = +1.000, so no improper rotation exists to undo.
  H2 DIFFERENT CAMERA TYPE (behind-goal, extreme zoom): no transform helps and
     abstention is the correct answer.
  H3 DETECTOR FAILURE: too few or wrong detections; fix upstream, not here.

Nothing is transformed until the diagnosis names a cause.
"""
from __future__ import annotations
import numpy as np

from .geometry import Camera


def flip_pitch_half(cam: Camera) -> Camera:
    """Apply the pitch's 2-fold symmetry: 180 deg about the world z axis.

    R' = R @ Rz(pi)^T and C' = (-Cx, -Cy, Cz). Proper (det preserved), and the
    field template is invariant under it — which is precisely why the
    ambiguity exists.
    """
    Rz = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    from .geometry import R_to_ptr
    Rn = cam.R @ Rz.T
    pan, tilt, roll = R_to_ptr(Rn)
    Cn = np.array([-cam.C[0], -cam.C[1], cam.C[2]])
    out = Camera(cam.f, pan, tilt, roll, Cn, cam.k1, (cam.w, cam.h))
    out.pp = cam.pp.copy()
    return out


def diagnose_sequence_failure(seq_dir, cfg, kp_world, membership,
                              frames=None, n_sample=60, labels_dir=None,
                              verbose=True):
    """Return a verdict dict naming the cause and whether a fix applies.

    Samples `n_sample` EVENLY SPACED frames by default. An earlier version used
    four fixed frames and was badly misleading: it reported SNGS-118 as healthy
    at 13.5 px when its true median over 750 frames is 616 px, and SNGS-119 as
    broken at 902 px when its true median is 10.5 px. Both were sampling
    accidents, and one of them sent a whole investigation down the wrong path.
    """
    from pathlib import Path
    from .detection import load_cached, make_single_frame_calibrator
    from .pnl_adapter import keypoints_to_obs, lines_to_obs
    from .experiments import load_pitch_annotations, reprojection_error
    from .refine import score_camera

    seq_dir = Path(seq_dir)
    labels_dir = labels_dir or getattr(cfg.paths, "labels_dir", None)
    first = load_cached(cfg.paths.cache_dir, seq_dir.name, 1)
    if first is None:
        return {"verdict": "no-cache"}
    image_wh = tuple(first["wh"])
    ann = load_pitch_annotations(seq_dir, labels_dir, image_wh=image_wh)
    cal = make_single_frame_calibrator(cfg.paths.pnlcalib_repo, image_wh,
                                       cfg.refine, kp_world, membership,
                                       use_refine=False)
    if frames is None:
        from .experiments import frame_indices
        all_idx = frame_indices(cfg.paths.cache_dir, seq_dir.name)
        step = max(1, len(all_idx) // max(n_sample, 1))
        frames = all_idx[::step][:n_sample] or all_idx
    rows, n_det, n_none = [], [], 0
    for fi in frames:
        det = load_cached(cfg.paths.cache_dir, seq_dir.name, fi)
        if det is None or fi not in ann:
            continue
        kp_obs = keypoints_to_obs(det["kp"], image_wh, kp_world)
        line_obs = lines_to_obs(det["lines"], image_wh)
        n_det.append((len(kp_obs[0]), len(line_obs)))
        cam = cal({"kp": det["kp"], "lines": det["lines"]})
        if cam is None:
            n_none += 1
            continue
        e_direct = reprojection_error(cam, ann[fi])
        e_flip = reprojection_error(flip_pitch_half(cam), ann[fi])
        # self-consistency: does the camera fit its OWN detections?
        s_self = score_camera(cam, kp_obs, line_obs, cfg.refine)
        rows.append((fi, e_direct, e_flip, s_self, cam))
    if not rows:
        return {"verdict": "no-solution",
                "note": "PnLCalib produced no camera on the sampled frames"}

    dir_e = np.array([r[1] if r[1] is not None else np.nan for r in rows])
    flp_e = np.array([r[2] if r[2] is not None else np.nan for r in rows])
    self_e = np.array([r[3] for r in rows])
    md, mf, ms = (np.nanmedian(dir_e), np.nanmedian(flp_e), np.nanmedian(self_e))
    kp_med = np.median([d[0] for d in n_det]) if n_det else 0
    ln_med = np.median([d[1] for d in n_det]) if n_det else 0

    if verbose:
        print(f"[parity] {seq_dir.name}: annotation err direct {md:8.1f} px | "
              f"pitch-half-flipped {mf:8.1f} px  ({len(rows)} frames sampled)")
        print(f"[parity]   self-fit on own detections {ms:6.1f} px | "
              f"median detections: {kp_med:.0f} kps, {ln_med:.0f} lines")

    if np.isfinite(md) and md < 20.0:
        verdict = "healthy"
        note = f"direct annotation error {md:.1f} px — no failure to diagnose"
        if verbose:
            print(f"[parity]   VERDICT: {verdict} — {note}")
        return {"verdict": verdict, "note": note, "direct": md, "flipped": mf,
                "self_fit": ms, "kp_median": kp_med, "line_median": ln_med}
    if np.isfinite(mf) and mf < 0.5 * md and mf < 50.0:
        verdict = "H1-pitch-half"
        note = ("the 180 deg pitch symmetry was resolved the wrong way; "
                "flipping fixes it")
    elif ms > 30.0 or kp_med < 6:
        verdict = "H3-detector"
        note = ("the camera does not even fit its own detections, or there are "
                "too few — a detector problem, not a geometry one")
    else:
        verdict = "H2-camera-type-or-hard-view"
        note = ("geometry is self-consistent but disagrees with the pitch "
                "model; likely a view this method does not cover — abstaining "
                "is correct")
    if verbose:
        print(f"[parity]   VERDICT: {verdict} — {note}")
    return {"verdict": verdict, "note": note, "direct": md, "flipped": mf,
            "self_fit": ms, "kp_median": kp_med, "line_median": ln_med}
