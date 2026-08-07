"""SoccerNet pitch model, built parametrically and RESOLVED from data.

Why this exists: the previous name->geometry map was produced by zipping a
hand-written list of class names against PnLCalib's internal line ordering.
That pairing was never verified, and on real data it capped the achievable
reprojection error at ~80-105 px (a camera fitted directly to ground truth
could not do better), which is the signature of a partially mismatched map.

Here every element is derived from the official pitch dimensions and the
SEMANTICS of its name ("left"/"right" -> x, "top"/"bottom" -> y). That leaves
exactly two binary unknowns — which x sign "right" means and which y sign
"top" means — and `resolve_convention` determines them from the annotations
themselves instead of assuming.

Dimensions (metres, FIFA/SoccerNet): pitch 105 x 68; penalty area 16.5 deep,
40.32 wide (+-20.16); goal area 5.5 deep, 18.32 wide (+-9.16); goal 7.32 wide
(+-3.66), 2.44 high; circle radius 9.15; penalty mark 11 from the goal line.
"""
from __future__ import annotations
import itertools
import numpy as np

L, W = 105.0, 68.0
HL, HW = L / 2.0, W / 2.0
PEN_D, PEN_HW = 16.5, 20.16
GOAL_AREA_D, GOAL_AREA_HW = 5.5, 9.16
GOAL_HW, GOAL_H = 3.66, 2.44
R_CIRCLE = 9.15
PEN_MARK = 11.0

N_SAMPLES = 60
N_ARC = 240


def _seg(a, b, n=N_SAMPLES):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, 1, n)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def _circle(cx, cy, r=R_CIRCLE, n=N_ARC):
    a = np.linspace(0, 2 * np.pi, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros_like(a)], 1)


def build_name_map(right_is_positive_x=True, top_is_positive_y=True):
    """SoccerNet class name -> (N,3) world points, in pitch-centered metres.

    `right_is_positive_x`: does the "right" half sit at +x?
    `top_is_positive_y`:   does the "top" touchline sit at +y?
    """
    sx = 1.0 if right_is_positive_x else -1.0
    sy = 1.0 if top_is_positive_y else -1.0
    g = {}

    # touchlines and goal lines
    g["Side line top"] = _seg([-HL, sy * HW, 0], [HL, sy * HW, 0])
    g["Side line bottom"] = _seg([-HL, -sy * HW, 0], [HL, -sy * HW, 0])
    g["Side line left"] = _seg([-sx * HL, -HW, 0], [-sx * HL, HW, 0])
    g["Side line right"] = _seg([sx * HL, -HW, 0], [sx * HL, HW, 0])
    g["Middle line"] = _seg([0, -HW, 0], [0, HW, 0])

    for side, ss in (("left", -sx), ("right", sx)):
        gl = ss * HL                       # goal line x
        # penalty ("big") area
        inner = ss * (HL - PEN_D)
        g[f"Big rect. {side} main"] = _seg([inner, -PEN_HW, 0], [inner, PEN_HW, 0])
        g[f"Big rect. {side} top"] = _seg([gl, sy * PEN_HW, 0], [inner, sy * PEN_HW, 0])
        g[f"Big rect. {side} bottom"] = _seg([gl, -sy * PEN_HW, 0], [inner, -sy * PEN_HW, 0])
        # goal ("small") area
        inner_s = ss * (HL - GOAL_AREA_D)
        g[f"Small rect. {side} main"] = _seg([inner_s, -GOAL_AREA_HW, 0],
                                             [inner_s, GOAL_AREA_HW, 0])
        g[f"Small rect. {side} top"] = _seg([gl, sy * GOAL_AREA_HW, 0],
                                            [inner_s, sy * GOAL_AREA_HW, 0])
        g[f"Small rect. {side} bottom"] = _seg([gl, -sy * GOAL_AREA_HW, 0],
                                               [inner_s, -sy * GOAL_AREA_HW, 0])
        # goal frame (z is negative upwards in this world frame)
        g[f"Goal {side} crossbar"] = _seg([gl, -GOAL_HW, -GOAL_H],
                                          [gl, GOAL_HW, -GOAL_H])
        g[f"Goal {side} post left"] = _seg([gl, sy * GOAL_HW, 0],
                                           [gl, sy * GOAL_HW, -GOAL_H])
        g[f"Goal {side} post right"] = _seg([gl, -sy * GOAL_HW, 0],
                                            [gl, -sy * GOAL_HW, -GOAL_H])
        # penalty arc / circle
        g[f"Circle {side}"] = _circle(ss * (HL - PEN_MARK), 0.0)

    g["Circle central"] = _circle(0.0, 0.0)
    return g


CONVENTIONS = list(itertools.product([True, False], [True, False]))


def resolve_convention(ann, image_wh, n_frames=5, verbose=True,
                       ref_cameras=None):
    """Determine the (x, y) sign convention FROM THE DATA.

    Step 1 - fit a camera directly to the annotations under each of the four
    candidates. The correct ones reach a few pixels; the wrong ones cannot.

    NOTE a genuine degeneracy: flipping BOTH signs is a 180 deg rotation about
    z, a rigid transform the camera absorbs exactly, so (rx, ty) and
    (not rx, not ty) are indistinguishable from annotations alone. Annotations
    fix only the RELATIVE sign.

    Step 2 - break the tie with `ref_cameras` ({frame: Camera}, e.g. real
    PnLCalib estimates). Those cameras live in the KEYPOINT world frame, so
    whichever surviving candidate they fit is the frame everything must share.
    Without ref_cameras the tie is broken arbitrarily and a warning is issued.

    Returns (right_is_positive_x, top_is_positive_y, report_dict).
    """
    from .diagnose import fit_camera_to_annotations
    frames = [f for f in sorted(ann) if ann[f]][:n_frames]
    report = {}
    for rx, ty in CONVENTIONS:
        geom = build_name_map(rx, ty)
        errs = []
        for f in frames:
            _, e = fit_camera_to_annotations(ann[f], image_wh, n_starts=6,
                                             geom=geom)
            if np.isfinite(e):
                errs.append(e)
        report[(rx, ty)] = float(np.median(errs)) if errs else float("inf")
        if verbose:
            print(f"[pitch] right_x={'+' if rx else '-'} "
                  f"top_y={'+' if ty else '-'} -> best achievable "
                  f"{report[(rx, ty)]:8.2f} px")
    ranked = sorted(report, key=report.get)
    best = ranked[0]
    # the equivalent partner (both signs flipped) scores the same up to noise
    partner = (not best[0], not best[1])
    tie = abs(report[best] - report.get(partner, np.inf)) < max(
        1.0, 0.25 * report[best])
    if tie:
        if ref_cameras:
            from .experiments import reprojection_error
            scores = {}
            for cand in (best, partner):
                geom = build_name_map(*cand)
                es = []
                for f, cam in ref_cameras.items():
                    if f in ann and ann[f]:
                        e = reprojection_error(cam, ann[f], geom=geom)
                        if e is not None and np.isfinite(e):
                            es.append(e)
                scores[cand] = float(np.median(es)) if es else float("inf")
            if verbose:
                for cand, sc in scores.items():
                    print(f"[pitch] tie-break with real cameras "
                          f"{cand} -> {sc:8.2f} px")
            best = min(scores, key=scores.get)
        elif verbose:
            print("[pitch] WARNING: two conventions are equivalent under a "
                  "180 deg rotation; pass ref_cameras to disambiguate. "
                  "Metrics are unaffected, but the world frame may not match "
                  "the keypoint frame.")
    if verbose:
        print(f"[pitch] RESOLVED: right_is_positive_x={best[0]}, "
              f"top_is_positive_y={best[1]}  ({report[best]:.2f} px)")
        if min(report.values()) > 20:
            print("[pitch] WARNING: even the best convention exceeds 20 px — "
                  "the pitch model or the annotations need further inspection.")
    return best[0], best[1], report
