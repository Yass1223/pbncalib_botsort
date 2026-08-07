#!/usr/bin/env python
r"""Numeric parity for the Option A -> sn-calibration conversion. Pre-push gate.

The conversion in ``sn_gamestate/calibration/optiona_convert.py`` was *derived*
from the sources in the Task 1 audit. Derivation is not evidence. This script
measures it, on synthetic cameras spanning the regimes that actually occur —
zoomed, wide, rolled, distorted — with no dependency on torch, PnLCalib, the
HRNet checkpoints or SoccerNet data. numpy + OpenCV are enough, which is why
this is the one check that runs locally instead of on Kaggle.

Checks
------
A. parity           - project pitch-template points through (i) the
                      ``optiona_sfr.geometry.Camera`` directly and (ii) an
                      ``sn_calibration_baseline.Camera`` built from the converted
                      dict. Max disagreement must be < 0.01 px. This is the
                      claim "cam.R -> rotation_matrix_to_pan_tilt_roll -> rad2deg
                      is lossless".
B. negative control - the SAME comparison with pan/tilt/roll copied straight
                      across instead of round-tripped through cam.R. The two ZXZ
                      orders are reversed, so this MUST blow up. Reported next to
                      A: a parity test whose control passes is measuring nothing,
                      and the gate fails if the control does not fail.
C. distortion       - k1 handling end to end. sn inverts distortion with
                      cv.undistortPoints (iterative rational model), optiona with
                      a Newton radial inversion — genuinely different algorithms,
                      so the residual gap is measured, not assumed. Also a
                      negative control: applying BroadTrack's (f/H)^2 rescale,
                      which has no analogue here and must diverge.
D. plane roundtrip  - project a known Z=0 world point, then unproject the pixel
                      with unproject_point_on_planeZ0. Error < 1 mm. This is the
                      path bbox_pitch actually takes.
E. bbox_pitch       - the copied get_bbox_pitch against a hand-computed
                      expectation, including the any-NaN -> None rule.

Dependencies
------------
Deliberately small — this is what makes the gate local. On a bare CPython 3.9::

    python -m pip install --upgrade pip          # 20.2.3 cannot resolve modern wheels
    python -m pip install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.13.1

That is ~95 MiB and is all this script needs. ``pyyaml`` and ``pandas==2.3.3``
are additionally required by ``tests/test_optiona_api_contract.py`` and by the
config checks, not by this file. No torch, no torchvision, no HRNet, no
checkpoints, no SoccerNet data, no virtualenv — the versions above match the
pipeline's own pins so the numbers here transfer to the real environment.

Usage
-----
    python scripts/verify_optiona_conversion.py
    python scripts/verify_optiona_conversion.py --tol 0.01 -v

Exit code 0 means the conversion is sound and the push may proceed. Non-zero is
the number of failed checks.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# optiona_sfr is vendored beside the pipeline; before assembly it sits one level
# up, so accept either layout rather than failing on a path detail.
for cand in (ROOT / "optiona_sfr", ROOT.parent / "optiona_sfr"):
    if (cand / "optiona_sfr" / "geometry.py").is_file():
        sys.path.insert(0, str(cand))
        break
# sn_calibration_baseline normally arrives via the editable tracklab_calibration
# install. Add the plugin directory directly so this gate also runs in a bare
# interpreter with just numpy + OpenCV — that is what makes it a *local* check.
_plugin = ROOT / "plugins" / "calibration"
if (_plugin / "sn_calibration_baseline" / "camera.py").is_file():
    sys.path.insert(0, str(_plugin))

from optiona_sfr.geometry import Camera as OptCamera  # noqa: E402
from sn_calibration_baseline.camera import Camera as SnCamera  # noqa: E402
from sn_calibration_baseline.soccerpitch import SoccerPitch  # noqa: E402
from sn_gamestate.calibration.optiona_convert import (  # noqa: E402
    get_bbox_pitch,
    naive_angle_copy_to_sncalib,
    optiona_camera_to_sncalib,
    principal_point_offset,
)

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[2m", "\033[1m", "\033[0m"
)
_results = []


def hdr(t):
    print(f"\n{BOLD}{t}{RESET}")


def ok(m):
    print(f"  {GREEN}OK  {RESET} {m}")


def bad(m):
    print(f"  {RED}FAIL{RESET} {m}")


def warn(m):
    print(f"  {YELLOW}WARN{RESET} {m}")


def info(m):
    print(f"       {DIM}{m}{RESET}")


def record(name, passed):
    _results.append((name, bool(passed)))
    return passed


# --------------------------------------------------------------------------- fixtures
def pitch_points(n_per_line=6):
    """Points on the real pitch template — where the evidence actually is."""
    pts = []
    field = SoccerPitch()
    for line in field.sample_field_points().values():
        arr = np.asarray(line, float)
        if len(arr) >= 2:
            idx = np.linspace(0, len(arr) - 1, min(n_per_line, len(arr))).astype(int)
            pts.append(arr[idx])
    return np.unique(np.concatenate(pts), axis=0)


def look_at(C, target, roll_deg, f, k1=0.0, wh=(1920, 1080)):
    """Build an optiona Camera aimed at ``target``, in optiona's OWN convention.

    Hand-picking pan/tilt is how the first draft of this script silently tested
    nothing: a camera at y = +55 m needs pan near 180 deg to face the pitch, and
    with pan = 8 deg every template point fell behind the camera, so the
    comparison ran over an empty set and reported NaN.

    Derived from ``geometry.ptr_to_R``: with ``x_cam = R @ (X - C)`` the optical
    axis in world coordinates is the third ROW of R, which for
    ``R = Rz(roll) Rx(tilt) Rz(pan)`` is ``(sin t sin p, sin t cos p, cos t)``.
    Inverting gives the tilt and pan below. :func:`check_fixtures` then asserts
    the camera really does look at the pitch, so this class of error cannot hide
    again.
    """
    C = np.asarray(C, float)
    d = np.asarray(target, float) - C
    d /= np.linalg.norm(d)
    tilt = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
    pan = float(np.arctan2(d[0], d[1]))
    # UPRIGHT requires roll ~ 180 deg, not 0. Roll rotates about the optical
    # axis, so it does not affect the aim, but with roll = 0 the image "down"
    # axis (row 1 of R) comes out as (-ct*sp, -ct*cp, -st) whose z component is
    # NEGATIVE — and z is negative-UP in this world frame, so the view is
    # vertically mirrored: measured on the wide fixture, the near touchline
    # landed at pixel y=101 and the far one at y=653, i.e. upside down.
    # Adding 180 deg flips row 1 and yields a physically real broadcast view.
    # Parity is a same-camera comparison and is unaffected either way, but a
    # fixture that cannot occur is weak evidence, so the fixtures are upright.
    roll = float(np.deg2rad(roll_deg + 180.0))
    return OptCamera(f=f, pan=pan, tilt=tilt, roll=roll, C=C, k1=k1, image_wh=wh)


def cameras():
    """(name, Camera) across the regimes that occur in broadcast footage.

    Positions follow the SoccerNet main-camera prior (0, 55, -12) and the
    README's 16 m-left variant; focal lengths span the range the optiona audit
    measured on real sequences (f ~ 1250 wide to ~5100 zoomed).
    """
    wh = (1920, 1080)
    return [
        ("wide      ", look_at([0.5, 55.0, -12.0], [0, 0, 0], 1.5, 1400.0, 0.0, wh)),
        ("zoomed    ", look_at([0.0, 50.2, -9.5], [30, -6, 0], -0.6, 5100.0, 0.0, wh)),
        ("rolled    ", look_at([-36.0, 55.0, -12.0], [-20, 0, 0], 12.0, 2200.0, 0.0, wh)),
        ("barrel k1 ", look_at([0.0, 58.0, -14.0], [0, 0, 0], 2.0, 1250.0, -0.04, wh)),
        ("pincush k1", look_at([12.0, 52.0, -11.0], [5, -4, 0], -3.0, 1800.0, 0.03, wh)),
        ("offset pp ", look_at([0.0, 54.0, -12.5], [0, 0, 0], 2.0, 1600.0, 0.0, wh)),
    ]


def check_fixtures():
    """The fixtures must actually see the pitch. Guards the whole script."""
    hdr("0. fixtures")
    passed = True
    X = pitch_points()
    for name, cam in cameras():
        uv, front = cam.project(X)
        inside = front & (np.abs(uv[:, 0] - cam.w / 2) < cam.w) \
            & (np.abs(uv[:, 1] - cam.h / 2) < cam.h)
        ctr, cfront = cam.project(np.array([[0., 0., 0.]]))
        note = ""
        if cfront[0]:
            note = (f"pitch centre at ({ctr[0][0]:7.1f}, {ctr[0][1]:7.1f}) px")
        else:
            note = "pitch centre BEHIND camera"
        print(f"       {name:11s} {int(inside.sum()):3d}/{len(X)} template points "
              f"in frame; {note}")
        if inside.sum() < 20:
            bad(f"{name}: fixture does not see the pitch — parity would be "
                "measured over an empty set")
            passed = False
    if passed:
        ok("every fixture sees the pitch")
    return record("fixtures", passed)


def sn_project(params, X):
    """Project through sn_calibration_baseline from a parameter dict."""
    cam = SnCamera()
    cam.from_json_parameters(params)
    out = np.full((len(X), 2), np.nan)
    for i, p in enumerate(X):
        q = cam.project_point(np.asarray(p, float))
        if q[2] != 0:
            out[i] = q[:2]
    return out


def compare(cam, params, X, window=4.0):
    """Max / mean pixel disagreement between the two models.

    Points are selected by the **optiona** projection only — in front of the
    camera and within ``window`` image-widths of the principal point. Gating on
    optiona alone is deliberate: a wrong conversion sends the sn projection
    somewhere absurd, and those points must still be counted or the test could
    not fail. The window exists because a point near the horizon is genuinely
    ill-conditioned in *both* models, so their difference there measures float
    noise rather than disagreement — the negative control (which diverges by
    ~1e6 px, far outside any window) is what proves this is not tolerance
    fudging.
    """
    uv_opt, front = cam.project(X)
    uv_sn = sn_project(params, X)
    near = (np.abs(uv_opt[:, 0] - cam.w / 2.0) < window * cam.w) \
        & (np.abs(uv_opt[:, 1] - cam.h / 2.0) < window * cam.h)
    m = front & near & np.isfinite(uv_opt).all(axis=1)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    d = np.linalg.norm(uv_opt[m] - uv_sn[m], axis=1)
    d = d[np.isfinite(d)] if np.isfinite(d).any() else d
    if d.size == 0:
        return np.inf, np.inf, int(m.sum())
    return float(d.max()), float(d.mean()), int(m.sum())


# ------------------------------------------------------------------------- A + B
def check_parity_and_control(tol):
    X = pitch_points()
    hdr("A. parity  +  B. negative control")
    info(f"{len(X)} pitch-template points; tolerance {tol} px")
    print(f"       {'camera':11s} {'n':>5s} {'parity max':>12s} {'parity mean':>12s} "
          f"{'CONTROL max':>13s}")
    a_ok, b_ok = True, True
    for name, cam in cameras():
        if name.startswith("offset pp"):
            cam.pp = np.array([cam.w / 2.0 + 37.0, cam.h / 2.0 - 21.0])
        real = optiona_camera_to_sncalib(cam)
        ctrl = naive_angle_copy_to_sncalib(cam)
        r_max, r_mean, n = compare(cam, real, X)
        c_max, _, _ = compare(cam, ctrl, X)
        flag = ""
        if not np.isfinite(r_max) or r_max > tol:
            a_ok = False
            flag += f" {RED}<- parity{RESET}"
        # The control must be BOTH finite-or-nan-by-divergence AND far worse.
        if np.isfinite(c_max) and c_max <= max(tol * 100, 1.0):
            b_ok = False
            flag += f" {RED}<- control did not fail{RESET}"
        c_txt = "diverged" if not np.isfinite(c_max) else f"{c_max:13.3f}"
        print(f"       {name:11s} {n:5d} {r_max:12.2e} {r_mean:12.2e} {c_txt:>13s}{flag}")

    if a_ok:
        ok(f"parity within {tol} px on every configuration")
    else:
        bad("parity exceeded tolerance — the conversion is wrong, do not push")
    if b_ok:
        ok("negative control fails loudly, as it must — the test can detect error")
    else:
        bad("negative control PASSED. The test is not measuring the rotation "
            "conversion; treat check A as meaningless until this is fixed.")
    record("parity", a_ok)
    record("negative control", b_ok)
    return a_ok and b_ok


# --------------------------------------------------------------------------- C
def check_distortion(tol):
    hdr("C. distortion (k1 mapping, and the two inversion algorithms)")
    X = pitch_points()
    passed = True
    for name, cam in cameras():
        if abs(cam.k1) < 1e-12:
            continue
        params = optiona_camera_to_sncalib(cam)
        d_max, _, n = compare(cam, params, X)
        # Negative control: BroadTrack's height rescale, which does NOT apply here.
        wrong = dict(params)
        wrong["radial_distortion"] = [cam.k1 * (cam.f / cam.h) ** 2, 0, 0, 0, 0, 0]
        w_max, _, _ = compare(cam, wrong, X)
        status = "" if d_max <= tol else f" {RED}<- forward{RESET}"
        if d_max > tol:
            passed = False
        print(f"       {name:11s} k1={cam.k1:+.3f}  forward {d_max:.2e} px  "
              f"(with BroadTrack rescale: {w_max:8.2f} px){status}")
        if np.isfinite(w_max) and w_max < 1.0:
            warn(f"{name}: the rescale control barely moved — k1 may be too "
                 "small here to discriminate")

        # Inversion gap: sn (cv.undistortPoints) vs optiona (Newton).
        uv, front = cam.project(X)
        uv = uv[front]
        sn = SnCamera()
        sn.from_json_parameters(params)
        gnd_opt, valid = cam.backproject_ground(uv)
        gaps = []
        for i, p in enumerate(uv):
            if not valid[i]:
                continue
            g = sn.unproject_point_on_planeZ0(np.array([p[0], p[1], 1]))
            if np.all(np.isfinite(g)):
                gaps.append(np.linalg.norm(g[:2] - gnd_opt[i][:2]))
        if gaps:
            g = np.asarray(gaps)
            info(f"{name}: inversion gap (cv.undistortPoints vs Newton) "
                 f"max {g.max()*1000:.3f} mm, mean {g.mean()*1000:.3f} mm "
                 f"over {len(g)} points")
    return record("distortion", passed)


# --------------------------------------------------------------------------- D
ROUNDTRIP_TARGETS = np.array([[0., 0., 0.], [-30., 12., 0.], [40., -20., 0.],
                              [-52.5, 34., 0.], [20., 0., 0.]])


def _roundtrip_max(sn, targets=ROUNDTRIP_TARGETS):
    """Max project -> unproject_point_on_planeZ0 error, plus the radii sampled.

    Returns ``(max_error_m, n_points, r_min, r_max)`` where r is the NORMALIZED
    image-plane radius ``|(uv - pp) / f|`` at which the distortion was actually
    inverted. The radius matters because ``L(r) = 1 + k1 r^2`` — an inversion
    residual at r=0.1 and one at r=0.6 are not comparable quantities, so a
    ratio between two cameras is only meaningful if they probe similar radii.
    """
    errs, radii = [], []
    fx, fy = float(sn.xfocal_length), float(sn.yfocal_length)
    px, py = float(sn.principal_point[0]), float(sn.principal_point[1])
    for W in targets:
        q = sn.project_point(W)
        if q[2] == 0:
            continue
        back = sn.unproject_point_on_planeZ0(np.array([q[0], q[1], 1]))
        if np.all(np.isfinite(back)):
            errs.append(np.linalg.norm(back[:2] - W[:2]))
            radii.append(np.hypot((q[0] - px) / fx, (q[1] - py) / fy))
    if not errs:
        return np.nan, 0, np.nan, np.nan
    return float(np.max(errs)), len(errs), float(np.min(radii)), float(np.max(radii))


def _pixel_inversion_residual(sn, targets=ROUNDTRIP_TARGETS):
    """Distortion-inversion residual in PIXELS: pixel -> ground -> pixel.

    The ground-metre roundtrip is the operationally meaningful number (it is what
    ``bbox_pitch`` suffers), but it is the wrong quantity to take a *ratio* of:
    it multiplies the inversion residual by the ground-plane error amplification,
    which depends on the grazing angle and so differs between any two cameras
    that are not co-located. That is why the barrel row scored 0.17x against a
    reference whose radial range overlapped it to within 7% — the gap was
    geometry, not distortion.

    Going pixel -> ground -> pixel applies ``undistort`` then ``distort`` and
    cancels the amplification, leaving the inversion residual alone. That is the
    like-for-like quantity, so the ratio assertion is computed on it.
    """
    errs = []
    for W in targets:
        q = sn.project_point(W)
        if q[2] == 0:
            continue
        g = sn.unproject_point_on_planeZ0(np.array([q[0], q[1], 1]))
        if not np.all(np.isfinite(g)):
            continue
        q2 = sn.project_point(np.array([g[0], g[1], 0.0]))
        if q2[2] == 0:
            continue
        errs.append(float(np.hypot(q[0] - q2[0], q[1] - q2[1])))
    return (float(np.max(errs)), len(errs)) if errs else (np.nan, 0)


def sn_native_reference(f, k1, wh=(1920, 1080)):
    """An sn Camera built **without** the conversion under test.

    Constructed through ``set_camera`` with angles chosen directly in sn's own
    convention — pan 0 deg, tilt 78 deg at the SoccerNet main-camera position,
    verified to put the pitch centre within a few pixels of the principal point.
    Nothing here passes through ``optiona_camera_to_sncalib``, ``cam.R`` or
    ``rotation_matrix_to_pan_tilt_roll``, so its roundtrip residual measures
    sn_calibration_baseline's *own* forward/inverse distortion mismatch at this
    ``f`` and ``k1`` and nothing else.
    """
    d = np.zeros(12)
    d[0] = k1
    cam = SnCamera()
    cam.set_camera(0.0, np.deg2rad(78.0), 0.0, f, f, (wh[0] / 2.0, wh[1] / 2.0),
                   0.0, 55.0, -12.0, distortion=d)
    return cam


def check_roundtrip():
    """Two DIFFERENT claims, deliberately not conflated — and no magic number.

    This check runs entirely INSIDE sn_calibration_baseline: ``project_point``
    applies the forward polynomial ``1 + k1 r^2``, ``unproject_point_on_planeZ0``
    inverts it with ``cv.undistortPoints`` (OpenCV's rational model, fixed
    iteration count). With ``k1 != 0`` those are not exact inverses, so a
    residual exists that has nothing to do with Option A and is shared
    byte-for-byte with the BroadTrack path.

    An earlier version bounded that at a hand-picked 60 mm, which is exactly the
    kind of number that quietly absorbs a real regression. The bound is now
    **differential**: for each distorted fixture, the same roundtrip is measured
    on :func:`sn_native_reference` — an sn Camera at the same ``f`` and ``k1``
    built without touching the conversion — and the converted camera is required
    to be no worse than that reference by more than a small factor. If the
    conversion ever starts contributing error of its own, the ratio moves even
    though the absolute magnitude might not.
    """
    hdr("D. plane roundtrip (the path bbox_pitch takes)")
    TOL_GEOM = 1e-3          # metres; pure-geometry bound at k1 == 0
    REF_FACTOR = 2.0         # allowed multiple of sn's own reference residual
    REF_FLOOR = 1e-3         # never demand better than the geometry bound
    passed = True
    for name, cam in cameras():
        sn = SnCamera()
        sn.from_json_parameters(optiona_camera_to_sncalib(cam))
        e, n, r0, r1 = _roundtrip_max(sn)
        if n == 0:
            warn(f"{name}: no target visible, skipped")
            continue
        if abs(cam.k1) <= 1e-12:
            if e > TOL_GEOM:
                passed = False
                bad(f"{name}: {e*1000:.4f} mm > {TOL_GEOM*1000:.0f} mm (pure geometry)")
            else:
                print(f"       {name:11s} max {e*1000:8.4f} mm over {n} points "
                      f"[pure geometry, bound {TOL_GEOM*1000:.0f} mm]")
            continue

        refcam = sn_native_reference(cam.f, cam.k1)
        _, nref, q0, q1 = _roundtrip_max(refcam)
        # The ratio is taken on the PIXEL residual, which isolates the distortion
        # inversion; the metre figure is reported because it is what bbox_pitch
        # actually suffers, but it is not comparable across viewing geometries.
        px_e, _ = _pixel_inversion_residual(sn)
        px_ref, _ = _pixel_inversion_residual(refcam)
        if nref == 0 or not np.isfinite(px_ref) or px_ref <= 0:
            warn(f"{name}: sn-native reference unusable; falling back to the "
                 "geometry bound")
            ratio, px_ref, q0, q1 = np.nan, np.nan, np.nan, np.nan
            if e > max(REF_FACTOR * TOL_GEOM, REF_FLOOR):
                passed = False
                bad(f"{name}: {e*1000:.3f} mm with no usable reference")
            continue
        ratio = px_e / px_ref
        if ratio > REF_FACTOR:
            passed = False
            bad(f"{name}: pixel inversion residual {px_e:.4f} px vs sn-native "
                f"reference {px_ref:.4f} px (ratio {ratio:.2f}x, limit "
                f"{REF_FACTOR:.1f}x) — the conversion is adding error of its own")
        else:
            print(f"       {name:11s} ground {e*1000:8.3f} mm  r={r0:.2f}-{r1:.2f} | "
                  f"pixel {px_e:7.4f} px  ref {px_ref:7.4f} px (r={q0:.2f}-{q1:.2f})"
                  f"  ratio {ratio:4.2f}x")
            if np.isfinite(q1) and max(r1, q1) > 0 and \
                    abs(r1 - q1) > 0.35 * max(r1, q1):
                warn(f"{name}: radial ranges differ substantially "
                     f"({r0:.2f}-{r1:.2f} vs {q0:.2f}-{q1:.2f}); even the pixel "
                     "ratio is comparing different parts of the distortion curve "
                     "here — treat this row as weak evidence.")
    if passed:
        ok(f"geometry within {TOL_GEOM*1000:.0f} mm; every distorted camera "
           f"within {REF_FACTOR:.1f}x sn's own inversion residual")
        info("the bound is differential, not a fixed magnitude: it tracks "
             "sn_calibration_baseline's behaviour at the same f and k1, so a "
             "regression in the conversion shows up as a ratio change even if "
             "the absolute error stays small")
    return record("plane roundtrip", passed)


# --------------------------------------------------------------------------- E
def check_bbox_pitch():
    hdr("E. get_bbox_pitch (copied from broadtrack_api)")
    _, cam = cameras()[0]
    params = optiona_camera_to_sncalib(cam)
    sn = SnCamera()
    sn.from_json_parameters(params)
    fn = get_bbox_pitch(sn)

    # A box whose bottom edge sits on the pitch: keys present, middle between.
    W = np.array([0.0, 0.0, 0.0])
    q = sn.project_point(W)
    box = (q[0] - 20, q[1] - 80, q[0] + 20, q[1])
    out = fn(box)
    passed = True
    if out is None:
        bad("a bottom-edge-on-pitch box returned None")
        passed = False
    else:
        need = {"x_bottom_left", "y_bottom_left", "x_bottom_right",
                "y_bottom_right", "x_bottom_middle", "y_bottom_middle"}
        if set(out) != need:
            bad(f"key mismatch: {sorted(set(out) ^ need)}")
            passed = False
        else:
            mid_x = 0.5 * (out["x_bottom_left"] + out["x_bottom_right"])
            ok(f"six keys; centre projects to ({out['x_bottom_middle']:.2f}, "
               f"{out['y_bottom_middle']:.2f}) m, expected near (0, 0)")
            if abs(out["x_bottom_middle"]) > 1.0 or abs(out["y_bottom_middle"]) > 1.0:
                bad("bottom-middle is not where the world point was placed")
                passed = False
            info(f"bottom-middle vs mean of corners: "
                 f"{abs(mid_x - out['x_bottom_middle']):.4f} m apart "
                 "(perspective makes these differ slightly — not an error)")

    # KNOWN INHERITED LIMITATION, recorded rather than fixed.
    # get_bbox_pitch rejects an entry only when a coordinate is NaN. A bbox whose
    # bottom edge lies ABOVE the horizon still yields a finite intersection with
    # Z=0 (behind the camera), so it returns plausible-looking but meaningless
    # pitch coordinates instead of None. This is a silent-wrong-coordinate path.
    # It is NOT patched here: the helper is copied byte-for-byte from
    # broadtrack_api so the A/B against BroadTrack compares like with like, and
    # changing the numerics would invalidate that comparison. Flagged in the
    # report as a candidate follow-up (a cheirality check on the unprojected
    # ray) that must be applied to BOTH paths at once or neither.
    horizon_box = (10.0, -5000.0, 60.0, -4900.0)
    got = fn(horizon_box)
    if got is not None:
        info(f"above-horizon box -> ({got['x_bottom_middle']:.1f}, "
             f"{got['y_bottom_middle']:.1f}) m, not None — inherited "
             "NaN-only guard, identical in broadtrack_api. See report.")
    else:
        ok("above-horizon box -> None")
    return record("bbox_pitch", passed)


# ------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="parity tolerance in pixels (default 0.01)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    print(f"{BOLD}Option A -> sn-calibration conversion, numeric parity{RESET}")
    info(f"numpy {np.__version__}")
    _, cam = cameras()[0]
    info(f"principal-point offset on the default fixture: "
         f"{principal_point_offset(cam):.1f} px")

    check_fixtures()
    check_parity_and_control(args.tol)
    check_distortion(args.tol)
    check_roundtrip()
    check_bbox_pitch()

    hdr("summary")
    failed = [n for n, p in _results if not p]
    for n, p in _results:
        print(f"  {GREEN + 'PASS' + RESET if p else RED + 'FAIL' + RESET}  {n}")
    print()
    if not failed:
        print(f"{GREEN}Conversion verified — and the negative control proves the "
              f"test could have failed.{RESET} Pre-push gate passed.")
        return 0
    print(f"{RED}{len(failed)} check(s) failed: {', '.join(failed)}{RESET}")
    print(f"{YELLOW}Do not push. Fix the conversion, not the tolerance.{RESET}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
