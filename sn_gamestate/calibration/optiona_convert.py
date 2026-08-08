"""Pure-geometry half of the Option A calibration stage.

Deliberately separated from ``optiona_api``: this module imports **numpy only**
(plus ``sn_calibration_baseline``, which needs numpy + cv2). No tracklab, no
torch, no PnLCalib. That is what makes ``scripts/verify_optiona_conversion.py``
runnable as a local pre-push gate on a bare Python 3.9 with ~95 MiB of wheels,
instead of requiring the full Kaggle round trip.

Two things live here, and they are the two things that can silently produce
wrong pitch coordinates without raising:

* :func:`optiona_camera_to_sncalib` — the schema conversion, derived in the
  Task 1 audit and verified numerically by the gate.
* :func:`get_bbox_pitch` — originally copied verbatim from ``broadtrack_api.py``
  so the ``bbox_pitch`` numerics were byte-compatible with the BroadTrack arm.
  Both that module and ``sn_gamestate.calibration.bbox2pitch`` are gone (see
  KNOWN_LIMITATIONS.md section 7); this is now the only definition.
"""
import numpy as np

from sn_calibration_baseline.camera import rotation_matrix_to_pan_tilt_roll


def optiona_camera_to_sncalib(cam) -> dict:
    """``optiona_sfr.geometry.Camera`` -> sn-calibration parameter dict.

    Consumable as-is by ``sn_calibration_baseline.Camera.from_json_parameters``.

    Rationale, all established against the sources in the Task 1 audit — none of
    it is guessed, and the numeric check lives in
    ``scripts/verify_optiona_conversion.py``:

    * **Rotation.** The two ZXZ conventions are *reversed*:
      ``sn`` builds ``transpose(Rz(pan) @ Rx(tilt) @ Rz(roll))`` while
      ``optiona.geometry.ptr_to_R`` builds ``Rz(roll) @ Rx(tilt) @ Rz(pan)``.
      Copying ``cam.pan/tilt/roll`` across is therefore **wrong** — that is the
      gate's negative control. But both models use ``x_cam = R @ (X - C)``, so
      ``cam.R`` *is* sn's ``self.rotation``, and the safe conversion is to
      round-trip the matrix through sn's own decomposition. This is exactly what
      PnLCalib does at ``utils/utils_calib.py:450-454`` to emit ``cam_params``,
      and those ``cam_params`` are scored directly by
      ``sn_calibration/src/evalai_camera.py`` — so the frame is provably shared.

    * **Distortion.** ``optiona`` applies ``L(r) = 1 + k1 r^2`` on the normalized
      image plane (``geometry.py:139-142``); ``sn.distort`` applies
      ``(1 + sum k_i r^(2(i+1))) / (1 + sum k_(i+3) r^(2(i+1)))`` on the *same*
      normalized plane. So ``k1`` maps into slot 0 **unscaled**. BroadTrack's
      ``(f/H)^(2(i+1))`` rescale was an artefact of its height-normalized JSON
      export and has no analogue here.

    * **Principal point.** ``list(cam.pp)``, *not* the image centre.
      ``Camera.from_vec`` resets ``pp`` to the centre, but ``from_pnlcalib``
      honours PnLCalib's own ``principal_point`` and the "do no harm" path in
      ``refine_camera`` can return that seed untouched. Note the consequence:
      ``from_json_parameters`` derives ``image_width = 2 * pp[0]``, so an
      off-centre ``pp`` silently changes the recorded image size — see
      :func:`principal_point_offset`.

    * **Focal length.** ``optiona`` carries a single ``f``; PnLCalib fixes the
      aspect ratio upstream (``CALIB_FIX_ASPECT_RATIO``), so ``fx == fy == f``.
    """
    pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(np.asarray(cam.R, float))
    pp = np.asarray(cam.pp, float).ravel()[:2]
    return {
        "principal_point": [float(pp[0]), float(pp[1])],
        "x_focal_length": float(cam.f),
        "y_focal_length": float(cam.f),
        "pan_degrees": float(np.rad2deg(pan)),
        "tilt_degrees": float(np.rad2deg(tilt)),
        "roll_degrees": float(np.rad2deg(roll)),
        "position_meters": [float(v) for v in np.asarray(cam.C, float).ravel()[:3]],
        # Only k1 is modelled; slots 1..5 (k2, k3 and the denominator k4..k6)
        # stay zero. No rescale — see the docstring.
        "radial_distortion": [float(cam.k1), 0.0, 0.0, 0.0, 0.0, 0.0],
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
    }


def naive_angle_copy_to_sncalib(cam) -> dict:
    """**Deliberately wrong.** The negative control for the parity gate.

    Identical to :func:`optiona_camera_to_sncalib` except that
    ``cam.pan/tilt/roll`` are copied straight across instead of being recovered
    from ``cam.R``. Because the two ZXZ orders are reversed this must produce a
    large projection error. A parity test whose control passes is not testing
    anything, so the gate reports both numbers side by side.
    """
    d = optiona_camera_to_sncalib(cam)
    d["pan_degrees"] = float(np.rad2deg(cam.pan))
    d["tilt_degrees"] = float(np.rad2deg(cam.tilt))
    d["roll_degrees"] = float(np.rad2deg(cam.roll))
    return d


def principal_point_offset(cam) -> float:
    """Pixel distance from ``cam.pp`` to the true image centre.

    Non-zero is legitimate (PnLCalib may hand back its own principal point) but
    it means ``from_json_parameters`` will record an image size of
    ``2 * pp``, not ``(cam.w, cam.h)``. The module logs this above a threshold
    rather than silently normalising it away.
    """
    pp = np.asarray(cam.pp, float).ravel()[:2]
    return float(np.hypot(pp[0] - cam.w / 2.0, pp[1] - cam.h / 2.0))


def get_bbox_pitch(cam):
    """Bottom-left / bottom-right / bottom-middle unprojection onto the Z=0 plane.

    Originally copied verbatim from ``broadtrack_api.get_bbox_pitch`` (removed; see
    KNOWN_LIMITATIONS.md section 7) so the schema and numerics matched the
    BroadTrack arm exactly while the A/B was live. ``unproject_point_on_planeZ0`` undistorts by
    default, so ``k1`` is honoured.
    """
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l + (r - l) / 2, b, 1])
        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox
