"""Engine-contract test for OptionACalibration, runnable without tracklab.

Every post-push fix costs a new commit and a fresh Kaggle session, so the parts
of ``process()`` that can be checked locally are checked locally. Almost all of
it is pure pandas — reindexing onto the detections index, ``.where(notna(),
None)``, filename-sorted iteration, in-place assignment to ``metadatas`` — and
none of that needs torch, PnLCalib, a checkpoint or a GPU. The single tracklab
dependency inside the hot path is ``detections.bbox.ltrb()``, which is stubbed
here with an equivalent pandas accessor.

What this does NOT cover: ``compute_cameras`` (the torch boundary) is bypassed
entirely by pre-seeding the per-sequence JSON cache. Whether PnLCalib runs under
torch 1.13.1 is answered by ``scripts/verify_pnlcalib_env.py`` on Kaggle.

    python tests/test_optiona_api_contract.py      # or: pytest tests/
"""
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plugins" / "calibration"))
for _c in (ROOT / "optiona_sfr", ROOT.parent / "optiona_sfr"):
    if (_c / "optiona_sfr" / "geometry.py").is_file():
        sys.path.insert(0, str(_c))
        break

# --- stub tracklab: only the base class is touched at import time ------------
_tl = types.ModuleType("tracklab")
_pl = types.ModuleType("tracklab.pipeline")
_vm = types.ModuleType("tracklab.pipeline.videolevel_module")


class VideoLevelModule:  # noqa: D101
    pass


_vm.VideoLevelModule = VideoLevelModule
_pl.videolevel_module = _vm
sys.modules.setdefault("tracklab", _tl)
sys.modules.setdefault("tracklab.pipeline", _pl)
sys.modules.setdefault("tracklab.pipeline.videolevel_module", _vm)


# --- stub the one tracklab accessor process() calls -------------------------
@pd.api.extensions.register_dataframe_accessor("bbox")
class _BboxAccessor:
    """Equivalent of tracklab's ``.bbox.ltrb()`` for ltwh-stored boxes."""

    def __init__(self, df):
        self._df = df

    def ltrb(self):
        return self._df["bbox_ltwh"].apply(
            lambda v: np.array([v[0], v[1], v[0] + v[2], v[1] + v[3]], float))


from optiona_sfr.geometry import Camera as OptCamera  # noqa: E402
from sn_gamestate.calibration.optiona_api import OptionACalibration  # noqa: E402
from sn_gamestate.calibration.optiona_convert import (  # noqa: E402
    optiona_camera_to_sncalib,
)

N_FRAMES = 6
SEQ = "SNGS-TEST"


def _camera():
    """An UPRIGHT main-camera view that sees the pitch.

    Two non-obvious bits, both measured rather than assumed: a camera at
    y = +55 m needs pan ~180 deg to face the pitch, and roll ~180 deg to be the
    right way up — with roll = 0 this ZXZ convention mirrors the image
    vertically (near touchline at pixel y=101, far at y=653). Upright, the
    ordering is near 979 / centre 540 / far 427, so a bbox with its bottom edge
    around y=700-900 stands on the pitch.
    """
    C = np.array([0.5, 55.0, -12.0])
    d = -C / np.linalg.norm(C)
    return OptCamera(f=1400.0, pan=float(np.arctan2(d[0], d[1])),
                     tilt=float(np.arccos(np.clip(d[2], -1, 1))),
                     roll=float(np.pi + 0.02), C=C, k1=0.0, image_wh=(1920, 1080))


def _fixture(tmp, calibrated_frames, shuffle=True):
    """Build (module, detections, metadatas, frame_names).

    ``calibrated_frames`` is the set of 1-based frame numbers present in the
    cache; the rest exercise carry-forward and the None path.
    """
    calib_dir = Path(tmp) / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{i:06d}.jpg" for i in range(1, N_FRAMES + 1)]
    params = optiona_camera_to_sncalib(_camera())
    payload = {"sequence": SEQ, "image_wh": [1920, 1080], "info": {},
               "frames": {names[i - 1]: {"parameters": params, "s": 0.9}
                          for i in calibrated_frames}}
    (calib_dir / f"{SEQ}.json").write_text(json.dumps(payload))

    cfg = types.SimpleNamespace(calib_dir=str(calib_dir), use_cached_json=True,
                                use_prev_parameters=True, min_score=0.0)
    mod = OptionACalibration(cfg, device="cpu")

    image_ids = list(range(100, 100 + N_FRAMES))
    metadatas = pd.DataFrame(
        {"file_path": [str(Path(tmp) / SEQ / "img1" / n) for n in names]},
        index=image_ids)

    rows = []
    for k, iid in enumerate(image_ids):
        for j in range(2):
            # Bottom edge lands at y=790..880 px: between the image centre (540)
            # and the near touchline (979), i.e. standing on the pitch.
            rows.append({"bbox_ltwh": [800.0 + 30 * j, 700.0 + 15 * k, 40.0, 90.0],
                         "image_id": iid})
    detections = pd.DataFrame(rows, index=[10 * i for i in range(len(rows))])
    if shuffle:                       # incoming row order must not matter
        detections = detections.sample(frac=1.0, random_state=0)
        metadatas = metadatas.sample(frac=1.0, random_state=1)
    return mod, detections, metadatas, names


def test_returns_dataframe_only():
    """Returning a tuple corrupts OfflineTrackingEngine.video_loop."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        out = mod.process(det, meta)
    assert isinstance(out, pd.DataFrame), type(out)
    assert not isinstance(out, tuple)


def test_output_columns_and_index_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        out = mod.process(det, meta)
    assert "bbox_pitch" in out.columns
    assert list(out.index) == list(det.index), "index must be preserved"
    assert len(out) == len(det)
    keys = {"x_bottom_left", "y_bottom_left", "x_bottom_right",
            "y_bottom_right", "x_bottom_middle", "y_bottom_middle"}
    vals = [v for v in out["bbox_pitch"] if v is not None]
    assert vals, "no detection was projected"
    assert all(set(v) == keys for v in vals)


def test_healthy_projection_lands_on_the_pitch():
    """The out-of-bounds diagnostic must read 0 on a good camera.

    Guards against the counter being vacuously quiet AND against the fixture
    silently drifting off-pitch: bboxes standing between the image centre and
    the near touchline have to unproject inside the field.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        out = mod.process(det, meta)
    vals = [v for v in out["bbox_pitch"] if isinstance(v, dict)]
    assert vals, "nothing projected"
    xs = np.array([v["x_bottom_middle"] for v in vals])
    ys = np.array([v["y_bottom_middle"] for v in vals])
    assert np.all(np.abs(xs) <= 60.0) and np.all(np.abs(ys) <= 45.0), (
        f"projected off-pitch: x {xs.min():.1f}..{xs.max():.1f}, "
        f"y {ys.min():.1f}..{ys.max():.1f}")


def test_parameters_written_in_place_on_metadatas():
    """TrackerState persists the metadatas object itself, not a copy."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        before = id(meta)
        mod.process(det, meta)
    assert id(meta) == before
    assert "parameters" in meta.columns
    assert all(isinstance(v, dict) for v in meta["parameters"])
    assert all(v for v in meta["parameters"]), "all frames were calibrated"


def test_uncalibrated_frames_get_empty_dict_not_nan():
    """With carry-forward off, a frame with no camera must yield {} and None."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, names = _fixture(tmp, [3, 4, 5, 6])
        mod.use_prev_parameters = False
        out = mod.process(det, meta)
    order = meta["file_path"].astype(str).map(lambda p: Path(p).name)
    missing = [i for i in meta.index if order[i] in names[:2]]
    assert all(meta.loc[i, "parameters"] == {} for i in missing)
    ids = set(missing)
    sub = out.loc[det["image_id"].isin(ids).reindex(out.index).fillna(False)]
    assert sub["bbox_pitch"].isna().all() or (sub["bbox_pitch"] == None).all()  # noqa: E711
    # and NaN must never survive into the column
    assert not any(isinstance(v, float) and np.isnan(v) for v in out["bbox_pitch"])


def test_carry_forward_follows_filename_order_not_row_order():
    """Rows arrive shuffled; carry-forward must still run chronologically.

    Frames 1-2 are uncalibrated and 3-6 are. In filename order nothing precedes
    frame 1, so it stays empty, while 3-6 are filled directly. If the loop
    followed the shuffled row order instead, an early row could inherit a
    *later* frame's camera — silently wrong, and invisible in completeness.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, names = _fixture(tmp, [3, 4, 5, 6])
        mod.process(det, meta)
    order = meta["file_path"].astype(str).map(lambda p: Path(p).name)
    by_name = {order[i]: meta.loc[i, "parameters"] for i in meta.index}
    assert by_name[names[0]] == {}, "frame 1 has no predecessor to inherit from"
    assert by_name[names[1]] == {}, "frame 2 has no predecessor to inherit from"
    for n in names[2:]:
        assert by_name[n], f"{n} should be calibrated"


def test_carry_forward_fills_a_later_gap():
    """A gap AFTER a good frame is filled from the previous camera."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, names = _fixture(tmp, [1, 2, 3])
        mod.process(det, meta)
    order = meta["file_path"].astype(str).map(lambda p: Path(p).name)
    by_name = {order[i]: meta.loc[i, "parameters"] for i in meta.index}
    for n in names:
        assert by_name[n], f"{n} should be carried forward"


def test_min_score_gates_on_s():
    """s below min_score must be rejected even though a camera exists."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        mod.min_score = 0.95           # every cached s is 0.9
        mod.use_prev_parameters = False
        mod.process(det, meta)
    assert all(v == {} for v in meta["parameters"]), \
        "min_score did not gate on s"


def test_empty_metadatas_is_a_noop():
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, _, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        out = mod.process(det, pd.DataFrame(columns=["file_path"]))
    assert isinstance(out, pd.DataFrame)


def test_empty_outputs_shape():
    """Every early return goes through this; it must satisfy the contract."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        out = OptionACalibration._empty_outputs(det, meta)
    assert isinstance(out, pd.DataFrame)
    assert "bbox_pitch" in out.columns and out["bbox_pitch"].isna().all()
    assert list(meta["parameters"]) == [{}] * len(meta)


def test_missing_cache_falls_back_cleanly():
    """An unreadable cache must not raise out of process()."""
    with tempfile.TemporaryDirectory() as tmp:
        mod, det, meta, _ = _fixture(tmp, range(1, N_FRAMES + 1))
        (Path(tmp) / "calib" / f"{SEQ}.json").write_text("{ not json")
        # compute_cameras will fail (no PnLCalib here) -> _empty_outputs
        out = mod.process(det, meta)
    assert isinstance(out, pd.DataFrame)
    assert "bbox_pitch" in out.columns
    assert all(isinstance(v, dict) for v in meta["parameters"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(failed)
