"""Final-check suite: precision and robustness beyond the core-math tests.

Covers: (1) gross-outlier robustness of the Cauchy-weighted refinement,
(2) k1 recovery by the temporal layer, (3) synthetic end-to-end tracking of a
panning-zooming camera over 60 frames incl. Jaccard confidence and forced
reinitialization, (4) precision of the reprojection-error metric.
Runs fully offline (no GPU, no dataset).
"""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from optiona_sfr.geometry import Camera, LINES_WORLD, CIRCLES_WORLD, ptr_to_R
from optiona_sfr.refine import refine_camera
from optiona_sfr.config import RefineCfg, TrackerCfg
from optiona_sfr.tracker import BroadTrackLayer, jaccard_confidence
from optiona_sfr.experiments import reprojection_error, _world_geometry_by_name

rng = np.random.default_rng(7)
WH = (1920, 1080)


def gt_cam(t=0.0, k1=0.0):
    """Panning + zooming main-camera trajectory."""
    return Camera(f=3000.0 + 600.0 * np.sin(0.05 * t),
                  pan=np.deg2rad(6.0) + 0.004 * t,
                  tilt=np.deg2rad(75.0) + 0.0004 * np.sin(0.1 * t),
                  roll=np.deg2rad(0.4),
                  C=[0.0, -55.0, -14.0], k1=k1, image_wh=WH)


def observations(cam, kp_noise=1.2, n_kp=16, outlier_frac=0.0):
    ii = rng.choice(len(LINES_WORLD), n_kp)
    tt = rng.uniform(0, 1, n_kp)
    X = np.array([LINES_WORLD[i][0] * (1 - t) + LINES_WORLD[i][1] * t
                  for i, t in zip(ii, tt)])
    uv, front = cam.project(X)
    uv = uv + rng.normal(0, kp_noise, uv.shape)
    n_out = int(outlier_frac * front.sum())
    if n_out:
        ids = rng.choice(np.where(front)[0], n_out, replace=False)
        uv[ids] += rng.uniform(-250, 250, (n_out, 2))     # gross outliers
    kp_obs = (np.arange(n_kp)[front], uv[front], np.ones(front.sum()), X[front])
    line_obs = []
    for li in [13, 14, 17, 19]:
        A, B = LINES_WORLD[li - 1]
        prj, fr = cam.project(np.stack([A, B]))
        if fr.all():
            line_obs.append((li, prj + rng.normal(0, kp_noise, prj.shape), 0.9))
    circle_obs = []
    cx, cy, r = CIRCLES_WORLD[0]
    a = rng.uniform(0, 2 * np.pi, 20)
    W = np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros_like(a)], 1)
    cuv, fr = cam.project(W)
    circle_obs.append((0, cuv[fr] + rng.normal(0, kp_noise, cuv[fr].shape),
                       np.full(fr.sum(), 0.9)))
    return kp_obs, line_obs, circle_obs


def grid_err(cam, cam_gt):
    G = np.column_stack([rng.uniform(-45, 45, 80), rng.uniform(-30, 30, 80),
                         np.zeros(80)])
    a, fa = cam_gt.project(G)
    b, _ = cam.project(G)
    return float(np.linalg.norm(a[fa] - b[fa], axis=1).mean())


def test_outlier_robustness():
    """20% gross keypoint outliers: Cauchy-weighted refinement must still land
    within a few px of GT (linear loss would be dragged away)."""
    cg = gt_cam(0.0)
    kp, ln, ci = observations(cg, outlier_frac=0.20)
    c0 = Camera(cg.f * 1.05, cg.pan + 0.02, cg.tilt - 0.012, cg.roll,
                cg.C + np.array([1.0, -1.5, 0.6]), image_wh=WH)
    cfg = RefineCfg()
    c_rob, _ = refine_camera(c0, kp, ln, ci, cfg)
    cfg_lin = RefineCfg(robust_loss="linear")
    c_lin, _ = refine_camera(c0, kp, ln, ci, cfg_lin)
    e_rob, e_lin = grid_err(c_rob, cg), grid_err(c_lin, cg)
    print(f"outliers 20%%: cauchy {e_rob:.2f} px vs linear {e_lin:.2f} px")
    assert e_rob < 6.0, e_rob
    assert e_rob <= e_lin + 1e-6


def fake_calibrator(cam_gt_fn, t_holder, noise=0.03):
    """Simulates PnLCalib heuristic_voting: GT + noise, occasionally failing."""
    def cal(_ctx):
        cg = cam_gt_fn(t_holder[0])
        return Camera(cg.f * (1 + rng.normal(0, noise)),
                      cg.pan + rng.normal(0, noise / 2),
                      cg.tilt + rng.normal(0, noise / 3),
                      cg.roll + rng.normal(0, noise / 4),
                      cg.C + rng.normal(0, 0.8, 3), k1=0.0, image_wh=WH)
    return cal


def run_tracking(k1_gt=0.0, optimize_k1=False, n=60, dropout_at=None):
    rcfg = RefineCfg()
    tcfg = TrackerCfg(use_optical_flow=False, use_tripod=False,
                      optimize_k1=optimize_k1)
    layer = BroadTrackLayer(rcfg, tcfg, WH)
    t_holder = [0.0]
    cal = fake_calibrator(lambda t: gt_cam(t, k1_gt), t_holder)
    errs, ss, k1s = [], [], []
    for t in range(n):
        t_holder[0] = float(t)
        cg = gt_cam(t, k1_gt)
        if dropout_at and t in dropout_at:      # frames with almost no cues
            kp, ln, ci = observations(cg, n_kp=3)
            ln, ci = [], []
        else:
            kp, ln, ci = observations(cg)
        cam, s = layer.step(kp, ln, ci, gray=None,
                            single_frame_calibrate=cal, frame_ctx={})
        assert cam is not None
        errs.append(grid_err(cam, cg)); ss.append(s); k1s.append(cam.k1)
    return np.array(errs), np.array(ss), np.array(k1s), layer


def test_tracking_accuracy_and_stability():
    errs, ss, _, layer = run_tracking()
    print(f"tracking: mean {errs.mean():.2f} px, max {errs.max():.2f} px, "
          f"mean s {ss.mean():.2f}, reinits {layer.n_reinit}")
    assert errs.mean() < 4.0, errs.mean()
    assert errs.max() < 12.0, errs.max()
    assert ss.mean() > 0.5


def test_tracking_survives_detection_dropout():
    """Frames 25-27 have almost no cues; warm start must carry the camera
    through and the error must recover afterwards."""
    errs, ss, _, layer = run_tracking(dropout_at={25, 26, 27})
    post = errs[30:].mean()
    print(f"dropout: during {errs[25:28].max():.2f} px, after {post:.2f} px, "
          f"reinits {layer.n_reinit}")
    assert post < 4.0, post
    # During the blackout the min-support gate freezes the warm-started
    # camera, so the residual error equals the GT camera motion accumulated
    # over the frozen frames (~40-70 px for this trajectory) — the physical
    # floor for any cue-free tracker. Without the gate this diverges to
    # >200 px (overfitting 3 noisy points); with optical flow enabled (off in
    # this test) the L_OF term bridges the gap further.
    assert errs[25:28].max() < 90.0, errs[25:28].max()


def test_k1_recovery():
    """GT has barrel distortion k1 = -0.04; tracker with optimize_k1=True must
    (a) beat the k1=0 tracker on reprojection and (b) estimate k1 with the
    right sign and magnitude."""
    e_no, _, _, _ = run_tracking(k1_gt=-0.04, optimize_k1=False)
    e_yes, _, k1s, _ = run_tracking(k1_gt=-0.04, optimize_k1=True)
    k1_est = np.median(k1s[10:])
    print(f"k1: err no-opt {e_no.mean():.2f} px, opt {e_yes.mean():.2f} px, "
          f"k1_est {k1_est:.4f} (gt -0.04)")
    assert e_yes.mean() < e_no.mean()
    assert -0.08 < k1_est < -0.01, k1_est


def test_jaccard_confidence_discriminates():
    cg = gt_cam(0.0)
    kp, ln, _ = observations(cg, n_kp=20)
    s_good = jaccard_confidence(cg, kp[1], ln, WH)
    bad = Camera(cg.f, cg.pan + 0.15, cg.tilt - 0.06, cg.roll, cg.C,
                 image_wh=WH)
    s_bad = jaccard_confidence(bad, kp[1], ln, WH)
    print(f"jaccard: good {s_good:.2f}, bad {s_bad:.2f}")
    assert s_good > s_bad + 0.1
    assert s_bad < 0.5              # a badly wrong camera must trigger reinit


def test_reprojection_metric_precision():
    """The metric must read ~0 for GT and ~=delta for a known px shift."""
    cg = gt_cam(0.0)
    geom = _world_geometry_by_name()
    ann = {}
    for name in ["Middle line", "Side line top", "Circle central"]:
        W = geom[name]
        uv, fr = cg.project(W)
        uv = uv[fr]
        keep = (uv[:, 0] > 0) & (uv[:, 0] < WH[0]) & (uv[:, 1] > 0) & (uv[:, 1] < WH[1])
        ann[name] = uv[keep][::4]
    e0 = reprojection_error(cg, ann)
    ann_s = {k: v + np.array([3.0, 0.0]) for k, v in ann.items()}
    e3 = reprojection_error(cg, ann_s)
    print(f"metric: gt {e0:.3f} px, +3px-shift {e3:.3f} px")
    assert e0 < 0.6, e0             # dense-sampling discretization bound
    assert 1.5 < e3 < 3.3, e3       # shift along the element can be < 3




def test_cached_flow_matches_live_geometry():
    """The cached-flow path must reproduce the live path's geometry exactly:
    same back-projection, same residuals — only the source of correspondences
    differs. This is what makes pruning images after Stage 2 sound."""
    from optiona_sfr.tracker import flow_from_pairs
    from optiona_sfr.config import TrackerCfg as TC
    cam = gt_cam(0.0)
    X = np.column_stack([rng.uniform(-45, 45, 90), rng.uniform(-30, 30, 90),
                         np.zeros(90)])
    uv, fr = cam.project(X)
    p0 = uv[fr].astype("float32")
    cam_t = gt_cam(1.0)
    Xf = X[fr]
    p1, fr2 = cam_t.project(Xf)
    assert fr2.all()                       # all still in front of cam_t
    Xw, uvt = flow_from_pairs(cam, (p0, p1.astype("float32")), TC())
    assert len(Xw) == len(Xf)              # nothing filtered for on-pitch pts
    assert np.abs(Xw[:, :2] - Xf[:, :2]).max() < 1e-3
    # those correspondences must be exactly consistent with cam_t
    prj, _ = cam_t.project(Xw)
    assert np.linalg.norm(prj - uvt, axis=1).max() < 1e-3
    print(f"cached flow: {len(Xw)} correspondences, exact")


def test_tracking_with_cached_flow_runs():
    """End-to-end step() using flow_pair instead of images."""
    from optiona_sfr.config import TrackerCfg as TC
    rcfg = RefineCfg()
    layer = BroadTrackLayer(rcfg, TC(use_optical_flow=True, use_tripod=False),
                            WH)
    t_holder = [0.0]
    cal = fake_calibrator(lambda t: gt_cam(t), t_holder)
    # a FIXED set of ground points, so consecutive projections are genuine
    # correspondences (as real optical flow would produce)
    Xfix = np.column_stack([rng.uniform(-45, 45, 60),
                            rng.uniform(-30, 30, 60), np.zeros(60)])
    errs = []
    prev_uv = None
    for t in range(25):
        t_holder[0] = float(t)
        cg = gt_cam(t)
        kp, ln, ci = observations(cg)
        uv, fr = cg.project(Xfix)
        pair = None
        if prev_uv is not None:
            m = fr & prev_fr
            if m.sum() >= 12:
                pair = (prev_uv[m].astype("float32"), uv[m].astype("float32"))
        cam, s = layer.step(kp, ln, ci, flow_pair=pair,
                            single_frame_calibrate=cal, frame_ctx={})
        prev_uv, prev_fr = uv, fr
        errs.append(grid_err(cam, cg))
    errs = np.array(errs)
    print(f"cached-flow tracking: mean {errs.mean():.2f} px, "
          f"max {errs.max():.2f} px")
    assert errs.mean() < 5.0, errs.mean()




def test_confidence_robust_to_degenerate_geometry():
    """Real-data crash regression (Kaggle, cv2 4.13): jaccard_confidence must
    never raise, whatever the camera or detections produce — NaN/inf
    projections from a degenerate camera, absurd magnitudes, empty inputs,
    or odd numpy scalar types."""
    from optiona_sfr.tracker import jaccard_confidence
    cam = gt_cam(0.0)
    # NaN & inf keypoints and lines
    kp_uv = np.array([[np.nan, 10.0], [np.inf, -np.inf], [100.0, 100.0]])
    ln = [(13, np.array([[np.nan, 5.0], [1e12, -3e9]]), 0.9),
          (14, np.array([[50.0, 50.0], [500.0, 400.0]]), 0.9)]
    s = jaccard_confidence(cam, kp_uv, ln, WH)
    assert 0.0 <= s <= 1.0
    # degenerate camera: near-zero focal, position at the pitch plane
    bad = Camera(1e-6, 0.0, 1e-9, 0.0, [0.0, 0.0, -1e-9], image_wh=WH)
    s2 = jaccard_confidence(bad, kp_uv, ln, WH)
    assert 0.0 <= s2 <= 1.0
    # float32 inputs (as cached pickles round-trip them)
    s3 = jaccard_confidence(cam, kp_uv.astype(np.float32),
                            [(13, ln[1][1].astype(np.float32), 0.9)], WH)
    assert 0.0 <= s3 <= 1.0
    print(f"degenerate-geometry confidence: {s:.2f}/{s2:.2f}/{s3:.2f}, no crash")




def test_world_frame_convention_is_global_not_per_point():
    """Real-data regression: a PER-POINT centering test shifts only the
    positive quadrant of an already-centered table, scrambling the world model
    (~1500 px systematic error). The convention must be inferred once for the
    whole table."""
    from optiona_sfr.pnl_adapter import infer_and_center
    from optiona_sfr.geometry import PITCH_L, PITCH_W
    centered = np.column_stack([rng.uniform(-52.5, 52.5, 60),
                                rng.uniform(-34, 34, 60), np.zeros(60)])
    out, was = infer_and_center(centered)
    assert not was and np.allclose(out, centered)          # untouched
    corner = centered + np.array([PITCH_L / 2, PITCH_W / 2, 0])
    out2, was2 = infer_and_center(corner)
    assert was2 and np.allclose(out2, centered, atol=1e-9)  # uniformly shifted
    # the failure mode itself: a mixed result must be impossible
    assert np.abs(out2 - centered).max() < 1e-9


def test_optimizer_cannot_diverge_from_a_terrible_init():
    """Real-data regression: unbounded LM reached f~1e14 and |C|~1e11 m,
    poisoning every aggregate. With physical bounds the result is either
    plausible or None — never absurd."""
    from optiona_sfr.geometry import camera_is_plausible
    cam_gt = gt_cam(0.0)
    kp, ln, ci = observations(cam_gt, kp_noise=40.0, n_kp=4, outlier_frac=0.5)
    for scale in [1.0, 50.0, 5000.0]:
        cam0 = Camera(cam_gt.f * scale, cam_gt.pan + 1.2, cam_gt.tilt * 0.4,
                      cam_gt.roll, cam_gt.C * scale, image_wh=WH)
        out, cost = refine_camera(cam0, kp, ln, ci, RefineCfg())
        assert out is None or camera_is_plausible(out), \
            f"implausible camera accepted at scale {scale}"
        if out is not None:
            assert 100 <= out.f <= 30000 and abs(out.C[2]) <= 120
    print("bounded optimizer: no runaway from any of 3 bad inits")


def test_plausibility_gate_rejects_absurd_cameras():
    from optiona_sfr.geometry import camera_is_plausible
    assert camera_is_plausible(gt_cam(0.0))
    bad = [Camera(1e14, 0.1, 1.3, 0.0, [6e10, -1.5e11, 5e9], image_wh=WH),
           Camera(np.nan, 0.1, 1.3, 0.0, [0, -55, -14], image_wh=WH),
           Camera(3000, 0.1, 1.3, 0.0, [0, -55, +14], image_wh=WH),   # below pitch
           Camera(3000, 0.1, 1.3, 0.0, [0, -5000, -14], image_wh=WH)]
    for b in bad:
        assert not camera_is_plausible(b)
    print("plausibility gate: 4/4 absurd cameras rejected")




def test_behind_camera_solution_is_penalised_not_rewarded():
    """THE real-data root cause: residuals that mask out behind-camera points
    make 'push the whole pitch behind the camera' a ZERO-cost optimum. The
    cheirality penalty must make that configuration expensive, and refinement
    must therefore recover a forward-facing camera from a flipped init."""
    from optiona_sfr.refine import Residuals
    from optiona_sfr.geometry import camera_is_plausible
    cam_gt = gt_cam(0.0)
    kp, ln, ci = observations(cam_gt, n_kp=6)
    fn = Residuals(kp, ln, ci, RefineCfg(), WH)
    good = np.linalg.norm(fn(cam_gt.to_vec(optimize_k1=False)))
    # a camera rotated so the pitch is behind it
    flipped = Camera(cam_gt.f, cam_gt.pan + np.pi, np.pi - cam_gt.tilt,
                     cam_gt.roll, cam_gt.C, image_wh=WH)
    bad = np.linalg.norm(fn(flipped.to_vec(optimize_k1=False)))
    print(f"cost: correct {good:.1f} vs behind-camera {bad:.1f}")
    assert bad > 100.0, "behind-camera configuration is not penalised"
    assert bad > good, "behind-camera configuration is cheaper than the truth!"

    # and the optimizer must not settle there from a bad start
    cam0 = Camera(cam_gt.f * 1.1, cam_gt.pan + 0.35, cam_gt.tilt - 0.25,
                  cam_gt.roll, cam_gt.C + np.array([6.0, -8.0, 3.0]),
                  image_wh=WH)
    out, _ = refine_camera(cam0, kp, ln, ci, RefineCfg())
    assert out is not None and camera_is_plausible(out)
    _, front = out.project(np.array([[0.0, 0.0, 0.0]]))
    assert front[0], "recovered camera does not see the pitch centre"
    print(f"recovered from bad init: grid err {grid_err(out, cam_gt):.2f} px")




def test_results_cache_invalidates_on_config_or_version_change():
    """Real-data hazard: run_sequence caches per-sequence CSVs. After fixing
    the pipeline, a stale CSV would silently re-report the broken numbers."""
    from optiona_sfr.experiments import _result_stamp
    from optiona_sfr.config import ExperimentCfg
    a, b, c = ExperimentCfg(), ExperimentCfg(), ExperimentCfg()
    assert _result_stamp(a) == _result_stamp(b)
    c.refine.beta_conic = 0.42
    assert _result_stamp(a) != _result_stamp(c)
    d = ExperimentCfg(); d.tracker.use_tripod = not d.tracker.use_tripod
    assert _result_stamp(a) != _result_stamp(d)
    print("result stamps: stable for identical cfg, distinct on change")




def test_residual_vector_length_is_invariant():
    """THE silent-failure bug: scipy requires a fixed-length residual. A
    conditionally-appended cheirality term made the length vary, scipy raised
    mid-optimization, and a broad `except` returned the UNREFINED camera — so
    refinement silently did nothing. Length must be constant for ANY
    parameters, including wild ones that put the pitch behind the camera."""
    from optiona_sfr.refine import Residuals
    cam = gt_cam(0.0)
    kp, ln, ci = observations(cam, n_kp=8)
    fn = Residuals(kp, ln, ci, RefineCfg(), WH)
    fn.gate_lines(cam)
    base = cam.to_vec(optimize_k1=False)
    lens = set()
    scales = np.array([2.0, 1.5, 1.0, 8000.0, 200.0, 200.0, 60.0])
    for _ in range(400):
        v = base + rng.normal(0, 1, 7) * scales
        r = fn(v)
        lens.add(len(r))
        assert np.isfinite(r).all(), "residuals must always be finite"
    assert len(lens) == 1, f"residual length varies: {sorted(lens)}"
    print(f"residual length invariant at {lens.pop()} over 400 wild parameter draws")


def test_refinement_actually_runs_and_improves():
    """Guards the silent no-op: the refined camera must DIFFER from the input
    and be closer to ground truth. Previously an exception path returned the
    input unchanged, which looks like success but refines nothing."""
    cam_gt = gt_cam(0.0)
    kp, ln, ci = observations(cam_gt, n_kp=8)
    cam0 = Camera(cam_gt.f * 1.08, cam_gt.pan + 0.03, cam_gt.tilt - 0.02,
                  cam_gt.roll, cam_gt.C + np.array([2.0, -2.5, 1.0]),
                  image_wh=WH)
    out, cost = refine_camera(cam0, kp, ln, ci, RefineCfg())
    assert out is not None
    moved = abs(out.f - cam0.f) + np.linalg.norm(out.C - cam0.C)
    assert moved > 1e-6, "refinement returned the input unchanged (silent no-op)"
    e0, e1 = grid_err(cam0, cam_gt), grid_err(out, cam_gt)
    print(f"refinement: {e0:.1f} px -> {e1:.1f} px (moved {moved:.2f})")
    assert e1 < e0 * 0.5, (e0, e1)




def test_normalized_annotations_are_denormalized():
    """ROOT CAUSE of the ~1150 px systematic error on real data: SoccerNet-GSR
    stores pitch annotations NORMALIZED to [0,1]. Read as pixels they collapse
    onto the image corner, roughly 1100 px from where the pitch projects — so
    a perfectly good camera scored ~1150 px while a camera fitted to those
    same corner points scored 0.8 px. The loader must detect and undo it."""
    import json, tempfile
    from pathlib import Path
    from optiona_sfr.geometry import R_to_ptr
    from optiona_sfr.experiments import (load_pitch_annotations,
                                         reprojection_error,
                                         _world_geometry_by_name)
    C = np.array([-1.35, 55.09, -9.67])
    z = np.array([25.0, 0.0, 0.0]) - C; z /= np.linalg.norm(z)
    x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    pan, tilt, roll = R_to_ptr(R)
    cam = Camera(2200, pan, tilt, roll, C, 0.0, WH)
    geom = _world_geometry_by_name()
    lines = {}
    for name in ["Big rect. right main", "Circle right", "Side line top"]:
        p, fr = cam.project(geom[name]); p = p[fr]
        k = (p[:, 0] > 0) & (p[:, 0] < WH[0]) & (p[:, 1] > 0) & (p[:, 1] < WH[1])
        if k.sum() > 3:
            lines[name] = [{"x": float(q[0] / WH[0]), "y": float(q[1] / WH[1])}
                           for q in p[k][::5]]
    assert lines, "test setup produced no visible annotation classes"
    d = Path(tempfile.mkdtemp()) / "SNGS-999"; d.mkdir(parents=True)
    (d / "Labels-GameState.json").write_text(json.dumps(
        {"info": {"version": "1.3"},
         "annotations": [{"image_id": "1000001", "lines": lines}]}))
    ann = load_pitch_annotations(d, image_wh=WH)
    pts = np.concatenate(list(ann[1].values()))
    assert pts.max() > 100, "annotations were not denormalized"
    e_ok = reprojection_error(cam, ann[1])
    assert e_ok < 2.0, f"true camera should score ~0 px, got {e_ok}"
    raw = {k: v / np.array(WH) for k, v in ann[1].items()}
    e_bug = reprojection_error(cam, raw)
    assert e_bug > 300, "the bug should be reproducible for contrast"
    print(f"annotations: denormalized -> {e_ok:.2f} px; as-pixels bug -> {e_bug:.0f} px")


def test_pixel_annotations_are_left_alone():
    """Auto-detection must not rescale annotations that are already pixels."""
    import json, tempfile
    from pathlib import Path
    from optiona_sfr.experiments import load_pitch_annotations
    d = Path(tempfile.mkdtemp()) / "SNGS-998"; d.mkdir(parents=True)
    (d / "Labels-GameState.json").write_text(json.dumps(
        {"info": {"version": "1.3"}, "annotations": [{"image_id": "1000001",
         "lines": {"Middle line": [{"x": 900.0, "y": 500.0},
                                   {"x": 950.0, "y": 700.0}]}}]}))
    ann = load_pitch_annotations(d, image_wh=WH)
    assert np.allclose(ann[1]["Middle line"][0], [900.0, 500.0])




def test_pitch_convention_is_resolved_from_data_not_guessed():
    """Real-data regression: the annotation-name -> world map was built by
    zipping a hand-written name list against PnLCalib's line order. That guess
    capped the achievable error at ~80-105 px. The convention must be resolved
    empirically, including the 180-degree rotation degeneracy (flipping BOTH
    signs is a rigid transform, so annotations alone cannot decide it)."""
    from optiona_sfr.geometry import R_to_ptr
    from optiona_sfr.pitch_model import build_name_map, resolve_convention
    truth = (True, False)
    geom = build_name_map(*truth)
    C = np.array([0.0, 55.0, -12.0])
    z = np.array([30.0, 0.0, 0.0]) - C; z /= np.linalg.norm(z)
    x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
    pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
    cam = Camera(2400, pan, tilt, roll, C, 0.0, WH)
    ann = {}
    for f in (1, 2):
        d = {}
        for name, Wm in geom.items():
            uv, fr = cam.project(Wm); uv = uv[fr]
            k = (uv[:, 0] > 0) & (uv[:, 0] < WH[0]) & \
                (uv[:, 1] > 0) & (uv[:, 1] < WH[1])
            if k.sum() > 4:
                d[name] = uv[k][::6]
        ann[f] = d
    assert len(ann[1]) >= 8, "test setup produced too few visible classes"
    rx, ty, rep = resolve_convention(ann, WH, n_frames=1, verbose=False,
                                     ref_cameras={1: cam})
    assert (rx, ty) == truth, f"resolved {(rx, ty)}, truth {truth}"
    # the true convention must be achievable to a few px, wrong ones must not
    assert min(rep.values()) < 5.0
    wrong = [v for k, v in rep.items() if k not in (truth, (not truth[0], not truth[1]))]
    assert min(wrong) > 10.0, f"wrong conventions too cheap: {wrong}"
    print(f"convention resolved correctly; best {min(rep.values()):.2f} px, "
          f"wrong {min(wrong):.1f}+ px")


def test_pitch_model_matches_known_dimensions():
    """The parametric model must reproduce FIFA/SoccerNet dimensions."""
    from optiona_sfr.pitch_model import build_name_map
    g = build_name_map(True, True)
    for need in ["Middle line", "Circle central", "Circle left", "Circle right",
                 "Big rect. left main", "Small rect. right top",
                 "Goal left crossbar", "Side line top"]:
        assert need in g, f"missing class {need}"
    side = g["Side line top"]
    assert abs(side[:, 0].max() - 52.5) < 1e-6 and abs(side[:, 0].min() + 52.5) < 1e-6
    circ = g["Circle central"]
    r = np.linalg.norm(circ[:, :2], axis=1)
    assert abs(r.mean() - 9.15) < 1e-6
    bar = g["Goal left crossbar"]
    assert abs(bar[:, 2].mean() + 2.44) < 1e-6          # 2.44 m high
    assert abs(bar[:, 1].max() - 3.66) < 1e-6           # 7.32 m wide
    pen = g["Big rect. left main"]
    assert abs(pen[:, 0].mean() + 36.0) < 1e-6          # 16.5 m from goal line




def test_convention_is_in_result_stamp_and_reaches_workers():
    """Two hazards found while tracing how the resolved convention travels:
    (a) it changes every reprojection number, so cached CSVs must invalidate;
    (b) spawned workers start with fresh module state, so it must be passed
    explicitly or Stage 4 silently runs on the default guess."""
    import inspect
    from optiona_sfr.experiments import (_result_stamp, set_pitch_convention,
                                         _CONVENTION)
    from optiona_sfr.config import ExperimentCfg
    from optiona_sfr import multigpu
    cfg = ExperimentCfg()
    before = _CONVENTION
    s1 = _result_stamp(cfg)
    set_pitch_convention(not before[0], before[1])
    s2 = _result_stamp(cfg)
    set_pitch_convention(*before)
    assert s1 != s2, "results cache does not invalidate on convention change"
    src = inspect.getsource(multigpu._exp_worker)
    assert "set_pitch_convention" in src, \
        "worker does not install the pitch convention"
    src2 = inspect.getsource(multigpu.parallel_run_experiments)
    assert "_CONVENTION" in src2, "convention is not passed to workers"
    print("convention: invalidates result cache and propagates to workers")




def test_annotation_fit_is_a_genuine_bound_when_seeded():
    """Real-data regression: 'best achievable' read 30 px while an actual
    camera scored 5.6 px — a failed multi-start fit, not a bound. Seeding the
    fit with real cameras guarantees the reported value is never worse than
    the best seed."""
    from optiona_sfr.geometry import R_to_ptr
    from optiona_sfr.pitch_model import build_name_map
    from optiona_sfr.experiments import (set_pitch_convention,
                                         reprojection_error)
    from optiona_sfr.diagnose import fit_camera_to_annotations
    prev = set_pitch_convention(True, False)
    try:
        geom = build_name_map(True, False)
        C = np.array([0.0, 55.0, -12.0])
        z = np.array([30.0, 0.0, 0.0]) - C; z /= np.linalg.norm(z)
        x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
        pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
        cam = Camera(2400, pan, tilt, roll, C, 0.0, WH)
        ann = {}
        for name, Wm in geom.items():
            uv, fr = cam.project(Wm); uv = uv[fr]
            k = (uv[:, 0] > 0) & (uv[:, 0] < WH[0]) & \
                (uv[:, 1] > 0) & (uv[:, 1] < WH[1])
            if k.sum() > 4:
                p = uv[k][::6]
                ann[name] = p + rng.normal(0, 1.0, p.shape)
        seed_err = reprojection_error(cam, ann)
        _, err = fit_camera_to_annotations(ann, WH, n_starts=2,
                                           seed_cameras=(cam,))
        assert err <= seed_err + 1e-6, (err, seed_err)
        print(f"seeded bound: {seed_err:.2f} px seed -> {err:.2f} px fit")
    finally:
        set_pitch_convention(*prev)




# ============================ method v2 ======================================
def test_fixed_center_removes_focal_distance_degeneracy():
    """THE motivation for v2: estimating 8 DOF per frame for a tripod-mounted
    camera admits the focal/distance trade-off, which on real data produced
    focal_travel of 39,475 m and jitter_f of 5512 px. Holding C fixed and
    solving 4 DOF must give measurably tighter focal estimates."""
    from optiona_sfr.method_v2 import refine_fixed_center
    Ctrue = np.array([0.0, -55.0, -14.0])
    e1, e2 = [], []
    for k in range(18):
        gt = Camera(3000 + 300 * np.sin(k * 0.3), np.deg2rad(6) + 0.004 * k,
                    np.deg2rad(75), np.deg2rad(0.4), Ctrue, 0.0, WH)
        kp, ln, ci = observations(gt, kp_noise=3.0, n_kp=8)
        seed = Camera(gt.f * 1.05, gt.pan + 0.01, gt.tilt - 0.008, gt.roll,
                      Ctrue + rng.normal(0, 1.5, 3), 0.0, WH)
        a, _ = refine_camera(seed, kp, ln, ci, RefineCfg())
        b, _ = refine_fixed_center(seed, kp, ln, ci, RefineCfg(), Ctrue)
        if a is not None:
            e1.append(a.f - gt.f)
        if b is not None:
            e2.append(b.f - gt.f)
    assert len(e2) >= 12, "v2 refinement abstained too often"
    s1, s2 = np.std(e1), np.std(e2)
    print(f"focal error std: v1 {s1:.1f} px -> v2 {s2:.1f} px ({s1/max(s2,1e-9):.1f}x)")
    assert s2 < s1


def test_estimate_fixed_center_is_robust_to_wild_outliers():
    """Per-frame 8-DOF fits produce centres scattered over tens of metres;
    the sequence centre must ignore them."""
    from optiona_sfr.method_v2 import estimate_fixed_center
    true = np.array([0.0, 52.0, -9.6])
    cams = []
    for i in range(200):
        C = true + (rng.normal(0, 0.4, 3) if i % 10 >= 3
                    else rng.normal(0, 60, 3))     # 30% wild outliers
        cams.append(Camera(3000, 0.1, 1.3, 0.0, C, 0.0, WH))
    C, spread = estimate_fixed_center(cams, verbose=False)
    err = np.linalg.norm(C - true)
    print(f"fixed centre recovered to {err:.3f} m with 30% outliers")
    assert err < 1.0, err


def test_smoothing_reduces_jitter_and_fills_gaps():
    from optiona_sfr.method_v2 import smooth_and_interpolate
    n = 300; t = np.arange(n)
    pan = 0.004 * t + np.deg2rad(5); tilt = np.full(n, 1.31)
    roll = np.zeros(n); f = 3000 + 400 * np.sin(0.02 * t)
    ok = np.ones(n, bool); ok[100:130] = False          # calibration failures
    pn = pan + rng.normal(0, 0.01, n); fn = f + rng.normal(0, 120, n)
    pn[200] += 1.5; fn[210] += 9000                     # gross outliers
    P, T, R, F, ok2, interp = smooth_and_interpolate(t, pn, tilt, roll, fn, ok)
    assert interp[100:130].all() and ok2.all(), "gaps not filled"
    assert np.sqrt(np.mean((F - f) ** 2)) < np.sqrt(np.mean((fn - f) ** 2)) / 3
    assert np.std(np.diff(F)) < np.std(np.diff(fn)) / 5
    print(f"smoothing: f RMSE {np.sqrt(np.mean((fn-f)**2)):.0f} -> "
          f"{np.sqrt(np.mean((F-f)**2)):.0f} px; "
          f"jitter {np.std(np.diff(fn)):.0f} -> {np.std(np.diff(F)):.0f}")


def test_b_grid_is_incremental_and_distinct():
    from optiona_sfr.config import b_grid
    from optiona_sfr.experiments import _result_stamp
    g = b_grid()
    assert [c.name for c in g][0] == "B0_pnlcalib_native"
    assert all(c.method == "v2" for c in g)
    assert all(not c.tracker.enabled for c in g), "v2 must not run the v1 tracker"
    stamps = {_result_stamp(c) for c in g}
    assert len(stamps) == len(g), "B-grid configs are not distinct in the cache"
    b0 = g[0].v2
    assert not b0.fix_center and not b0.refit_4dof and not b0.smooth, \
        "B0 must be the raw PnLCalib reference"




def test_mislabelled_keypoints_do_not_destroy_refinement():
    """THE v1 failure, reproduced and fixed. On real data 10% of detected
    keypoints matched a different world point than our index map assumed, and
    refinement turned a 5.6 px camera into 246-415 px. Measured on the real
    zoomed geometry (f~5080, right penalty area): ONE bad keypoint in nine took
    the result from 2.6 px to 438.9 px; two caused abstention. A robust loss
    only downweights — outliers must be REJECTED, and a refinement that fits
    worse than its seed must be discarded."""
    from optiona_sfr.geometry import R_to_ptr, LINES_WORLD
    C = np.array([0.01, 50.17, -9.50])
    z = np.array([40.0, 0.0, 0.0]) - C; z /= np.linalg.norm(z)
    x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
    pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
    gt = Camera(5079.5, pan, tilt, roll, C, 0.0, WH)
    base = [[52.5, -20.16, 0], [36, -20.16, 0], [36, 20.16, 0],
            [52.5, 20.16, 0], [47, -9.16, 0], [47, 9.16, 0],
            [52.5, -9.16, 0], [52.5, 9.16, 0], [41.5, 0, 0]]
    X, U = [], []
    for b in base:
        uv, fr = gt.project(np.array([b], float))
        if fr[0]:
            X.append(b); U.append(uv[0])
    X = np.array(X); U = np.array(U) + rng.normal(0, 2.0, (len(X), 2))
    ln = []
    for li in [4, 5, 6, 21, 22, 23]:
        A, B = LINES_WORLD[li - 1]
        prj, fr = gt.project(np.stack([A, B]))
        if fr.all():
            ln.append((li, prj + rng.normal(0, 2.0, prj.shape), 0.9))
    G = np.column_stack([rng.uniform(30, 52, 40), rng.uniform(-25, 25, 40),
                         np.zeros(40)])
    a, fa = gt.project(G)
    err = lambda c: float(np.linalg.norm(a[fa] - c.project(G)[0][fa], axis=1).mean())
    seed = Camera(gt.f * 1.01, gt.pan + 0.002, gt.tilt - 0.0015, gt.roll,
                  C + np.array([.3, -.4, .2]), 0.0, WH)
    for nbad in (1, 2, 3):
        Xb = X.copy()
        for j in range(nbad):
            Xb[j] = [-X[j][0], X[j][1], X[j][2]]      # mirrored = mislabelled
        kp = (np.arange(len(Xb)), U, np.ones(len(Xb)), Xb)
        out, _ = refine_camera(seed, kp, ln, [], RefineCfg())
        assert out is not None, f"abstained with {nbad}/9 bad keypoints"
        e = err(out)
        assert e < 15.0, f"{nbad}/9 bad keypoints -> {e:.1f} px"
    print(f"robust refinement: survives up to 3/9 mislabelled keypoints "
          f"(seed {err(seed):.1f} px -> {e:.1f} px)")


def test_refinement_never_returns_worse_than_seed():
    """'Do no harm': with a trusted seed, refinement must improve it or return
    it unchanged — never a regression, in either method."""
    from optiona_sfr.refine import score_camera
    from optiona_sfr.method_v2 import refine_fixed_center
    cam_gt = gt_cam(0.0)
    kp, ln, ci = observations(cam_gt, n_kp=8)
    seed = Camera(cam_gt.f * 1.02, cam_gt.pan + 0.01, cam_gt.tilt - 0.005,
                  cam_gt.roll, cam_gt.C, 0.0, WH)
    for label, fn in [("v1", lambda: refine_camera(seed, kp, ln, ci, RefineCfg())),
                      ("v2", lambda: refine_fixed_center(seed, kp, ln, ci,
                                                         RefineCfg(), cam_gt.C))]:
        out, _ = fn()
        assert out is not None, f"{label} abstained on a well-posed problem"
        s_seed = score_camera(seed, kp, ln, RefineCfg())
        s_out = score_camera(out, kp, ln, RefineCfg())
        assert s_out <= s_seed + 1e-9, f"{label} regressed: {s_seed} -> {s_out}"
    print("do-no-harm holds for both v1 and v2 refinement")




def test_wrong_fixed_center_is_detectably_worse():
    """The sequence-level guard in run_sequence_v2 adopts a single optical
    centre only if refitting about it fits at least as well as the per-frame
    cameras. This asserts the signal that guard reads: a wrong centre yields a
    clearly worse fit, so it can be detected and rejected. On real data an
    unchecked centre turned a 10.3 px sequence into 369.7 px."""
    from optiona_sfr.method_v2 import refine_fixed_center
    from optiona_sfr.refine import score_camera
    cam_gt = gt_cam(0.0)
    kp, ln, ci = observations(cam_gt, n_kp=10, kp_noise=1.5)
    good, _ = refine_fixed_center(cam_gt, kp, ln, ci, RefineCfg(), cam_gt.C)
    wrong_C = cam_gt.C + np.array([12.0, 15.0, 4.0])
    bad, _ = refine_fixed_center(cam_gt, kp, ln, ci, RefineCfg(), wrong_C)
    s_good = score_camera(good, kp, ln, RefineCfg())
    s_bad = score_camera(bad, kp, ln, RefineCfg()) if bad is not None else np.inf
    print(f"fixed centre: correct {s_good:.2f} px vs wrong {s_bad:.2f} px")
    assert s_bad > max(2.0 * s_good, s_good + 5.0), \
        "a wrong centre is not detectably worse — the guard cannot fire"


def test_quality_weighted_center_is_accurate():
    from optiona_sfr.method_v2 import estimate_fixed_center
    true = np.array([0.0, 52.0, -9.6])
    cams, w = [], []
    for i in range(150):
        bad = i % 4 == 0
        C = true + (rng.normal(0, 40, 3) if bad else rng.normal(0, 0.3, 3))
        cams.append(Camera(3000, 0.1, 1.3, 0.0, C, 0.0, WH))
        w.append(1.0 / (1.0 + (200.0 if bad else 3.0)))
    C2, _ = estimate_fixed_center(cams, w, verbose=False)
    assert np.linalg.norm(C2 - true) < 0.5


def test_refinement_improves_a_good_seed():
    """Complement to do-no-harm: the guard must not block genuine gains. With
    the median-based test it blocked EVERY improvement (real data showed
    A0 == A1 == A2 == 155.184 px exactly); comparing the robust cost that LM
    minimises fixes that."""
    cam_gt = gt_cam(0.0)
    kp, ln, ci = observations(cam_gt, n_kp=10, kp_noise=1.5)
    seed = Camera(cam_gt.f * 1.01, cam_gt.pan + 0.004, cam_gt.tilt - 0.003,
                  cam_gt.roll, cam_gt.C + np.array([0.4, -0.5, 0.2]), 0.0, WH)
    out, _ = refine_camera(seed, kp, ln, ci, RefineCfg())
    e0, e1 = grid_err(seed, cam_gt), grid_err(out, cam_gt)
    print(f"refinement improves a good seed: {e0:.2f} -> {e1:.2f} px")
    assert e1 < e0 * 0.6, (e0, e1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
