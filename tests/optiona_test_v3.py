"""Method v3 regressions: parity, sub-pixel, so(3), centre prior, BIC, repair."""
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from optiona_sfr.geometry import Camera, R_to_ptr, ptr_to_R
from optiona_sfr.config import RefineCfg, c_grid
from optiona_sfr.parity import flip_pitch_half
from optiona_sfr.method_v3 import (so3_log, so3_exp, slerp_rotations,
                                   smooth_rotations_so3, smooth_log_focal,
                                   center_prior_residual, bic, bic_select,
                                   mahalanobis_repair)

rng = np.random.default_rng(0)
WH = (1920, 1080)


def _cam(C=(0.01, 50.17, -9.50), f=5079.5, target=(40.0, 0.0, 0.0)):
    C = np.asarray(C, float)
    z = np.asarray(target, float) - C; z /= np.linalg.norm(z)
    x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
    pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
    return Camera(f, pan, tilt, roll, C, 0.0, WH)


def test_pitch_half_flip_is_proper_and_symmetry_preserving():
    """The realizable ambiguity is a PROPER 180 deg rotation (det=+1), not a
    reflection (det=-1) — every audited frame had det(R)=+1.000, so there is
    no improper rotation to undo."""
    cam = _cam()
    fl = flip_pitch_half(cam)
    assert abs(np.linalg.det(fl.R) - 1.0) < 1e-9, "flip must stay proper"
    P = np.array([[30., 12., 0.], [-20., -8., 0.], [41.5, 0., 0.]])
    Pm = np.column_stack([-P[:, 0], -P[:, 1], P[:, 2]])
    a, _ = cam.project(P); b, _ = fl.project(Pm)
    assert np.abs(a - b).max() < 1e-6, "flip must map the pitch onto itself"
    assert np.allclose(flip_pitch_half(fl).C, cam.C, atol=1e-9), "involutive"


def test_so3_log_exp_roundtrip():
    for _ in range(300):
        w = rng.normal(0, 1, 3)
        w = w / np.linalg.norm(w) * rng.uniform(0, np.pi * 0.999)
        assert np.linalg.norm(so3_log(so3_exp(w)) - w) < 1e-8


def test_geodesic_interpolation_beats_euler():
    """Euler interpolation is non-geodesic; our ZXZ parameterisation couples
    pan and roll, so the midpoint is NOT equidistant from the endpoints."""
    R0, R1 = ptr_to_R(0.1, 1.2, 0.05), ptr_to_R(2.9, 1.5, -0.4)
    d = lambda A, B: np.linalg.norm(so3_log(A.T @ B))
    m_geo = slerp_rotations(R0, R1, 0.5)
    p0, t0, r0 = R_to_ptr(R0); p1, t1, r1 = R_to_ptr(R1)
    m_eul = ptr_to_R((p0 + p1) / 2, (t0 + t1) / 2, (r0 + r1) / 2)
    assert abs(d(R0, m_geo) - d(m_geo, R1)) < 1e-6
    assert abs(d(R0, m_eul) - d(m_eul, R1)) > 1e-3


def test_so3_smoothing_reduces_error_and_fills_gaps():
    n = 200
    Rs = [ptr_to_R(0.004 * i, 1.31 + 0.0004 * np.sin(0.1 * i), 0.01)
          for i in range(n)]
    ok = np.ones(n, bool); ok[80:100] = False
    noisy = [R @ so3_exp(rng.normal(0, 0.004, 3)) for R in Rs]
    sm, interp = smooth_rotations_so3(noisy, ok)
    e_raw = np.mean([np.linalg.norm(so3_log(Rs[i].T @ noisy[i]))
                     for i in range(n) if ok[i]])
    e_sm = np.mean([np.linalg.norm(so3_log(Rs[i].T @ sm[i])) for i in range(n)])
    assert e_sm < e_raw and interp[80:100].all()


def test_focal_smoothed_in_log_space():
    n = 200
    f = 3000 * np.exp(0.002 * np.arange(n))
    fn = f * np.exp(rng.normal(0, 0.05, n))
    fs = smooth_log_focal(fn, np.ones(n, bool))
    # multiplicative error must shrink; the factor depends on the window
    # (0.039 -> 0.021 at the default 9, -> 0.010 at 31)
    assert np.mean(np.abs(fs / f - 1)) < np.mean(np.abs(fn / f - 1)) / 1.7
    wide = smooth_log_focal(fn, np.ones(n, bool), window=31)
    assert np.mean(np.abs(wide / f - 1)) < np.mean(np.abs(fs / f - 1))


def test_center_prior_has_gaussian_core_and_hard_box():
    """Pure Huber would make LARGE deviations cheap along the focal/distance
    direction — the degeneracy the prior exists to close. Core must be
    quadratic, tail bounded, and the box must be a wall."""
    r = [float(np.linalg.norm(center_prior_residual(np.array([d, 0, 0]),
                                                    np.zeros(3))))
         for d in (0.1, 0.3, 0.9, 1.5, 2.5)]
    assert r[0] < r[1] < r[2] < r[3] < r[4]
    assert abs(r[1] - 1.0) < 1e-6                 # 1 sigma -> unit residual
    assert r[4] > 10 * r[3]                       # steep wall past the box


def test_bic_penalises_extra_parameters():
    r = rng.normal(0, 1, 20)
    assert bic(r, 4) < bic(r, 8)


def test_bic_envelope_keeps_baseline_without_a_margin_win():
    """Non-inferiority: an equal-fitting candidate with MORE parameters must
    not displace the baseline."""
    cam = _cam()
    X = np.array([[52.5, -20.16, 0], [36, -20.16, 0], [36, 20.16, 0],
                  [47, -9.16, 0], [47, 9.16, 0], [41.5, 0, 0]], float)
    uv, fr = cam.project(X)
    kp_obs = (np.arange(len(X)), uv + rng.normal(0, 1.5, uv.shape),
              np.ones(len(X)), X)
    name, _ = bic_select([("baseline", cam), ("v3", cam)], kp_obs, [],
                         RefineCfg(), {"baseline": 4, "v3": 8})
    assert name == "baseline"


def test_mahalanobis_repair_recovers_true_indices():
    cam = _cam()
    kp_world = {i + 1: np.array(p, float) for i, p in enumerate(
        [[52.5, -20.16, 0], [36, -20.16, 0], [36, 20.16, 0], [52.5, 20.16, 0],
         [47, -9.16, 0], [47, 9.16, 0], [52.5, -9.16, 0], [52.5, 9.16, 0],
         [41.5, 0, 0], [52.5, -3.66, -2.44], [0, 0, 0], [-36, 0, 0]])}
    truth = [1, 2, 4, 5, 7, 9]
    uv = np.array([cam.project(kp_world[k][None, :])[0][0] +
                   rng.normal(0, 2.0, 2) for k in truth])
    assign, _ = mahalanobis_repair(cam, uv, np.ones(len(uv)), kp_world,
                                   RefineCfg())
    assert sum(a == t for a, t in zip(assign, truth)) >= len(truth) - 1


def test_subpixel_beats_heatmap_quantisation():
    """The detector quantises to the heatmap stride (~2-8 px at 1080p), which
    plausibly IS the 7-10 px floor. This is the only v3 mechanism that creates
    information rather than redistributing it."""
    import cv2
    from optiona_sfr.subpixel import refine_corner
    for stride in (4, 8):
        raw, ref = [], []
        for _ in range(60):
            img = np.zeros((121, 121), np.float32)
            cx = 60 + rng.uniform(-stride / 2, stride / 2)
            cy = 60 + rng.uniform(-stride / 2, stride / 2)
            a1 = rng.uniform(0, np.pi); a2 = a1 + np.pi / 2 + rng.uniform(-.25, .25)
            yy, xx = np.mgrid[0:121, 0:121]
            for a in (a1, a2):
                d = np.abs(-np.sin(a) * (xx - cx) + np.cos(a) * (yy - cy))
                img = np.maximum(img, np.clip(2.0 - d, 0, 1))
            img = (cv2.GaussianBlur(img, (0, 0), 1.1) * 200 +
                   rng.normal(0, 3.0, (121, 121))).astype(np.float32)
            q = np.array([round(cx / stride) * stride,
                          round(cy / stride) * stride], float)
            p, _ = refine_corner(img, q, win=13, max_shift=stride * 1.5)
            raw.append(np.linalg.norm(q - [cx, cy]))
            ref.append(np.linalg.norm(p - [cx, cy]))
        assert np.mean(ref) < np.mean(raw) / 1.8, \
            f"stride {stride}: {np.mean(raw):.2f} -> {np.mean(ref):.2f}"


def test_c_grid_is_incremental_and_distinct():
    from optiona_sfr.experiments import _result_stamp
    g = c_grid()
    assert g[0].name == "C0_baseline" and all(c.method == "v3" for c in g)
    assert len({_result_stamp(c) for c in g}) == len(g)
    z = g[0].v3
    assert not (z.parity_check or z.shared_center or z.smooth or z.bic_envelope)




def test_trust_region_rejects_large_moves():
    """A cost-based do-no-harm test CANNOT detect harm caused by a wrong
    observation — it measures fit to that same wrong observation. On real data
    one mislabelled line in ten took a 13.9 px camera to 488.9 px while the
    robust cost improved. Bounding the move in GEOMETRY space caps the damage
    whatever is corrupt."""
    from optiona_sfr.refine import projected_displacement, refine_camera
    from optiona_sfr.geometry import LINES_WORLD
    cam = _cam()
    far = Camera(cam.f * 1.5, cam.pan + 0.05, cam.tilt - 0.03, cam.roll,
                 cam.C, 0.0, WH)
    d = projected_displacement(cam, far)
    assert d > 100, d
    assert projected_displacement(cam, cam) < 1e-9

    # observations poisoned so the optimiser is pulled far away
    X = np.array([[52.5, -20.16, 0], [36, -20.16, 0], [36, 20.16, 0],
                  [47, -9.16, 0], [47, 9.16, 0], [41.5, 0, 0]], float)
    uv, _ = cam.project(X)
    uv = uv + rng.normal(0, 2.0, uv.shape)
    uv[0] += np.array([600.0, 400.0])          # gross, but inside no gate
    kp = (np.arange(len(X)), uv, np.ones(len(X)), X)
    tight, _ = refine_camera(cam, kp, [], [], RefineCfg(trust_region_px=15.0,
                                                       trust_region_scale=0.0))
    # The guard bounds the move WHERE THE EVIDENCE IS (the observed world
    # points), not over a full-pitch grid — extrapolating far outside a zoomed
    # view amplifies small pose changes and would reject legitimate
    # corrections. Measure the same quantity the guard measures.
    assert projected_displacement(cam, tight, X=X) <= 15.0 + 1e-6, \
        "trust region did not bound the move over the observed region"


def test_parity_verdict_reports_healthy_sequences_as_healthy():
    """Logic regression: every sequence was labelled 'H2-camera-type' even at
    7-13 px direct error, because the healthy case was never checked first."""
    import inspect
    from optiona_sfr import parity
    src = inspect.getsource(parity.diagnose_sequence_failure)
    i_healthy = src.find('"healthy"')
    i_h1 = src.find('"H1-pitch-half"')
    assert 0 < i_healthy < i_h1, \
        "the healthy case must be tested before any failure hypothesis"


def test_baseline_cache_ignores_non_frame_pickles():
    """The per-sequence baseline cache lives beside the frame pickles as
    `_baseline_rlN.pkl`; frame enumeration must not mistake it for a frame.
    (Runtime context: recomputing PnLCalib for all 12 grid configs meant 36,000
    redundant heuristic_voting calls, which exhausted a Kaggle session before
    any results were produced.)"""
    import tempfile, pickle, pathlib
    from optiona_sfr.experiments import cached_sequence_ids
    d = pathlib.Path(tempfile.mkdtemp()) / "cache" / "SNGS-900"
    d.mkdir(parents=True)
    for i in (1, 2, 3):
        with open(d / f"{i:06d}.pkl", "wb") as f:
            pickle.dump({"kp": {}, "lines": {}, "wh": (1920, 1080),
                         "flow": None}, f)
    with open(d / "_baseline_rl1.pkl", "wb") as f:
        pickle.dump({1: None}, f)
    ids = cached_sequence_ids(d.parent)
    assert ids == ["SNGS-900"]
    frames = sorted(int(p.stem) for p in d.glob("*.pkl") if p.stem.isdigit())
    assert frames == [1, 2, 3], frames


def test_frame_enumeration_is_centralised_and_ignores_artifacts():
    """Crash regression: `invalid literal for int(): '_baseline_rl1'`. Three
    separate `int(p.stem)` globs existed; guarding two still left the v2/v3
    runners crashing. There is now ONE enumerator, the baseline cache lives in
    a separate directory, and both are asserted here."""
    import tempfile, pickle, pathlib, inspect
    from optiona_sfr.experiments import frame_indices, cached_sequence_ids
    from optiona_sfr.detection import baseline_cache_path
    from optiona_sfr import method_v2, method_v3, detection

    root = pathlib.Path(tempfile.mkdtemp()) / "cache"
    seq = root / "SNGS-900"; seq.mkdir(parents=True)
    for i in (1, 2, 3):
        with open(seq / f"{i:06d}.pkl", "wb") as f:
            pickle.dump({"kp": {}, "lines": {}, "wh": (1920, 1080),
                         "flow": None}, f)
    # legacy artifact sitting beside the frames must not break enumeration
    with open(seq / "_baseline_rl1.pkl", "wb") as f:
        pickle.dump({1: None}, f)
    assert frame_indices(root, "SNGS-900") == [1, 2, 3]
    assert cached_sequence_ids(root) == ["SNGS-900"]

    # the baseline cache is written OUTSIDE the frame directory
    p = baseline_cache_path(root, "SNGS-900", True)
    assert p.parent.name == "_baselines" and p.parent != seq
    with open(p, "wb") as f:
        pickle.dump({1: None}, f)
    assert cached_sequence_ids(root) == ["SNGS-900"], \
        "the _baselines directory must not be mistaken for a sequence"

    # no runner may enumerate frames on its own again
    for mod in (method_v2, method_v3, detection):
        src = inspect.getsource(mod)
        assert "int(p.stem)" not in src and "int(q.stem)" not in src, \
            f"{mod.__name__} enumerates frames itself; use frame_indices()"




def test_bundle_adjust_is_sparse_and_improves_the_centre():
    """Runtime regression: the shared-centre bundle optimised ~487 parameters
    with a DENSE numerical Jacobian, so every iteration cost ~487 evaluations
    of 120 residual blocks — a Kaggle session ended mid-grid because of it.
    The Jacobian is block-arrow (each frame touches only the shared centre and
    its own 4 parameters); declaring that sparsity is the fix."""
    import time, inspect
    from optiona_sfr.method_v3 import bundle_adjust
    from optiona_sfr.geometry import R_to_ptr
    assert "jac_sparsity" in inspect.getsource(bundle_adjust), \
        "bundle_adjust must declare its Jacobian sparsity"
    Ctrue = np.array([0.5, 50.0, -9.5])
    frames = list(range(1, 121)); obs = {}; cams = {}
    for t in frames:
        z = np.array([35. + 0.05 * t, 0, 0]) - Ctrue; z /= np.linalg.norm(z)
        x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
        pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
        gt = Camera(4900 + 80 * np.sin(0.05 * t), pan, tilt, roll, Ctrue, 0.0, WH)
        Xs, U = [], []
        for b in [[52.5, -20.16, 0], [36, -20.16, 0], [36, 20.16, 0],
                  [52.5, 20.16, 0], [47, -9.16, 0], [47, 9.16, 0],
                  [41.5, 0, 0], [52.5, 9.16, 0]]:
            uv, fr = gt.project(np.array([b], float))
            if fr[0]:
                Xs.append(b); U.append(uv[0] + rng.normal(0, 2, 2))
        Xs = np.array(Xs); U = np.array(U)
        obs[t] = ((np.arange(len(Xs)), U, np.ones(len(Xs)), Xs), [], [], None)
        cams[t] = Camera(gt.f * (1 + rng.normal(0, 0.01)),
                         gt.pan + rng.normal(0, 0.002),
                         gt.tilt + rng.normal(0, 0.002), gt.roll,
                         Ctrue + rng.normal(0, 1.5, 3), 0.0, WH)
    C0 = np.median([c.C for c in cams.values()], axis=0)
    t0 = time.time()
    C, out = bundle_adjust(frames, obs, cams, C0, RefineCfg(), verbose=False)
    dt = time.time() - t0
    assert C is not None
    assert np.linalg.norm(C - Ctrue) <= np.linalg.norm(C0 - Ctrue) + 1e-9
    assert dt < 60, f"bundle took {dt:.0f}s — too slow for a 12-config grid"
    print(f"bundle: {dt:.1f}s, centre err "
          f"{np.linalg.norm(C0-Ctrue):.3f} -> {np.linalg.norm(C-Ctrue):.3f} m")




def test_summarize_handles_configs_with_no_results():
    """Blocking bug at the very last step: Stage 5 iterates every grid,
    including configs that were never run (A-grid with RUN_A=False), and
    `summarize` raised KeyError: "['seq'] not found in axis" on the empty
    frame — losing an entire completed run's results at the reporting step."""
    import tempfile, pathlib
    from optiona_sfr.experiments import summarize
    d = pathlib.Path(tempfile.mkdtemp())
    per, agg = summarize("never_run_config", d)
    assert len(per) == 0 and len(agg) == 0
    (d / "some_config").mkdir()
    per, agg = summarize("some_config", d)
    assert len(per) == 0 and len(agg) == 0




def test_parity_samples_enough_frames_to_be_trustworthy():
    """A four-frame sample reported SNGS-118 healthy at 13.5 px when its true
    median over 750 frames is 616 px, and SNGS-119 broken at 902 px when its
    true median is 10.5 px — and that false signal sent an entire
    investigation down the wrong path. The default must be a broad,
    evenly-spaced sample."""
    import inspect
    from optiona_sfr import parity
    sig = inspect.signature(parity.diagnose_sequence_failure)
    assert sig.parameters["frames"].default is None, \
        "frames must default to a computed sample, not a fixed short list"
    assert sig.parameters["n_sample"].default >= 30, \
        "too few frames to distinguish a broken sequence from a lucky sample"
    src = inspect.getsource(parity.diagnose_sequence_failure)
    assert "frame_indices" in src and "step" in src, \
        "the sample must be spread across the whole sequence"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
