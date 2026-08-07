"""Temporal smoothing of PnLCalib output: jitter down, accuracy not sacrificed."""
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from optiona_sfr.geometry import Camera, R_to_ptr
from optiona_sfr.smooth import smooth_cameras, cap_deviation, _displacement
from optiona_sfr.config import s_grid

WH = (1920, 1080)
G = np.column_stack([np.linspace(-45, 45, 50), np.linspace(-28, 28, 50),
                     np.zeros(50)])


def _trajectory(seed=0, n=400, fail_rate=0.05, outlier_rate=0.02, noise=1.0):
    """A broadcast pan/zoom, plus PnLCalib-like per-frame noise, failures and
    occasional gross outliers. Noise is calibrated so the median error lands in
    the 7-10 px band measured on SoccerNet-GSR."""
    rng = np.random.default_rng(seed)
    frames = list(range(1, n + 1))
    gt, obs = {}, {}
    for i, t in enumerate(frames):
        C = np.array([0.5 + 0.3 * np.sin(0.01 * i), 50.0, -9.5])
        z = np.array([20 + 30 * np.sin(0.006 * i), 0, 0]) - C
        z /= np.linalg.norm(z)
        x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
        pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
        gt[t] = Camera(4900 + 300 * np.sin(0.004 * i), pan, tilt, roll, C,
                       0.0, WH)
        if rng.random() < fail_rate:
            obs[t] = None
            continue
        s = 8.0 if rng.random() < outlier_rate else 1.0
        s *= noise
        g = gt[t]
        obs[t] = Camera(g.f * (1 + rng.normal(0, 0.0013 * s)),
                        g.pan + rng.normal(0, 0.00027 * s),
                        g.tilt + rng.normal(0, 0.00013 * s),
                        g.roll + rng.normal(0, 0.00013 * s),
                        g.C + rng.normal(0, 0.04 * s, 3), 0.0, WH)
    return frames, gt, obs


def _err(frames, gt, cams):
    e = []
    for t in frames:
        c = cams.get(t)
        if c is None:
            continue
        a, fa = gt[t].project(G); b, _ = c.project(G)
        m = fa & np.isfinite(b).all(1)
        if m.sum() > 4:
            e.append(np.linalg.norm(a[m] - b[m], axis=1).mean())
    return float(np.median(e))


def _jit(frames, cams, attr):
    v = [getattr(cams[t], attr) for t in frames if cams.get(t) is not None]
    return float(np.std(np.diff(v)))


def _comp(frames, cams):
    return sum(cams.get(t) is not None for t in frames) / len(frames)


def test_smoothing_improves_jitter_completeness_and_accuracy():
    """In the measured regime (7-10 px per-frame error) the smoother must not
    trade accuracy for smoothness: it should win on every axis."""
    frames, gt, obs = _trajectory()
    sm, info = smooth_cameras(frames, obs, max_dev_px=8.0)
    e0, e1 = _err(frames, gt, obs), _err(frames, gt, sm)
    j0, j1 = _jit(frames, obs, "f"), _jit(frames, sm, "f")
    t0, t1 = _jit(frames, obs, "tilt"), _jit(frames, sm, "tilt")
    c0, c1 = _comp(frames, obs), _comp(frames, sm)
    print(f"medre {e0:.2f}->{e1:.2f} | jitter_f {j0:.1f}->{j1:.1f} | "
          f"jitter_tilt {t0:.6f}->{t1:.6f} | completeness {c0:.3f}->{c1:.3f}")
    assert j1 < j0 * 0.75, (j0, j1)
    assert t1 < t0
    assert c1 > c0 and c1 == 1.0
    assert e1 <= e0, "smoothing must not cost accuracy in this regime"


def test_deviation_cap_is_respected_per_frame():
    """The accuracy guarantee: no frame may move further than the cap from
    PnLCalib's own estimate, so worst-case damage is bounded by construction."""
    frames, gt, obs = _trajectory(seed=3, noise=6.0)     # heavy noise
    for cap in (2.0, 8.0):
        sm, _ = smooth_cameras(frames, obs, max_dev_px=cap)
        worst = 0.0
        for t in frames:
            if obs.get(t) is None or sm.get(t) is None:
                continue
            worst = max(worst, _displacement(obs[t], sm[t]))
        assert worst <= cap + 0.5, f"cap {cap}: worst move {worst:.2f} px"


def test_cap_blends_along_the_geodesic():
    frames, gt, obs = _trajectory(seed=1, n=60)
    a = obs[[t for t in frames if obs.get(t)][0]]
    far = Camera(a.f * 1.3, a.pan + 0.02, a.tilt - 0.01, a.roll, a.C, 0.0, WH)
    for cap in (5.0, 25.0):
        out = cap_deviation(a, far, cap)
        assert _displacement(a, out) <= cap + 0.5


def test_gross_outlier_frames_do_not_bend_the_trajectory():
    """One badly wrong frame must not drag its neighbours (MAD rejection)."""
    frames, gt, obs = _trajectory(seed=5, outlier_rate=0.0)
    bad = frames[len(frames) // 2]
    g = gt[bad]
    obs[bad] = Camera(g.f * 1.6, g.pan + 0.15, g.tilt - 0.08, g.roll,
                      g.C + np.array([4.0, 6.0, 2.0]), 0.0, WH)
    sm, info = smooth_cameras(frames, obs, max_dev_px=8.0)
    assert info["rejected"] >= 1, "the gross outlier was not rejected"
    nb = [bad - 3, bad - 2, bad + 2, bad + 3]
    for t in nb:
        a, fa = gt[t].project(G); b, _ = sm[t].project(G)
        m = fa & np.isfinite(b).all(1)
        assert np.linalg.norm(a[m] - b[m], axis=1).mean() < 25.0


def test_s_grid_isolates_each_effect():
    """The grid must let the variant gain and the smoothing gain be read
    separately, otherwise a combined number cannot be attributed."""
    from optiona_sfr.experiments import _result_stamp
    g = {c.name: c for c in s_grid()}
    assert not g["S0_pnlcalib"].smooth.enabled
    assert not g["S0_pnlcalib"].smooth.select_variant
    assert g["S1_variant"].smooth.select_variant and not g["S1_variant"].smooth.enabled
    assert g["S2_smooth"].smooth.enabled and not g["S2_smooth"].smooth.select_variant
    assert g["S3_variant_smooth"].smooth.enabled and g["S3_variant_smooth"].smooth.select_variant
    assert all(c.method == "smooth" for c in g.values())
    assert len({_result_stamp(c) for c in g.values()}) == len(g)




def test_self_fit_selects_the_better_variant_without_ground_truth():
    """PnLCalib's `refine_lines` setting wins in opposite directions on
    different sequences — measured 13.5 vs 616.4 px on SNGS-118 and 902.3 vs
    10.5 px on SNGS-119 — so any fixed choice guarantees a broken sequence.
    Self-fit (how well a camera explains its OWN detections) is a
    ground-truth-free criterion for choosing per frame; this asserts it
    actually tracks true accuracy."""
    from optiona_sfr.refine import score_camera
    from optiona_sfr.config import RefineCfg
    rng2 = np.random.default_rng(0)
    sel, fixed, oracle = [], [], []
    for _ in range(200):
        C = np.array([0.5, 50.0, -9.5])
        z = np.array([rng2.uniform(10, 45), 0, 0]) - C; z /= np.linalg.norm(z)
        x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)
        pan, tilt, roll = R_to_ptr(np.stack([x, np.cross(z, x), z]))
        gt = Camera(rng2.uniform(4000, 5500), pan, tilt, roll, C, 0.0, WH)
        Xs, U = [], []
        for b in [[52.5, -20.16, 0], [36, -20.16, 0], [36, 20.16, 0],
                  [52.5, 20.16, 0], [47, -9.16, 0], [47, 9.16, 0],
                  [41.5, 0, 0], [52.5, 9.16, 0], [52.5, -9.16, 0]]:
            uv, fr = gt.project(np.array([b], float))
            if fr[0] and 0 < uv[0, 0] < WH[0] and 0 < uv[0, 1] < WH[1]:
                Xs.append(b); U.append(uv[0] + rng2.normal(0, 2.0, 2))
        if len(Xs) < 5:
            continue
        kp = (np.arange(len(Xs)), np.array(U), np.ones(len(Xs)), np.array(Xs))

        def perturb(scale):
            return Camera(gt.f * (1 + rng2.normal(0, 0.003 * scale)),
                          gt.pan + rng2.normal(0, 0.0008 * scale),
                          gt.tilt + rng2.normal(0, 0.0005 * scale), gt.roll,
                          C + rng2.normal(0, 0.15 * scale, 3), 0.0, WH)

        def terr(c):
            a, fa = gt.project(G); b_, _ = c.project(G)
            m = fa & np.isfinite(b_).all(1)
            return float(np.linalg.norm(a[m] - b_[m], axis=1).mean())

        sA = rng2.choice([1.0, 30.0], p=[0.7, 0.3])
        sB = 1.0 if sA > 10 else rng2.choice([1.0, 30.0], p=[0.7, 0.3])
        A, B = perturb(sA), perturb(sB)
        fA = score_camera(A, kp, [], RefineCfg())
        fB = score_camera(B, kp, [], RefineCfg())
        pick = B if fB < fA / 1.10 else A
        sel.append(terr(pick)); fixed.append(terr(A))
        oracle.append(min(terr(A), terr(B)))
    sel, fixed, oracle = map(np.array, (sel, fixed, oracle))
    agree = float(np.mean(sel <= oracle * 1.05))
    print(f"variant selection: fixed {np.median(fixed):.1f} px -> self-fit "
          f"{np.median(sel):.1f} px (oracle {np.median(oracle):.1f}); "
          f"picks the better one {100*agree:.0f}% of the time")
    assert np.median(sel) < np.median(fixed) * 0.75
    assert agree > 0.8


def test_variant_selection_is_wired_and_defaults_on():
    from optiona_sfr.config import SmoothCfg, s_grid
    from optiona_sfr.detection import select_baseline_variant
    import inspect
    from optiona_sfr import smooth as sm
    assert SmoothCfg().select_variant is True
    assert "select_baseline_variant" in inspect.getsource(sm.run_sequence_smooth)
    names = [c.name for c in s_grid()]
    assert names[:4] == ["S0_pnlcalib", "S1_variant", "S2_smooth",
                         "S3_variant_smooth"], names


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
