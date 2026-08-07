"""Core-math tests. These must pass BEFORE any Kaggle run (report Phase 2)."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from optiona_sfr.geometry import (Camera, ptr_to_R, R_to_ptr, circle_conic,
                                  conic_image, sampson_conic_distance,
                                  point_line_distance, LINES_WORLD,
                                  CIRCLES_WORLD)
from optiona_sfr.refine import refine_camera
from optiona_sfr.config import RefineCfg
from optiona_sfr.tracker import estimate_tripod

rng = np.random.default_rng(0)


def make_cam(k1=0.0):
    # Plausible elevated main broadcast camera (BroadTrack default region).
    return Camera(f=3200.0, pan=np.deg2rad(8.0), tilt=np.deg2rad(75.0),
                  roll=np.deg2rad(0.5), C=[0.0, -55.0, -14.0], k1=k1,
                  image_wh=(1920, 1080))


def test_euler_roundtrip():
    for _ in range(200):
        pan = rng.uniform(-np.pi, np.pi)
        tilt = rng.uniform(0.05, np.pi - 0.05)
        roll = rng.uniform(-0.5, 0.5)
        R = ptr_to_R(pan, tilt, roll)
        p2, t2, r2 = R_to_ptr(R)
        R2 = ptr_to_R(p2, t2, r2)
        assert np.allclose(R, R2, atol=1e-9)


def test_project_backproject_ground():
    for k1 in [0.0, -0.08, 0.05]:
        cam = make_cam(k1)
        X = np.column_stack([rng.uniform(-50, 50, 40),
                             rng.uniform(-33, 33, 40),
                             np.zeros(40)])
        uv, front = cam.project(X)
        inimg = front & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
        Xb, valid = cam.backproject_ground(uv[inimg])
        assert valid.all()          # in-image points must always invert
        assert np.allclose(Xb[:, :2], X[inimg][:, :2], atol=1e-3)


def test_conic_sampson_zero_on_circle():
    cam = make_cam(0.0)
    H = cam.homography_ground()
    for (cx, cy, r) in CIRCLES_WORLD:
        a = rng.uniform(0, 2 * np.pi, 50)
        W = np.stack([cx + r * np.cos(a), cy + r * np.sin(a),
                      np.zeros_like(a)], 1)
        uv, front = cam.project(W)
        Cimg = conic_image(circle_conic(cx, cy, r), H)
        d = sampson_conic_distance(Cimg, uv[front])
        assert d.max() < 1e-6
        # pixel-metric property: shifting 2 px along the conic normal must
        # yield a Sampson distance of ~2 px (first-order geometric distance).
        p = np.hstack([uv[front], np.ones((front.sum(), 1))])
        grad = 2.0 * (p @ Cimg.T)[:, :2]
        nrm = grad / np.linalg.norm(grad, axis=1, keepdims=True)
        d1 = sampson_conic_distance(Cimg, uv[front] + 2.0 * nrm)
        assert np.all(np.abs(d1 - 2.0) < 0.3), d1


def test_point_line_distance():
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    d = point_line_distance(np.array([[5.0, 3.0], [-2.0, -4.0]]), a, b)
    assert np.allclose(d, [3.0, 4.0])


def synth_observations(cam_gt, kp_noise=1.5, n_kp=14):
    ii = rng.choice(len(LINES_WORLD), n_kp)
    tt = rng.uniform(0, 1, n_kp)
    X = np.array([LINES_WORLD[i][0] * (1 - t) + LINES_WORLD[i][1] * t
                  for i, t in zip(ii, tt)])
    uv, front = cam_gt.project(X)
    uv = uv + rng.normal(0, kp_noise, uv.shape)
    kp_obs = (np.arange(n_kp)[front], uv[front], np.ones(front.sum()), X[front])
    line_obs = []
    for li in [13, 14, 15, 17]:            # middle line + side lines (1-based)
        A, B = LINES_WORLD[li - 1]
        prj, fr = cam_gt.project(np.stack([A, B]))
        if fr.all():
            line_obs.append((li, prj + rng.normal(0, kp_noise, prj.shape), 0.9))
    circle_obs = []
    cx, cy, r = CIRCLES_WORLD[0]
    a = rng.uniform(0, 2 * np.pi, 24)
    W = np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.zeros_like(a)], 1)
    cuv, fr = cam_gt.project(W)
    circle_obs.append((0, cuv[fr] + rng.normal(0, kp_noise, cuv[fr].shape),
                       np.full(fr.sum(), 0.9)))
    return kp_obs, line_obs, circle_obs


def test_refinement_recovers_perturbed_camera():
    cam_gt = make_cam(0.0)
    kp_obs, line_obs, circle_obs = synth_observations(cam_gt)
    cam0 = Camera(cam_gt.f * 1.06, cam_gt.pan + 0.02, cam_gt.tilt - 0.015,
                  cam_gt.roll + 0.01, cam_gt.C + np.array([1.5, -2.0, 0.8]),
                  image_wh=(1920, 1080))
    cfg = RefineCfg()
    cam1, cost = refine_camera(cam0, kp_obs, line_obs, circle_obs, cfg)
    assert cam1 is not None, "refinement abstained on a well-posed problem"
    # projection agreement with GT on a grid of ground points
    G = np.column_stack([rng.uniform(-45, 45, 60), rng.uniform(-30, 30, 60),
                         np.zeros(60)])
    uv_gt, f1 = cam_gt.project(G)
    uv_0, _ = cam0.project(G)
    uv_1, _ = cam1.project(G)
    e0 = np.linalg.norm(uv_0[f1] - uv_gt[f1], axis=1).mean()
    e1 = np.linalg.norm(uv_1[f1] - uv_gt[f1], axis=1).mean()
    assert e1 < e0 * 0.35, (e0, e1)
    assert e1 < 6.0, e1


def test_conic_term_helps_when_keypoints_sparse():
    cam_gt = make_cam(0.0)
    kp_obs, line_obs, circle_obs = synth_observations(cam_gt, n_kp=5)
    cam0 = Camera(cam_gt.f * 1.08, cam_gt.pan + 0.025, cam_gt.tilt - 0.02,
                  cam_gt.roll, cam_gt.C + np.array([2.0, -2.5, 1.0]),
                  image_wh=(1920, 1080))
    G = np.column_stack([rng.uniform(-45, 45, 60), rng.uniform(-30, 30, 60),
                         np.zeros(60)])
    uv_gt, fr = cam_gt.project(G)

    def err(cam):
        uv, _ = cam.project(G)
        return np.linalg.norm(uv[fr] - uv_gt[fr], axis=1).mean()

    cfg_no = RefineCfg(use_conic_term=False)
    cfg_yes = RefineCfg(use_conic_term=True)
    cam_no, _ = refine_camera(cam0, kp_obs, line_obs, [], cfg_no)
    cam_yes, _ = refine_camera(cam0, kp_obs, line_obs, circle_obs, cfg_yes)
    # refine_camera returns None when it abstains (implausible solution);
    # the conic variant must not be the one that fails.
    assert cam_yes is not None, "conic variant abstained where no-conic did not"
    if cam_no is None:
        print("sparse-kp err: no-conic ABSTAINED, conic %.2f px" % err(cam_yes))
        return
    assert err(cam_yes) <= err(cam_no) * 1.05   # never worse, usually better
    print("sparse-kp err: no-conic %.2f px, conic %.2f px" %
          (err(cam_no), err(cam_yes)))


def test_tripod_estimation():
    T_gt = np.array([0.0, -56.0, -13.0])
    cams = []
    for _ in range(40):
        pan = rng.uniform(-0.4, 0.4); tilt = rng.uniform(1.15, 1.35)
        R = ptr_to_R(pan, tilt, 0.0)
        r3 = R[2, :]
        n = np.cross(r3, [0.0, 0.0, 1.0]); n /= np.linalg.norm(n)
        # camera sits 0.6 m behind T along the axis, offset delta=0.35 m
        # perpendicular to it (delta is the axis-to-T distance).
        C = T_gt - 0.6 * r3 + 0.35 * n + rng.normal(0, 0.02, 3)
        cams.append(Camera(3000, pan, tilt, 0.0, C, image_wh=(1920, 1080)))
    T, delta = estimate_tripod(cams)
    # T and a systematic perpendicular offset are only weakly separable (the
    # reason BroadTrack estimates delta jointly); what the tracker needs is
    # that (T, delta) fits every camera axis, and that T lands near T_gt.
    assert np.linalg.norm(T - T_gt) < 0.6, (T, T_gt)
    resid = []
    for cam in cams:
        r3 = cam.R[2, :]
        Ostar = cam.C + np.dot(T - cam.C, r3) * r3
        resid.append(abs(np.linalg.norm(Ostar - T) - delta))
    assert np.mean(resid) < 0.15, np.mean(resid)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
