"""Camera model and field geometry.

Camera model follows BroadTrack (paper Sec. 3.1): pinhole with a single focal
length f, principal point fixed at the image center, pose = focal point C and
rotation R = Rz(roll) @ Rx(tilt) @ Rz(pan) (ZXZ Euler), plus one Brown-Conrady
radial term k1 applied in normalized coordinates:
    x_px = f * L(r) * x_n + pp,   L(r) = 1 + k1 r^2.

World frame follows PnLCalib's inference code (verified from the repo):
pitch-centered coordinates, x in [-52.5, 52.5], y in [-34, 34] (105 x 68 m),
z <= 0 pointing up (crossbar at z = -2.44).
"""
from __future__ import annotations
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0
CIRCLE_R = 9.15

# 23 straight field lines, copied verbatim from PnLCalib/inference.py
# (coordinates there are un-centered 0..105 / 0..68; we center them below).
_LINES_RAW = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]], [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]],
    [[0., 30.34, 0.], [0., 30.34, -2.44]], [[105., 37.66, -2.44], [105., 30.34, -2.44]],
    [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
    [[52.5, 0., 0.], [52.5, 68., 0.]], [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]], [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]], [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]], [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]], [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]],
]


def _center(p):
    return np.array([p[0] - PITCH_L / 2.0, p[1] - PITCH_W / 2.0, p[2]], float)


LINES_WORLD = [(_center(a), _center(b)) for a, b in _LINES_RAW]

# Circles in the Z=0 plane, pitch-centered: (cx, cy, radius).
CIRCLES_WORLD = [
    (0.0, 0.0, CIRCLE_R),                       # center circle
    (11.0 - PITCH_L / 2.0, 0.0, CIRCLE_R),      # left penalty arc
    (94.0 - PITCH_L / 2.0, 0.0, CIRCLE_R),      # right penalty arc
]


# ----------------------------------------------------------------------------- rotations
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def ptr_to_R(pan, tilt, roll):
    """R = Rz(roll) @ Rx(tilt) @ Rz(pan) — BroadTrack Euler convention."""
    return rot_z(roll) @ rot_x(tilt) @ rot_z(pan)


def R_to_ptr(R):
    """Inverse of ptr_to_R (ZXZ). Verified against ptr_to_R in unit tests."""
    tilt = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    if abs(np.sin(tilt)) < 1e-8:            # degenerate: pan/roll coupled
        pan = np.arctan2(R[1, 0], R[0, 0])
        return pan, tilt, 0.0
    pan = np.arctan2(R[2, 0], R[2, 1])
    roll = np.arctan2(R[0, 2], -R[1, 2])
    return pan, tilt, roll


# ----------------------------------------------------------------------------- camera
class Camera:
    """kappa = {f, k1, pan, tilt, roll, Cx, Cy, Cz}; pp fixed at image center."""

    PARAM_NAMES = ["pan", "tilt", "roll", "f", "cx", "cy", "cz", "k1"]

    def __init__(self, f, pan, tilt, roll, C, k1=0.0, image_wh=(1920, 1080)):
        self.f = float(f)
        self.k1 = float(k1)
        self.pan, self.tilt, self.roll = float(pan), float(tilt), float(roll)
        self.C = np.asarray(C, float)
        self.w, self.h = image_wh
        self.pp = np.array([self.w / 2.0, self.h / 2.0])

    # -- vector interface for the optimizer ----------------------------------
    def to_vec(self, optimize_k1=True):
        v = [self.pan, self.tilt, self.roll, self.f, *self.C]
        if optimize_k1:
            v.append(self.k1)
        return np.array(v, float)

    @classmethod
    def from_vec(cls, v, image_wh, k1_fixed=0.0):
        k1 = v[7] if len(v) > 7 else k1_fixed
        return cls(f=v[3], pan=v[0], tilt=v[1], roll=v[2], C=v[4:7], k1=k1,
                   image_wh=image_wh)

    @classmethod
    def from_pnlcalib(cls, final_params_dict, image_wh):
        """Build from PnLCalib's `heuristic_voting` output (verified schema:
        cam_params: x_focal_length, y_focal_length, principal_point,
        position_meters, rotation_matrix). PnLCalib has no k1 -> k1 = 0."""
        cp = final_params_dict["cam_params"]
        f = 0.5 * (float(cp["x_focal_length"]) + float(cp["y_focal_length"]))
        R = np.asarray(cp["rotation_matrix"], float)
        pan, tilt, roll = R_to_ptr(R)
        cam = cls(f=f, pan=pan, tilt=tilt, roll=roll,
                  C=np.asarray(cp["position_meters"], float),
                  k1=0.0, image_wh=image_wh)
        pp = cp.get("principal_point")
        if pp is not None:                    # honour PnLCalib's pp exactly
            cam.pp = np.asarray(pp, float).ravel()[:2]
        return cam

    # -- projection ----------------------------------------------------------
    @property
    def R(self):
        return ptr_to_R(self.pan, self.tilt, self.roll)

    def project_full(self, X):
        """X: (N,3) -> (uv(N,2), front(N,), z(N,)) with camera-frame depth z.
        Depth is needed for the cheirality penalty: residuals that simply drop
        behind-camera points create a ZERO-cost degenerate optimum (push the
        whole pitch behind the camera and every term vanishes)."""
        X = np.atleast_2d(np.asarray(X, float))
        Xc = (self.R @ (X - self.C).T).T
        z = Xc[:, 2]
        front = z > 1e-6
        zs = np.where(front, z, 1.0)
        xn = Xc[:, 0] / zs
        yn = Xc[:, 1] / zs
        r2 = xn * xn + yn * yn
        L = 1.0 + self.k1 * r2
        u = self.f * L * xn + self.pp[0]
        v = self.f * L * yn + self.pp[1]
        return np.stack([u, v], axis=1), front, z

    def project(self, X):
        """X: (N,3) world points -> (N,2) pixels, (N,) in-front mask."""
        X = np.atleast_2d(np.asarray(X, float))
        Xc = (self.R @ (X - self.C).T).T
        z = Xc[:, 2]
        front = z > 1e-6
        zs = np.where(front, z, 1.0)
        xn = Xc[:, 0] / zs
        yn = Xc[:, 1] / zs
        r2 = xn * xn + yn * yn
        L = 1.0 + self.k1 * r2
        u = self.f * L * xn + self.pp[0]
        v = self.f * L * yn + self.pp[1]
        return np.stack([u, v], axis=1), front

    def backproject_ground(self, uv, iters=5):
        """Pixels -> Z=0 plane world points (inverts k1 by fixed-point iters)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        xd = (uv[:, 0] - self.pp[0]) / self.f
        yd = (uv[:, 1] - self.pp[1]) / self.f
        # Invert rd = ru (1 + k1 ru^2) with Newton on the radius (robust for
        # strong barrel distortion where naive fixed-point iteration diverges).
        rd = np.hypot(xd, yd)
        ru = rd.copy()
        for _ in range(max(iters, 10)):
            f = ru + self.k1 * ru ** 3 - rd
            fp = 1.0 + 3.0 * self.k1 * ru ** 2
            ru = ru - f / np.where(np.abs(fp) < 1e-9, 1e-9, fp)
        scale = np.where(rd > 1e-12, ru / np.maximum(rd, 1e-12), 1.0)
        xu, yu = xd * scale, yd * scale
        # For barrel distortion (k1 < 0) the map rd(ru) is only monotonic up
        # to ru* = 1/sqrt(-3 k1); beyond it inversion is ill-posed (far
        # outside any realistic FOV) -> mark invalid.
        if self.k1 < 0:
            ru_star = 1.0 / np.sqrt(-3.0 * self.k1)
            rd_max = ru_star + self.k1 * ru_star ** 3
            invertible = rd < 0.999 * rd_max
        else:
            invertible = np.ones_like(rd, bool)
        d = np.stack([xu, yu, np.ones_like(xu)], axis=1)     # rays, camera frame
        dw = (self.R.T @ d.T).T                              # world direction
        t = -self.C[2] / np.where(np.abs(dw[:, 2]) < 1e-9, 1e-9, dw[:, 2])
        X = self.C[None, :] + t[:, None] * dw
        valid = (t > 0) & np.isfinite(t) & invertible
        return X, valid

    def homography_ground(self):
        """3x3 map from ground plane (X, Y, 1) to image, valid when k1 == 0.
        Used only to transform conics; the conic residual re-applies distortion
        implicitly by evaluating on *undistorted* sample points (see refine)."""
        K = np.array([[self.f, 0, self.pp[0]], [0, self.f, self.pp[1]], [0, 0, 1.0]])
        Rt = np.hstack([self.R, (-self.R @ self.C).reshape(3, 1)])
        P = K @ Rt
        return P[:, [0, 1, 3]]


# ----------------------------------------------------------------------------- parameter bounds
# Physically meaningful limits for a broadcast camera. Order matches
# Camera.to_vec: [pan, tilt, roll, f, cx, cy, cz, (k1)].
# z is negative upwards in this world frame, so cz in [-120, -1] keeps the
# camera above the pitch surface.
PARAM_LO = np.array([-2*np.pi, 0.05, -np.pi/3, 100.0, -400.0, -400.0, -120.0, -0.6])
PARAM_HI = np.array([ 2*np.pi, np.pi-0.05,  np.pi/3, 30000.0, 400.0, 400.0, -1.0, 0.6])


def param_bounds(n):
    return PARAM_LO[:n].copy(), PARAM_HI[:n].copy()


def clip_to_bounds(v):
    n = len(v)
    lo, hi = param_bounds(n)
    v = np.asarray(v, float).copy()
    v[~np.isfinite(v)] = np.clip(0.0, lo, hi)[~np.isfinite(v)]
    return np.clip(v, lo + 1e-9, hi - 1e-9)


def camera_is_plausible(cam, image_wh=None):
    """Reject cameras that cannot correspond to a real broadcast view.

    Without this, a frame with degenerate observations can drive the optimizer
    to |C| ~ 1e11 m and f ~ 1e14 - values that then poison every aggregate
    (the 1e108 reprojection errors seen on real data). Abstaining is correct:
    a wrong homography is worse than none for GSR.
    """
    if cam is None:
        return False
    vals = np.array([cam.f, cam.k1, *cam.C, cam.pan, cam.tilt, cam.roll], float)
    if not np.all(np.isfinite(vals)):
        return False
    if not (100.0 <= cam.f <= 30000.0):
        return False
    if not (-0.6 <= cam.k1 <= 0.6):
        return False
    if np.linalg.norm(cam.C[:2]) > 400.0:
        return False
    if not (-120.0 <= cam.C[2] <= -1.0):      # above the pitch, sane height
        return False
    # the camera must actually see a decent part of the pitch
    G = np.array([[0, 0, 0], [-52.5, -34, 0], [52.5, -34, 0],
                  [52.5, 34, 0], [-52.5, 34, 0]], float)
    uv, front = cam.project(G)
    if front.sum() == 0 or not np.isfinite(uv[front]).all():
        return False
    return True


# ----------------------------------------------------------------------------- residual helpers
def circle_conic(cx, cy, r):
    """Homogeneous conic matrix of (x-cx)^2 + (y-cy)^2 = r^2 (Broadcast2Pitch Eq. 3)."""
    return np.array([[1.0, 0.0, -cx],
                     [0.0, 1.0, -cy],
                     [-cx, -cy, cx * cx + cy * cy - r * r]])


def conic_image(Cw, H):
    """C' = H^{-T} C H^{-1} (conic transform under the ground homography)."""
    Hinv = np.linalg.inv(H)
    return Hinv.T @ Cw @ Hinv


def sampson_conic_distance(Cimg, uv):
    """First-order geometric (pixel) distance of points to a conic:
    |p^T C p| / ||grad||, grad = 2 (C p)[:2]. Commensurate with pixel residuals,
    which is the reconciliation required to mix Broadcast2Pitch's algebraic
    conic residual into PnLCalib's geometric cost (report Sec. 3, Upgrade 2)."""
    uv = np.atleast_2d(uv)
    p = np.hstack([uv, np.ones((len(uv), 1))])
    Cp = p @ Cimg.T
    alg = np.sum(p * Cp, axis=1)
    grad = 2.0 * Cp[:, :2]
    gn = np.linalg.norm(grad, axis=1)
    return np.abs(alg) / np.maximum(gn, 1e-9)


def point_line_distance(p, a, b):
    """Distance of points p (N,2) to the infinite line through a, b (PnLCalib Eq. 5)."""
    p = np.atleast_2d(p)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.linalg.norm(p - a[None, :], axis=1)
    cross = d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])
    return np.abs(cross) / n
