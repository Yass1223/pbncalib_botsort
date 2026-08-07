"""Central configuration. Every experiment is a Config, never a code edit."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Paths:
    # Kaggle defaults; override locally as needed.
    work: Path = Path("/kaggle/working")
    pnlcalib_repo: Path = Path("/kaggle/working/PnLCalib")
    weights_dir: Path = Path("/kaggle/working/weights")
    data_root: Path = Path("/kaggle/input")          # attached Kaggle datasets
    gsr_root: Path = Path("/kaggle/tmp/SoccerNetGS")   # scratch (~60 GiB); working is capped at 20 GiB
    cache_dir: Path = Path("/kaggle/working/det_cache")
    results_dir: Path = Path("/kaggle/working/results")
    labels_dir: Path = Path("/kaggle/working/labels")   # harvested annotations (persisted)

    def ensure(self):
        for p in [self.weights_dir, self.cache_dir, self.results_dir,
                  self.labels_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class DetectorCfg:
    # Verified defaults from PnLCalib/inference.py (repo main, checked 2026-08-02).
    kp_threshold: float = 0.3434
    line_threshold: float = 0.7867
    input_wh: tuple = (960, 540)     # network input; detections are normalized [0,1]
    weights_kp: str = "SV_kp"        # release v1.0.0 asset names
    weights_line: str = "SV_lines"
    device: str = "cuda:0"


@dataclass
class RefineCfg:
    """Upgraded PnLCalib refinement (Layer 2)."""
    use_confidence_weights: bool = True     # Broadcast2Pitch-style weighting
    use_conic_term: bool = True             # Sampson-normalized circle constraint
    alpha_lines: float = 0.6                # PnLCalib paper: alpha = 0.6
    beta_conic: float = 0.15                # tuned on VALID split only
    line_gate_px: float = 30.0              # projection-error gate before a line enters L (PnLCalib Sec. III)
    conic_gate_m: float = 0.75              # world-radius tolerance for circle-membership of keypoints
    inlier_k: float = 3.0              # MAD multiplier for outlier rejection
    inlier_min_px: float = 12.0        # never tighter than the detector noise floor
    trust_region_px: float = 30.0      # floor on the allowed move from the seed
    trust_region_scale: float = 5.0    # ... scaled by the seed's own fit quality
    max_nfev: int = 60
    robust_loss: str = "cauchy"             # BroadTrack uses Cauchy loss
    robust_fscale: float = 4.0              # pixels


@dataclass
class TrackerCfg:
    """BroadTrack-style temporal layer (Layer 3)."""
    enabled: bool = True
    optimize_k1: bool = True                # biggest single gain in BroadTrack Table 3
    use_optical_flow: bool = True
    use_tripod: bool = False                # enable after estimate_tripod() has run
    omega_tripod: float = 5.0               # weight of the tripod loss term
    tripod_T: tuple = (0.0, 55.0, -12.0)    # BroadTrack default main-camera focal point
    tripod_delta: float = 0.0
    reinit_below_s: float = 0.5             # Jaccard confidence threshold (BroadTrack)
    of_max_points: int = 150
    of_min_points: int = 12
    mask_players: bool = True               # discard flow near player boxes if provided
    jaccard_dilate_px: int = 7
    min_support: int = 14              # min weighted cue count before a per-frame update


@dataclass
class V2Cfg:
    """Method v2: fixed optical centre, 4 DOF per frame, temporal smoothing."""
    native_refine_lines: bool = True    # let PnLCalib do its own PnL refinement
    fix_center: bool = True             # ONE centre per sequence, held fixed
    refit_4dof: bool = True             # solve (pan, tilt, roll, f) only
    optimize_k1: bool = False           # extra DOF: off by default (v1 showed it hurts)
    use_flow: bool = False              # rotation-regime optical flow
    smooth: bool = True                 # robust Savitzky-Golay over the trajectory
    interpolate: bool = True            # fill frames where calibration failed
    window: int = 15
    polyorder: int = 2


@dataclass
class V3Cfg:
    """Method v3: parity, shared centre + bundle, so(3) smoothing, BIC."""
    parity_check: bool = True
    shared_center: bool = True
    bundle_adjust: bool = True
    sigma_center: float = 0.3          # Gaussian core of the centre prior (m)
    smooth: bool = True                # so(3) rotations + log-focal
    interpolate: bool = True
    window: int = 9
    bic_envelope: bool = True          # frame-level statistical gate
    bic_margin: float = 2.0
    subpixel: bool = True              # detection-time refinement
    mahalanobis_repair: bool = True


@dataclass
class SmoothCfg:
    """Temporal smoothing of PnLCalib output — the only thing this method does."""
    enabled: bool = True
    native_refine_lines: bool = True
    select_variant: bool = True   # choose refine_lines PER FRAME by self-fit
    variant_margin: float = 1.10
    window: int = 9
    polyorder: int = 2
    # Cap on how far a frame may move from PnLCalib's own estimate.
    # MEASURED on SoccerNet-GSR (4 sequences, 3000 frames): capping costs a
    # great deal of smoothness and buys no accuracy — jitter_f 68 uncapped vs
    # 255 at a 32 px cap, while medre is flat (161.6 vs 161.6). Default is
    # therefore uncapped; set a finite value only if a sequence shows
    # systematic (non-independent) per-frame errors.
    max_dev_px: float = float("inf")
    interpolate: bool = True    # fill frames PnLCalib could not solve
    reject_k: float = 4.0


@dataclass
class ExperimentCfg:
    name: str = "full"
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    refine: RefineCfg = field(default_factory=RefineCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    v2: V2Cfg = field(default_factory=V2Cfg)
    v3: V3Cfg = field(default_factory=V3Cfg)
    smooth: SmoothCfg = field(default_factory=SmoothCfg)
    method: str = "v1"                  # "v1" | "v2" | "v3" | "smooth"
    paths: Paths = field(default_factory=Paths)
    split: str = "test"
    seed: int = 0

    def dump(self, path):
        Path(path).write_text(json.dumps(asdict(self), default=str, indent=2))


def ablation_grid():
    """Mirrors BroadTrack Table 3 structure, extended with our two upgrades."""
    grid = []
    def mk(name, **kw):
        c = ExperimentCfg(name=name)
        for k, v in kw.items():
            obj, attr = k.split(".")
            setattr(getattr(c, obj), attr, v)
        return c
    grid.append(mk("A0_pnlcalib_baseline",
                   **{"refine.use_confidence_weights": False,
                      "refine.use_conic_term": False,
                      "tracker.enabled": False}))
    grid.append(mk("A1_conf_weights",
                   **{"refine.use_conic_term": False, "tracker.enabled": False}))
    grid.append(mk("A2_conf_plus_conic", **{"tracker.enabled": False}))
    grid.append(mk("A3_track_no_k1_no_of_no_tripod",
                   **{"tracker.optimize_k1": False,
                      "tracker.use_optical_flow": False,
                      "tracker.use_tripod": False}))
    grid.append(mk("A4_track_plus_k1",
                   **{"tracker.use_optical_flow": False, "tracker.use_tripod": False}))
    grid.append(mk("A5_track_plus_of", **{"tracker.use_tripod": False}))
    grid.append(mk("A6_full", **{"tracker.use_tripod": True}))
    return grid


def b_grid():
    """Method-v2 ablation. Each step adds ONE mechanism, so any change is
    attributable; B0 is raw PnLCalib (native refinement only), the reference
    the v1 grid never actually measured."""
    grid = []

    def mk(name, **kw):
        c = ExperimentCfg(name=name)
        c.method = "v2"
        c.tracker.enabled = False
        for k, v in kw.items():
            obj, attr = k.split(".")
            setattr(getattr(c, obj), attr, v)
        return c

    grid.append(mk("B0_pnlcalib_native",
                   **{"v2.fix_center": False, "v2.refit_4dof": False,
                      "v2.smooth": False, "v2.interpolate": False}))
    grid.append(mk("B1_fixed_center",
                   **{"v2.smooth": False, "v2.interpolate": False}))
    grid.append(mk("B2_smooth", **{"v2.interpolate": False}))
    grid.append(mk("B3_smooth_interp"))
    grid.append(mk("B4_flow", **{"v2.use_flow": True}))
    grid.append(mk("B5_flow_k1", **{"v2.use_flow": True,
                                    "v2.optimize_k1": True}))
    return grid


def c_grid():
    """Method-v3 ablation: one mechanism at a time over the B0 reference."""
    grid = []

    def mk(name, **kw):
        c = ExperimentCfg(name=name)
        c.method = "v3"
        c.tracker.enabled = False
        for k, v in kw.items():
            obj, attr = k.split(".")
            setattr(getattr(c, obj), attr, v)
        return c

    grid.append(mk("C0_baseline", **{
        "v3.parity_check": False, "v3.shared_center": False,
        "v3.bundle_adjust": False, "v3.smooth": False,
        "v3.interpolate": False, "v3.bic_envelope": False}))
    grid.append(mk("C1_parity", **{
        "v3.shared_center": False, "v3.bundle_adjust": False,
        "v3.smooth": False, "v3.interpolate": False,
        "v3.bic_envelope": False}))
    grid.append(mk("C2_shared_center", **{
        "v3.bundle_adjust": False, "v3.smooth": False,
        "v3.interpolate": False, "v3.bic_envelope": False}))
    grid.append(mk("C3_bundle", **{
        "v3.smooth": False, "v3.interpolate": False,
        "v3.bic_envelope": False}))
    grid.append(mk("C4_so3_smooth", **{"v3.bic_envelope": False}))
    grid.append(mk("C5_full_bic"))
    return grid


def s_grid():
    """Four configurations that separate the two effects.

    S0 is PnLCalib as normally used (fixed refine_lines=True). S1 adds only
    per-frame variant selection, so the variant gain is measured alone. S2 adds
    only smoothing. S3 is both — the proposal.
    """
    grid = []

    def mk(name, **kw):
        c = ExperimentCfg(name=name)
        c.method = "smooth"
        c.tracker.enabled = False
        for k, v in kw.items():
            setattr(c.smooth, k, v)
        return c

    grid.append(mk("S0_pnlcalib", enabled=False, select_variant=False))
    grid.append(mk("S1_variant", enabled=False, select_variant=True))
    grid.append(mk("S2_smooth", enabled=True, select_variant=False))
    grid.append(mk("S3_variant_smooth", enabled=True, select_variant=True))
    # reference points on the accuracy/smoothness cap
    grid.append(mk("S3_cap8px", enabled=True, select_variant=True,
                   max_dev_px=8.0))
    return grid
