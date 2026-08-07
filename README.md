# pbncalib_botsort — SoccerNet Game State Reconstruction

TrackLab / sn-gamestate pipeline with **Option A field registration**: PnLCalib
per-frame calibration plus temporal smoothing, replacing the BroadTrack
calibration stage. BroadTrack is kept intact as an A/B fallback.

```
bbox_detector  YOLO11-L SoccerNet fine-tune (HF)            -> bbox_ltwh, bbox_conf
reid           prtreid (bpbreid/HRNet32, 256-d)             -> embeddings, role
track          BoT-SORT · SOF + sports-OSNet                -> track_id
gta_link       offline tracklet stitching                   -> track_id (merged)
calibration    PnLCalib + so(3)/log-focal smoothing         -> parameters, bbox_pitch
jersey_number  jn_pipeline_gsr (legibility -> DBNet++ -> PARSeq)
tracklet_agg   majority vote over [jersey_number, role]
team           k-means on prtreid embeddings
team_side      mean pitch position
```

There is **no separate `pitch` stage**: PnLCalib runs its own keypoint (57+1) and
line (23+1) HRNet-w48 detectors and emits camera parameters directly.

## Entry points

| Config | Calibration stage |
|---|---|
| `tracklab -cn soccernet_optiona` | Option A (PnLCalib + smoothing) |
| `tracklab -cn soccernet` | BroadTrack (the A/B baseline) |

## Setup

```bash
bash scripts/setup_pnlcalib.sh    # clone PnLCalib + v1.0.0 SV_kp / SV_lines
bash scripts/setup_jn_gsr.sh      # jersey-number venv + checkpoints
```

PnLCalib is GPL-2.0 and is **cloned at runtime, never vendored**. Weights,
SoccerNet data, calibration caches and the clone are all gitignored.

`uv.lock` is deliberately **not** committed: `shapely`, `matplotlib` and `scipy`
became direct dependencies for the calibration stage and the previous lock
predates them, so a stale `uv sync` would install an environment in which the
stage fails at import. Resolve fresh:

```bash
uv venv --python 3.9 .venv && uv pip install --python .venv -e .
```

The environment is Python 3.9 with **torch 1.13.1** — required by the detector,
reid and tracking stages. PnLCalib's own `requirements.txt` asks for torch 2.3.1;
it is not used, and the pipeline's pins must not be bumped to suit it. Whether
PnLCalib runs correctly on 1.13.1 is answered by `scripts/verify_pnlcalib_env.py`.

## Verification

Run these before trusting any metric.

```bash
python scripts/verify_optiona_conversion.py    # local; numpy + OpenCV + scipy only
python scripts/preflight_imports.py            # every stage imports
python tests/test_optiona_api_contract.py      # engine contract, no torch needed
python scripts/verify_pnlcalib_env.py --repo ... --weights-kp ... --frame ...
```

`verify_optiona_conversion.py` checks the Option A → sn-calibration camera
conversion numerically, and carries a **negative control** (copying pan/tilt/roll
across, which the reversed ZXZ convention makes wrong). A parity test whose
control passes is measuring nothing, so both numbers are always reported.

## Reading the output

**Completeness is not evidence.** It is ~1.0 by construction on this stage:
smoothing interpolates gaps, carry-forward fills the rest, and the tracker layer
never returns `None` after lock-on. A camera that locks on at frame 1, drifts,
and is carried for the remaining 749 frames produces zero errors, 100%
completeness, and wrong pitch coordinates on every frame.

What to read instead, all logged per sequence:

- **`s` distribution** — min / median / max and the first lock-on frame. `s` is
  the detection-precision confidence `|D ∩ T| / |D|`, computed post-smoothing on
  the camera actually emitted. It is the only signal that separates a live camera
  from a stale one.
- **out-of-bounds count** — projected positions outside |x| ≤ 60 / |y| ≤ 45 m.
  A *lower bound* on the cheirality problem, not a measurement: a behind-camera
  unprojection can land inside those bounds by coincidence. `0%` means "none
  caught", not "none present".
- **camera parameter ranges** — focal length should not drift, and the optical
  centre should not wander off the touchline.
- **GS-HOTA A/B** against the BroadTrack run on identical sequences.

## Layout

```
sn_gamestate/        pipeline modules and Hydra configs
  calibration/optiona_api.py       the VideoLevelModule
  calibration/optiona_convert.py   pure-geometry conversion (numpy only)
optiona_sfr/         vendored Option A package (pure Python)
plugins/             sn_calibration_baseline and friends
scripts/             setup, verification and evaluation
tests/               contract and unit tests
notebooks/           Kaggle pipeline notebook
```
