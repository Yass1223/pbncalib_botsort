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

## Entry point

```bash
tracklab -cn soccernet
```

One config, one path. The BroadTrack arm was removed once the comparison was
settled — see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) section 7 for the
final numbers and how to recover it from history.

## Setup

```bash
bash scripts/setup_pnlcalib.sh    # clone PnLCalib + v1.0.0 SV_kp / SV_lines
bash scripts/setup_jn_gsr.sh      # jersey-number venv + checkpoints
```

**Both are mandatory.** The jersey stage now *raises* when unprovisioned rather
than logging an error and continuing. Earlier runs scored GS-HOTA with
`jersey_number` 0/12996 non-null and `tracklet_agg` voting over `{None: 0.0}` —
a metric missing a component is not a worse measurement, it is a different one
wearing the same name. Override deliberately with
`modules.jersey_number_detect.cfg.allow_unprovisioned=true`.

PnLCalib is GPL-2.0 and is **cloned at runtime, never vendored**. Weights,
SoccerNet data, calibration caches and the clone are all gitignored.

`uv.lock` is generated on Kaggle from the pinned `pyproject.toml` and committed
from there — it cannot be produced on a machine without `uv`. The two git
dependencies are now pinned to the commits the first successful run resolved to
(`docs/venv_freeze_2026-08-07.txt` is that environment). Until the lock lands,
resolve fresh:

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

## Known limitations

Seven entries in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), each with what it
does and does not invalidate — the image-space spatial gate, the inert `track`
batch size, an upstream tautology in the GMC guard, the evaluator dropping
detections with a null `bbox_pitch`, a disproved EMA concern, the unpinned
dependency tree, and the BroadTrack removal.

Measured closures and what is still open live in
[docs/AUDIT_FINDINGS.md](docs/AUDIT_FINDINGS.md), with every line marked
read-from-artifact or read-from-source.

## Instrumentation rules

Three, all learned the hard way:

0. **Any check that can state a precondition must ENFORCE it, not report it.**
   If a message tells the human what to fix, the code could have gated on it.
   Warnings are for conditions with no mechanical remedy; everything else raises.

   This was violated four times before it was written down, and each cost a run:
   `preflight_imports.py` detected the matplotlib backend failure and printed
   `export MPLBACKEND=Agg` instead of setting it; `setup_env.py` printed
   `strhub (PARSeq): BROKEN` and exited 0, leaving a venv that died at the jersey
   stage; the same file warned `str/parseq not found` and built the rest of the
   venv anyway; and the notebook printed `MANDATORY SEQUENCE(S) NOT PRESENT` and
   proceeded to audit only the easy sequences. All four now raise.

   The test is simple: if the message contains an imperative -- "run", "set",
   "attach", "must contain" -- the check knew the precondition and declined to
   enforce it. `s`-distribution warnings and the out-of-bounds counter stay
   warnings, because there is no mechanical remedy for "this camera looks stale";
   that is a judgement for a human.

Then:

1. **Never instantiate an `*Api` class from instrumentation.** `GTALink(cfg, ...)`
   crashes outside hydra — the yaml carries `${...}` interpolations and the custom
   `${hf:}` resolver, neither of which resolves in a bare `OmegaConf.load`.
   Instrumentation reads *artifacts* and builds models *directly*.
2. **A probe that cannot see its target reports UNRESOLVED, never a verdict.**
   `fork_probe.json` once reported `F4 = SILENT` because it grepped a dispatcher
   instead of the implementation, and `F7 = MIXED` because it grepped for symbol
   *presence* when the question was whether the call site is taken. Absence of
   evidence rendered as evidence of absence is the worst failure an instrument can
   have, because it looks like an answer.
