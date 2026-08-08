import ast, json, pathlib, sys, tokenize, io

OUT = pathlib.Path(sys.argv[1])

_CODE_CELL_N = 0


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip().splitlines(keepends=True)}


def _check_cell(body):
    """Reject a broken code cell AT GENERATION TIME, not after push.

    The recurring bug this guards: a nested ``\\n`` inside the generator's
    triple-quoted cell strings collapses into a real newline and splits a line
    mid-string, so the emitted cell is syntactically broken. It was repaired
    reactively three times this session. This makes the generator refuse to
    build such a cell in the first place.

    Two independent checks, because they catch different failures:
      1. ``ast.parse`` -- the cell is valid Python as a whole.
      2. ``tokenize`` -- catches an unterminated string literal even in the
         cases ast reports with a confusing downstream error, and pins the exact
         line. A string split across a newline tokenizes to a TokenError here.
    """
    global _CODE_CELL_N
    _CODE_CELL_N += 1
    idx = _CODE_CELL_N
    # 1. tokenizer: unbalanced / unterminated string literals, exact line
    try:
        list(tokenize.generate_tokens(io.StringIO(body).readline))
    except tokenize.TokenError as e:
        _die(idx, body, f"TokenError (unterminated string / bracket): {e}")
    except IndentationError as e:
        _die(idx, body, f"IndentationError: {e}")
    # 2. full parse
    try:
        ast.parse(body)
    except SyntaxError as e:
        _die(idx, body, f"SyntaxError at cell-line {e.lineno}: {e.msg}")


def _die(idx, body, reason):
    lines = body.splitlines()
    print(f"\n*** GENERATOR REFUSED to emit code cell #{idx}: {reason}",
          file=sys.stderr)
    for i, ln in enumerate(lines, 1):
        # flag any line whose double-quote count is odd (the classic split)
        mark = "  <-- odd quote count" if ln.count('"') % 2 else ""
        print(f"  {i:3d}| {ln}{mark}", file=sys.stderr)
    sys.exit(2)


def code(s):
    body = s.strip()
    _check_cell(body)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": body.splitlines(keepends=True)}


cells = []

cells.append(md(r"""
# pbncalib_botsort — Kaggle pipeline

Option A field registration (PnLCalib + temporal smoothing) inside the SoccerNet
GSR pipeline. Single arm: `tracklab -cn soccernet` IS Option A. BroadTrack was
removed once the comparison was settled (KNOWN_LIMITATIONS section 7).

**Four stages, each gating the next.** Run them in order and stop at the first
failure — a later stage's output is meaningless if an earlier one did not pass.

| Stage | What | Gates |
|---|---|---|
| 1 | build env (uv + Python 3.9) → `preflight_imports.py` | every stage imports |
| 2 | `verify_pnlcalib_env.py` | **answers the torch question** |
| 3 | shortest sequence end to end | `bbox_pitch` populated and trustworthy |
| 4 | GS-HOTA | the actual verdict |

### The environment build is not trivial

Kaggle's session Python is past 3.9 and ships torch 2.x. This pipeline needs
Python 3.9 and torch 1.13.1 — the detector, reid and tracking stages require it.
So Stage 1 installs `uv`, has it fetch a 3.9 interpreter, and builds a separate
venv: the work `preflight_cpu.sh` does elsewhere. **Expect a slow first cell.**

Do **not** `pip install -r optiona_sfr/requirements.txt` into the session. It
would try to move the session's torch and can leave CUDA mismatched against the
driver; the file's own header says so. Everything below installs into the venv
with `--python .venv`, never into the session.
"""))

cells.append(md("## Stage 1 — environment"))

cells.append(code(r"""
import os
import subprocess
from pathlib import Path

# MUST be set before anything imports matplotlib. tracklab pulls it in
# transitively and the default backend needs a display; on Kaggle that is an
# immediate ImportError at stage-import time. preflight_imports.py already names
# this fix when it fires -- this applies it.
os.environ["MPLBACKEND"] = "Agg"
# uv hardlinks into the venv by default; /kaggle/working and the uv cache are on
# different filesystems, so hardlinking fails or silently degrades. Copy instead.
os.environ["UV_LINK_MODE"] = "copy"

REPO_URL = "https://github.com/Yass1223/pbncalib_botsort.git"
PIN = "ff6028f"          # commit this notebook was written against
WORK = Path("/kaggle/working")
REPO = WORK / "pbncalib_botsort"
VENV = REPO / ".venv"


def sh(cmd, cwd=None, check=True):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise RuntimeError("failed (%d): %s" % (r.returncode, cmd))
    return r.returncode


# FULL clone, not --depth 1: a shallow clone cannot check out PIN, so the pin
# would be inert and any older commit unfetchable.
if not (REPO / "pyproject.toml").exists():
    sh("git clone %s %s" % (REPO_URL, REPO))
else:
    sh("git -C %s fetch --all --tags" % REPO, check=False)
sh("git -C %s checkout %s" % (REPO, PIN))
sh("git -C %s log --format='HEAD %%H %%s' -1" % REPO)
print("")
print("Confirm the hash above is %s before proceeding." % PIN)
"""))

cells.append(code(r"""
# uv, a Python 3.9 interpreter, and the venv.
sh("pip -q install uv", check=False)
sh("uv python install 3.9")
sh("uv venv --python 3.9 %s" % VENV, cwd=REPO)

# Resolve fresh from pyproject.toml. uv.lock is intentionally absent: shapely,
# matplotlib and scipy became direct dependencies of the calibration stage and
# the old lock predates them, so syncing it would build an environment in which
# the stage fails at import.
# GENERATE THE LOCK. This is the only place it can be produced -- uv does not
# exist on the authoring machine, so pyproject could not even be TOML-validated
# there (tomllib is 3.11+, that box is 3.9). Doing it before the install means a
# malformed pyproject fails here, loudly, instead of surfacing as a mystery
# resolution later.
sh("uv lock", cwd=REPO)
sh("uv pip install --python %s -e ." % VENV, cwd=REPO)

PY = "uv run --python %s python" % VENV
sh(PY + " -c \"import torch, numpy; print('venv torch', torch.__version__, "
        "'| numpy', numpy.__version__)\"", cwd=REPO)

# The two git dependencies are pinned to commits. Confirm the lock resolved
# THOSE commits and not a branch head -- the whole point of pinning is that this
# is checkable after the fact.
WANT = {"prtreid": "30617a75967e84d5d516959c4b84cbeea6f56493",
        "torchreid": "a2dc4304284784d8b2c061764512f0698aa38c21"}
lock = (REPO / "uv.lock").read_text() if (REPO / "uv.lock").exists() else ""
print("\nuv.lock: %d bytes" % len(lock))
for pkg, sha in WANT.items():
    hit = sha in lock
    print("  %-10s %s  %s" % (pkg, sha[:12], "RESOLVED" if hit else "*** NOT IN LOCK"))
    if not hit:
        for ln in lock.splitlines():
            if pkg in ln.lower() and ("git" in ln or "rev" in ln):
                print("      lock says:", ln.strip()[:120])
assert all(v in lock for v in WANT.values()), (
    "uv.lock did not resolve the pinned commits -- the pins are not taking "
    "effect and ReID behaviour is unrecorded again")
print("\nBoth git pins resolved as specified.")
"""))

cells.append(code(r"""
# GATE 1: every _target_ declared under configs/ must import.
rc = sh(PY + " scripts/preflight_imports.py", cwd=REPO, check=False)
assert rc == 0, ("Stage 1 failed: some pipeline stages do not import. Fix the "
                 "dependency first -- later stages cannot be interpreted if the "
                 "pipeline is only half-built.")
print("\nStage 1 PASSED")
"""))

cells.append(code(r"""
# FREEZE THE RESOLVED ENVIRONMENT, every run.
# tracklab bounds none of its 30 dependencies (KNOWN_LIMITATIONS section 6), so
# what actually got installed is only knowable after the fact. Two drift
# failures have already cost a session each. This is the record.
freeze = WORK / "venv_freeze.txt"
sh("uv pip freeze --python %s > %s" % (VENV, freeze), cwd=REPO, check=False)
try:
    txt = freeze.read_text()
    print("%d packages frozen -> %s" % (len(txt.splitlines()), freeze))
    for k in ("torch==", "numpy==", "huggingface", "prtreid", "torchreid",
              "ultralytics", "transformers"):
        for ln in txt.splitlines():
            if ln.lower().startswith(k.lower()) or k.lower() in ln.lower()[:40]:
                print("   ", ln)
                break
except Exception as e:
    print("freeze unreadable:", e)
print("\nCommit this as docs/venv_freeze_<date>.txt if the run is a keeper.")
"""))

cells.append(md(r"""
## Stage 1.5 — read the forks

The `track` / `gta_link` static audit ran against a checkout where `tracklab`,
`bot_sort` and `strong_sort` did **not exist on disk** — they ship inside the
installed tracklab package. Five findings were left UNVERIFIABLE STATICALLY for
exactly that reason. The venv has just materialised them, so settle them now,
before anything expensive runs.

| | Question | Why it matters |
|---|---|---|
| **F2** | Does the engine batch images for an `ImageLevelModule`? | `process()` takes `batch["input"][0]`; `soccernet.yaml` sets `track: {batch_size: 64}`. If honoured, **63 of every 64 frames are silently dropped** |
| **F4** | Does `GMC.apply` log on failure, and return what? | An identity warp means CMC is dead — HOTA 0.6315 vs 0.6687 |
| **F7** | Kalman state `(w,h)` or `(a,h)`? | The classic BoT-SORT/ByteTrack transplant error |
| **F8** | EMA appearance update in `STrack.update`? | Whether appearance genuinely accumulates |
| **F13** | Does the encoder drop NaN `track_id`? | GTA-Link's collision guard creates those deliberately |

**F2 and F4 gate Stage 4.** Neither changes what is being measured, but either
can depress absolute GS-HOTA far enough to make the number uninformative: a
tracker on 1/64 of the frames, or with camera-motion compensation dead.
"""))

cells.append(code(r"""
rc = sh(PY + " scripts/probe_forks.py --json /kaggle/working/fork_probe.json",
        cwd=REPO, check=False)

import json
probe = {}
try:
    probe = json.load(open("/kaggle/working/fork_probe.json"))
except Exception as e:
    print("could not read probe summary:", e)

# Record the two that gate Stage 4. Do not stop here -- Stage 3 is still
# informative even if these are bad; it is the A/B that becomes uninterpretable.
F2 = probe.get("F2", {}).get("answer", "UNRESOLVED")
F4 = probe.get("F4", {}).get("answer", "UNRESOLVED")
print("\n" + "=" * 70)
print("Stage 4 gates:  F2 (batching) = %s   |   F4 (GMC logging) = %s" % (F2, F4))
print("=" * 70)
print("Read the F2 evidence lines above and decide explicitly: if batch_size "
      "reaches a DataLoader for image-level modules, STOP and fix it -- every "
      "number after this point would be computed on 1/64 of the frames.")
"""))

cells.append(md(r"""
## Stage 2 — does PnLCalib run on torch 1.13.1?

**This is the gate the whole integration hangs on.** PnLCalib pins torch 2.3.1;
the pipeline pins 1.13.1 and cannot move. The models use only long-stable ops,
but two things cannot be settled by reading source:

- can torch 1.13.1 read a checkpoint archive written by 2.3.1? (check **D**)
- do the `state_dict` keys match the architecture from `hrnetv2_w48.yaml`? (check **E**)

If D or E fail, **do not bump the pipeline's torch.** The remedy — a subprocess
boundary, or re-serialised checkpoints — lands inside the single
`compute_cameras` function in `optiona_api.py`, which exists to contain exactly
this.
"""))

cells.append(code(r"""
# ~505 MiB of checkpoints. Resume-safe, so re-running is cheap.
# BOTH setups are mandatory. jn_gsr was never run by earlier revisions of this
# notebook, so .venv_jn could not exist and the jersey stage failed on every run
# -- jersey_number 0/12996, tracklet_agg voting over {None: 0.0}, and a GS-HOTA
# figure scoring a component that was structurally absent. The stage now RAISES
# when unprovisioned, so a missing setup stops the run instead of quietly
# changing what is being measured.
sh("bash scripts/setup_pnlcalib.sh", cwd=REPO)
sh("bash scripts/setup_jn_gsr.sh", cwd=REPO)

# The stage now RAISES when unprovisioned, so this only pre-empts a slow failure.
jn_py = REPO / "plugins/jn_gsr/.venv_jn/bin/python"
jn_worker = REPO / "plugins/jn_gsr/predict_tracklets.py"
print("jersey venv:   %s  %s" % (jn_py, "OK" if jn_py.is_file() else "MISSING"))
print("jersey worker: %s  %s" % (jn_worker, "OK" if jn_worker.is_file() else "MISSING"))
assert jn_py.is_file() and jn_worker.is_file(), (
    "jersey stage unprovisioned -- setup_jn_gsr.sh did not produce .venv_jn. "
    "Every previous run scored GS-HOTA with jersey_number 0/12996; do not "
    "repeat that.")
"""))

cells.append(code(r"""
FRAME = next(iter(sorted(f for base in (Path("/kaggle/input"), Path("/kaggle/tmp"),
                                        Path("/kaggle/working")) if base.exists()
                         for f in base.rglob("img1/000001.jpg"))), None)
print("frame:", FRAME)
assert FRAME is not None, "no SoccerNet frame found under any data root"

rc = sh(PY + " scripts/verify_pnlcalib_env.py"
        " --repo pretrained_models/pnlcalib/PnLCalib"
        " --weights-kp pretrained_models/pnlcalib/weights/SV_kp"
        " --weights-line pretrained_models/pnlcalib/weights/SV_lines"
        " --frame %s" % FRAME, cwd=REPO, check=False)
if rc != 0:
    raise SystemExit(
        "Stage 2 FAILED. Read which check failed above.\n"
        "  D (torch.load) -> 1.13.1 cannot read the 2.3.1 archive.\n"
        "  E (state_dict) -> keys/shapes disagree with hrnetv2_w48.yaml.\n"
        "Either way the fix goes inside compute_cameras() in optiona_api.py. "
        "DO NOT bump the pipeline's torch: the detector, reid and tracking "
        "stages require 1.13.1.")
print("\nStage 2 PASSED -- PnLCalib runs on the pipeline's pins")
"""))

cells.append(md(r"""
## Stage 3 — one sequence, end to end

Shortest available sequence, so a failure surfaces in minutes rather than after
a whole split. The two local gates run first: they need no GPU, and the
conversion gate carries the negative control that proves it can fail.
"""))

cells.append(code(r"""
# GATE 3a: camera-conversion parity + negative control.
rc = sh(PY + " scripts/verify_optiona_conversion.py", cwd=REPO, check=False)
assert rc == 0, "conversion parity failed -- bbox_pitch would be wrong everywhere"

# GATE 3b: engine contract (pure pandas, no torch).
rc = sh(PY + " tests/test_optiona_api_contract.py", cwd=REPO, check=False)
assert rc == 0, "engine contract broken"
"""))

cells.append(code(r"""
# Search BOTH roots. Attached Kaggle Datasets land under /kaggle/input;
# optiona_sfr.kaggle_setup.ensure_gsr_data stages to scratch (/kaggle/tmp),
# which is where a fresh download goes because /kaggle/working is capped at
# 20 GiB. Neither root alone covers both workflows.
DATA_ROOTS = [Path("/kaggle/input"), Path("/kaggle/tmp"), Path("/kaggle/working")]
roots = sorted({r for base in DATA_ROOTS if base.exists()
                for r in base.rglob("*/img1")})
if not roots:
    print("No sequences under", [str(b) for b in DATA_ROOTS])
    print("Either attach the SoccerNet-GSR dataset, or stage it:")
    print("  from optiona_sfr.kaggle_setup import ensure_gsr_data")
    print("  ensure_gsr_data(splits=('valid',))   # -> /kaggle/tmp/SoccerNetGS")
assert roots, "no SoccerNet-GSR sequences found under any data root"
seqs = sorted((len(list(r.glob("*.jpg"))), r.parent) for r in roots)
# THE AUDIT RUN: five sequences in one pass, SNGS-022 mandatory.
# 022 is the sequence whose calibration confidence bottomed at s min = 0.000 on
# the two-video run; excluding it would audit only the easy case. The rest are
# the shortest available, so the pass stays affordable.
MUST = ["SNGS-022"]
by_name = {p.name: (nf, p) for nf, p in seqs}
picked = [by_name[m] for m in MUST if m in by_name]
for nf, p in seqs:
    if len(picked) >= 5:
        break
    if p.name not in [q.name for _, q in picked]:
        picked.append((nf, p))
SEQS = [p.name for _, p in picked]
SEQ = picked[0][1]                      # kept for paths that still expect one
DATA_ROOT = SEQ.parent.parent
SPLIT = SEQ.parent.name                 # the split these sequences actually live in

# RESOLVED SPLIT GUARD.
# The run key must match the split on disk: passing vids_dict.test=[...] while
# the sequences sit under valid/ silently evaluates nothing. Worse, the default
# config is eval_set: "test", so an unguarded audit spends the test split.
#
# The protocol this project set for itself (OptionA_Field_Registration_Report
# section 5) is that alpha/beta are tuned on validation and the test set is
# touched ONCE, at the end. An audit run is the opposite of that: its purpose is
# to surface defects -- the team anomaly, jersey provisioning, TM1 -- which will
# then be fixed, and any later test number is contaminated by having been
# iterated against. Audit on valid; spend test on the result.
ALLOW_TEST = False                      # set True only to spend the touch-once budget
print("resolved split : %s" % SPLIT)
print("dataset root   : %s" % DATA_ROOT)
print("sequences (%d)  : %s" % (len(SEQS), ", ".join(SEQS)))
if SPLIT == "test" and not ALLOW_TEST:
    print("STOP: this would run on the TEST split.")
    print("  The audit exists to find defects, and fixing what it finds means")
    print("  iterating against test -- which spends the touch-once budget on")
    print("  plumbing rather than on the result.")
    print("  Attach the valid split and re-run, or set ALLOW_TEST=True to")
    print("  spend it deliberately.")
    raise SystemExit("refusing to run the audit on the test split")
missing = [m for m in MUST if m not in by_name]
if missing:
    print("*** MANDATORY SEQUENCE(S) NOT PRESENT: %s -- the audit is incomplete"
          % missing)
print("audit sequences (%d):" % len(SEQS))
for nf, p in picked:
    print("   %-12s %4d frames%s" % (p.name, nf, "   <- mandatory" if p.name in MUST else ""))
print("dataset root:", DATA_ROOT)
"""))

cells.append(code(r"""
# TEE STDOUT. boxmot's GMC reports a failed motion estimate with print(), not a
# log record (see the F4 probe), so the evidence never reaches the log file. Keep
# the raw stream: it is the primary source for the CMC assertion below.
RUNLOG = WORK / "run_audit5.txt"
EXP = "audit5"
# eval_set AND the vids_dict key both follow the resolved split -- hardcoding
# .test while the data sits under valid/ evaluates an empty set silently.
sh(PY + " -m tracklab.main -cn soccernet"
        " dataset.dataset_path=%s dataset.nvid=%d"
        " dataset.eval_set=%s"
        " 'dataset.vids_dict.%s=[%s]'"
        " experiment_name=%s 2>&1 | tee %s"
   % (DATA_ROOT, len(SEQS), SPLIT, SPLIT, ",".join(SEQS), EXP, RUNLOG), cwd=REPO)
"""))

cells.append(md(r"""
### Tracking assertions

Ordered by consequence, from the `track` / `gta_link` static audit. These are
instruments, not pass/fail gates — except the two that gate Stage 4.
"""))

cells.append(code(r"""
import re
text = RUNLOG.read_text(errors="replace") if RUNLOG.exists() else ""

print("1. CMC IS ALIVE  (F4 -- gates Stage 4)")
# boxmot prints on a failed estimate; count those against total frames.
warns = re.findall(r"(?i)not enough (?:matching )?points|warning.*gmc|cmc.*fail", text)
print("   GMC failure prints in captured stdout: %d" % len(warns))
TOTAL_FRAMES = sum(nf for nf, _ in picked)
print("   frames in run: %d across %d sequences" % (TOTAL_FRAMES, len(SEQS)))
if len(warns) > 0.05 * TOTAL_FRAMES:
    print("   *** CMC IS DEGRADED on >5%% of frames. You are running closer to the")
    print("       CMC-off configuration (HOTA 0.6315) than to 0.6687. Stage 4's")
    print("       absolute numbers are depressed for BOTH arms.")
elif warns:
    print("   some failures, below 5%% -- acceptable")
else:
    print("   no failure prints found. NOTE: absence of prints is not proof CMC is")
    print("   working -- it may simply not print. Confirm from the F4 probe whether")
    print("   the failure path prints at all.")

print("\n2. BATCH SIZE IS 1  (F2 -- gates Stage 4)")
print("   probe verdict: %s" % F2)
print("   (a silently batched tracker would still produce plausible output)")

print("\n3. FEATURE/DETECTION PARITY  (F6 -- now enforced)")
bad = re.findall(r"ReID returned \d+ features for \d+ detections", text)
print("   violations raised: %d  (a RuntimeError now stops the run)" % len(bad))

print("\n4/8. GTA-LINK MERGE HEALTH  (F9, F14)")
for line in re.findall(r"\[GTA-Link\][^\n]*", text):
    print("   " + line.strip()[:150])

print("\n6. ZERO-FEATURE RATE  (F11 -- now logged)")
z = re.findall(r"detections have a ZERO appearance feature", text)
print("   zero-feature warnings: %d  (0 expected on a healthy sequence)" % len(z))

print("\n9. DETECTOR CLASS IDS  (D2)")
# Expected: exactly {0}, and a single entry in model.names. Anything else means
# the cls==0 filter is discarding evaluated roles (goalkeeper / referee).
for line in re.findall(r"\[YOLO-SNFT\] class ids present[^\n]*", text):
    print("   " + line.strip()[:160])
bad_cls = re.findall(r"\[YOLO-SNFT\] checkpoint emits classes[^\n]*", text)
if bad_cls:
    print("   *** MULTI-CLASS CHECKPOINT -- roles are being dropped silently:")
    for line in bad_cls:
        print("       " + line.strip()[:160])
else:
    print("   no multi-class warning -- consistent with a single-class fine-tune")

print("\n10. DETECTOR imgsz  (D3)")
# Expected: matches the yaml's imgsz: 1280.
for line in re.findall(r"\[YOLO-SNFT\] (?:imgsz|checkpoint was TRAINED|checkpoint carries|could not read)[^\n]*", text):
    print("   " + line.strip()[:160])

print("\n11. ReID FEATURE NORMS  (a-path)")
for line in re.findall(r"\[BoT-SORT\] ReID feature norms[^\n]*", text):
    print("   " + line.strip()[:160])
degen = re.findall(r"\[BoT-SORT\] \d+ near-zero and \d+ non-finite[^\n]*", text)
if degen:
    print("   *** DEGENERATE FEATURES -- update_features divides by the norm with")
    print("       no epsilon, so these become nan in the cost matrix:")
    for line in degen:
        print("       " + line.strip()[:160])
"""))

cells.append(md(r"""
### Feature dump for the GTA-Link threshold probe

`docs/GTA_STC2025_PARAM_MAPPING.md` §4 records that STC-2025's
`merge_dist_thres=0.35` is measured with a **different estimator** to our
`appearance_thresh=0.25`: theirs is the mean pairwise cosine distance over all
instance pairs, ours the distance between EMA-averaged embeddings. By Jensen the
former is systematically larger, so the values are not interchangeable.

Settling the conversion needs **real** per-detection features — the gap depends
entirely on real within-tracklet variance, so synthetic data answers the wrong
question. This dumps them.
"""))

cells.append(code(r"""
# Dump per-detection OSNet features + track_id for the estimator-gap probe.
#
# RULE: instrumentation must never instantiate an *Api class.
# GTALink(cfg, ...) crashes outside hydra -- the yaml carries ${...}
# interpolations and the custom ${hf:} resolver, neither of which resolves in a
# bare OmegaConf.load. Earlier revisions did exactly that and the cell died every
# run. Instrumentation reads ARTIFACTS and builds models DIRECTLY.
import pickle
import zipfile

import numpy as np
import torch
from torchreid.utils import FeatureExtractor

feat_path = WORK / ("features_%s.npz" % EXP)
try:
    det, _ = read_state(EXP)
    assert det is not None and "track_id" in det.columns, "no state with track_id"
    work = det[det["track_id"].notna()].copy()

    # Standalone OSNet, same checkpoint the pipeline used, straight from the HF
    # cache. No hydra, no sn_gamestate import, no GTALink.
    w = next(iter(sorted(Path("/root/.cache/huggingface").rglob(
        "osnet_x1_0_sports.pt"))), None) or next(iter(sorted(
        (REPO / "pretrained_models").rglob("osnet_x1_0_sports.pt"))), None)
    assert w is not None, "osnet_x1_0_sports.pt not found in HF cache or pretrained_models"
    print("weights:", w)
    ex = FeatureExtractor(model_name="osnet_x1_0", model_path=str(w),
                          device="cuda" if torch.cuda.is_available() else "cpu")

    # Same preprocessing as gta_link and as gta-link's own generate_tracklets.py:
    # RGB crop -> Resize(256,128) -> ToTensor -> ImageNet norm, then L2 norm.
    import cv2
    import torchvision.transforms as T
    tfm = T.Compose([T.ToPILImage(), T.Resize((256, 128)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    _, img_meta = read_state(EXP)
    id2path = img_meta["file_path"].to_dict()

    feats = np.zeros((len(work), 512), np.float32)
    pos = {ix: i for i, ix in enumerate(work.index)}
    for image_id, grp in work.groupby("image_id"):
        fp = id2path.get(image_id)
        if fp is None:
            continue
        im = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        batch, rows = [], []
        for ix, r in grp.iterrows():
            l, t, ww, hh = [float(v) for v in r["bbox_ltwh"]]
            x1, y1 = max(0, int(l)), max(0, int(t))
            x2, y2 = min(W, int(l + ww)), min(H, int(t + hh))
            if x2 <= x1 or y2 <= y1:
                continue
            batch.append(tfm(im[y1:y2, x1:x2]))
            rows.append(ix)
        if batch:
            out = ex(torch.stack(batch)).cpu().numpy()
            out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
            for ix, f in zip(rows, out):
                feats[pos[ix]] = f

    np.savez_compressed(feat_path, feats=feats,
                        track_id=work["track_id"].to_numpy(),
                        image_id=work["image_id"].to_numpy())
    print("wrote %s  feats=%s  tracklets=%d"
          % (feat_path, feats.shape, work["track_id"].nunique()))
except Exception as e:
    import traceback
    print("feature dump FAILED (%s: %s)" % (type(e).__name__, e))
    traceback.print_exc()
"""))

cells.append(code(r"""
# 7. Tracklet length distribution + 8. identity counts, from the saved state.
# A .pklz is a ZIP (b'PK'), not gzip -- see read_state() above, which is the one
# loader. Earlier revisions used gzip.open here and silently got None every run.
import pickle
import zipfile

import numpy as np
import pandas as pd

det, _img = read_state(EXP)
if det is None:
    print("no state -- read the run log's eval table instead")

if det is not None and "track_id" in det.columns:
    tid = det["track_id"]
    print("\n7. TRACKLET LENGTH DISTRIBUTION")
    lens = tid.dropna().value_counts()
    print("   identities: %d   median len %.0f   p10 %.0f   p90 %.0f"
          % (len(lens), lens.median(), np.percentile(lens, 10),
             np.percentile(lens, 90)))
    short = int((lens < 20).sum())
    print("   below min_tracklet_len=20: %d/%d (%.0f%%) -- these are EXCLUDED from"
          % (short, len(lens), 100.0 * short / max(len(lens), 1)))
    print("   merging entirely, so a heavy short tail means GTA-Link operated on a")
    print("   small minority of tracks")

    print("\n8b. NON-NULL bbox_pitch  -- FIRST-CLASS RESULT, not a diagnostic")
    # soccernet_game_state.py:91-100 drops any detection with a null track_id,
    # bbox_ltwh OR bbox_pitch (how="any"). A calibration failure therefore does
    # not create false positives -- it removes the detection from scoring
    # entirely. Lost true positives cost recall, silently.
    if "bbox_pitch" in det.columns:
        nn = int(det["bbox_pitch"].notna().sum())
        print("   non-null bbox_pitch: %d/%d (%.1f%%)"
              % (nn, len(det), 100.0 * nn / max(len(det), 1)))
        # A whole sequence nulled by _empty_outputs looks like a clean run.
        for vid, g in det.groupby(det.get("video_id", pd.Series(0, index=det.index))):
            frac = g["bbox_pitch"].notna().mean()
            if frac < 0.10:
                print("   *** COLLAPSE: video %s has %.1f%% non-null bbox_pitch."
                      % (vid, 100.0 * frac))
                print("       _empty_outputs nulls an entire sequence on any early")
                print("       return -- this is a calibration FAILURE presenting as")
                print("       a clean run. Read the [optiona] log lines above.")
    else:
        print("   bbox_pitch column absent -- the calibration stage did not run")

    print("\n8. NaN track_id REACHING THE EVALUATOR  (F13)")
    n_nan = int(tid.isna().sum())
    print("   detections with NaN track_id: %d/%d (%.2f%%)"
          % (n_nan, len(tid), 100.0 * n_nan / max(len(tid), 1)))
    print("   probe verdict on encoder handling: %s"
          % probe.get("F13", {}).get("answer", "UNRESOLVED"))
    if n_nan and probe.get("F13", {}).get("answer") != "DROPS":
        print("   *** these may reach the encoder as spurious unmatched detections")
        print("       and depress GS-HOTA precision in BOTH arms")
"""))

cells.append(md(r"""
### Read the result — completeness is not evidence

Completeness is ~1.0 by construction: smoothing interpolates gaps, carry-forward
fills the rest, and the tracker never returns `None` after lock-on. A camera that
locks on at frame 1, drifts, and is carried for the remaining frames gives zero
errors, 100% completeness and wrong coordinates everywhere. These are the numbers
that separate that case from a working one.
"""))

cells.append(code(r"""
# PER-SEQUENCE AUDIT TABLE. This run closes the three OPEN items from
# docs/AUDIT_FINDINGS.md and gives per-sequence variance on the headline number.
import json
import re

import numpy as np
import pandas as pd

det, img = read_state(EXP)
assert det is not None, "no state -- the run did not complete"
log = RUNLOG.read_text(errors="replace") if RUNLOG.exists() else ""

# sequence label per detection: prefer video_id, else derive from the image path
vid2seq = {}
if img is not None and "file_path" in img.columns:
    for iid, fp in img["file_path"].items():
        vid2seq[iid] = Path(str(fp)).parent.parent.name
det = det.copy()
det["_seq"] = det["image_id"].map(vid2seq)
if det["_seq"].isna().all() and "video_id" in det.columns:
    det["_seq"] = det["video_id"].astype(str)

rows = []
for seq, g in det.groupby("_seq"):
    n = len(g)
    r = {"seq": seq, "dets": n}
    r["bbox_pitch"] = int(g["bbox_pitch"].notna().sum()) if "bbox_pitch" in g else 0
    r["track_id"] = int(g["track_id"].notna().sum()) if "track_id" in g else 0
    r["role"] = int(g["role"].notna().sum()) if "role" in g else 0
    r["team_cluster"] = int(g["team_cluster"].notna().sum()) if "team_cluster" in g else 0
    r["team"] = int(g["team"].notna().sum()) if "team" in g else 0
    if "jersey_number" in g:
        r["jn"] = int(g["jersey_number"].notna().sum())
    # calibration JSON: s percentiles, focal spread, corr(f,|C|)
    cj = REPO / "optiona_calib" / ("%s.json" % seq)
    if cj.is_file():
        blob = json.loads(cj.read_text())["frames"]
        names = sorted(blob)
        sv = np.array([blob[k].get("s", np.nan) for k in names], float)
        sv = sv[np.isfinite(sv)]
        if sv.size:
            r["s_min"] = round(float(sv.min()), 3)
            r["s_p10"] = round(float(np.percentile(sv, 10)), 3)
            r["s_med"] = round(float(np.median(sv)), 3)
            r["s_p90"] = round(float(np.percentile(sv, 90)), 3)
            r["s<0.5"] = int((sv < 0.5).sum())
        P_ = [blob[k]["parameters"] for k in names if blob[k].get("parameters")]
        if P_:
            f = np.array([q["x_focal_length"] for q in P_], float)
            C = np.array([q["position_meters"] for q in P_], float)
            d = np.linalg.norm(C, axis=1)
            r["f_spread%"] = round(100 * (f.max() - f.min()) / f.mean(), 1)
            r["corr_fZ"] = round(float(np.corrcoef(f, d)[0, 1]), 3)
    rows.append(r)

T = pd.DataFrame(rows).set_index("seq").sort_index()
pd.set_option("display.width", 200)
print("PER-SEQUENCE AUDIT")
print(T.to_string())

print()
print("=" * 74)
print("PRIORITY OPEN ITEM — the `team` anomaly")
print("=" * 74)
print("SNGS-021 showed team non-null 12996 while team_cluster was 10295 and role")
print("11870: 2701 detections with a team but no cluster. Systematic or one-off?")
for seq, r in T.iterrows():
    gap_c = r.get("team", 0) - r.get("team_cluster", 0)
    gap_r = r.get("team", 0) - r.get("role", 0)
    flag = "  <- team exceeds BOTH" if gap_c > 0 and gap_r > 0 else ""
    print("  %-12s team %5d  cluster %5d (gap %+5d)  role %5d (gap %+5d)%s"
          % (seq, r.get("team", 0), r.get("team_cluster", 0), gap_c,
             r.get("role", 0), gap_r, flag))
n_bad = sum(1 for _, r in T.iterrows() if r.get("team", 0) > r.get("team_cluster", 0))
print()
print("  %d/%d sequences show team > team_cluster -> %s"
      % (n_bad, len(T), "SYSTEMATIC" if n_bad > 1 else
         ("SNGS-021-only" if n_bad == 1 else "not reproduced")))

print()
print("=" * 74)
print("JERSEY — is this run comparable to the 54.321 baseline?")
print("=" * 74)
jn_tot = int(T["jn"].sum()) if "jn" in T else 0
print("  jersey_number non-null: %d / %d detections" % (jn_tot, int(T["dets"].sum())))
if jn_tot == 0:
    print("  *** STILL ZERO. Same failure as every prior run; GS-HOTA below is")
    print("      NOT the pipeline's score and NOT comparable to anything.")
else:
    print("  Jersey numbers ARE present for the first time.")
    print("  *** GS-HOTA from this run is therefore NOT comparable to 54.321:")
    print("      that figure was produced with jersey_number 0/12996, i.e. with")
    print("      the component structurally absent. Different measurement, same")
    print("      name. Compare only against runs that also had jersey numbers.")

print()
print("GTA-Link per sequence (from the log):")
for ln in re.findall(r"\[GTA-Link\].*", log):
    print("  ", ln.strip()[:120])
print()
print("IDSW / eval: read the CLEAR table in the run log (IDSW column).")
"""))

cells.append(md(r"""
## Stage 4 — result

BroadTrack was removed once the comparison was settled (KNOWN_LIMITATIONS
section 7). There is one arm now; `tracklab -cn soccernet` IS Option A.

Read the GS-HOTA table below together with the Stage 3 diagnostics — completeness
and a HOTA number say nothing on their own about whether the camera was live.
"""))

cells.append(code(r"""
import re

tag = EXP
d = REPO / "outputs" / tag
print("=== %s ===" % tag)
if not d.exists():
    print("  no output directory -- did the run complete?")
else:
    found = False
    for h in sorted(d.rglob("*.json"))[-5:]:
        try:
            obj = json.loads(h.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            if re.search(r"hota|deta|assa|accuracy", str(k), re.I):
                print("  %s: %s" % (k, v))
                found = True
    if not found:
        print("  no HOTA-like keys found; read the eval table in the run log")
"""))

cells.append(md(r"""
## What "ready" means

Not "it ran". Ready is:

- **parity + negative control** from Stage 3a, with the control failing loudly
- **`s` across the whole sequence**, with no falling quartile trend
- **out-of-bounds count** at or near zero — remembering it is a lower bound
- **camera parameters physically plausible** — focal not drifting, optical
  centre not wandering off the touchline
- **non-null `bbox_pitch` per arm**, balanced within 2% — or the A/B scored on
  the intersection instead
- **GS-HOTA** read together with everything above, not on its own

If any is missing, the run is not evidence — regardless of completeness.

See `KNOWN_LIMITATIONS.md` for the four defects that are understood and
deliberately not fixed, and what each one does and does not invalidate.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
# Backstop: re-parse every code cell AS IT WILL BE SERIALISED, from the joined
# source list, in case a transform between code() and here corrupted one.
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    joined = "".join(c["source"])
    try:
        ast.parse(joined)
    except SyntaxError as e:
        print(f"\n*** SERIALISED cell {i} is broken: {e.msg} at line {e.lineno}",
              file=sys.stderr)
        for k, ln in enumerate(joined.splitlines(), 1):
            print(f"  {k:3d}| {ln}", file=sys.stderr)
        sys.exit(2)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
print("wrote %s (%d cells, %d code cells all parse-checked)"
      % (OUT, len(cells), n_code))
