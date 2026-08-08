# Known limitations

Defects and design compromises that are understood, deliberate, and **not** fixed
in this repository. Recorded here so they are not rediscovered as surprises, and
so that anyone reading a result knows what it does and does not establish.

Each entry states what is wrong, why it is not fixed here, and what would fix it.

---

## 1. The GTA-Link spatial gate cannot be principled in image coordinates

`sn_gamestate/track/gta_link_api.py` gates a candidate tracklet merge on the
pixel distance between the last box centre of one tracklet and the first box
centre of the next, scaled by `spatial_thresh * sqrt(frame gap)`.

**The problem is the coordinate system, not the scaling.** With a panning and
zooming broadcast camera there is no physical bound on pixel displacement: the
same player standing still can move most of the frame width between two
temporally distant tracklets. No image-space threshold is therefore principled.
At the tuned `spatial_thresh = 150` the threshold crosses the 1920 px frame width
at a gap of ~164 frames (~6.5 s), beyond which the gate cannot reject anything and
appearance is the only remaining constraint — a silent no-op.

**What would fix it:** gate in *pitch* coordinates, where a bound is physical
(a player covers at most ~10 m/s of real ground regardless of camera motion).

**Why not here:** `bbox_pitch` does not exist yet at that point in the pipeline.
The stage order is `bbox_detector -> reid -> track -> gta_link -> calibration`,
so GTA-Link runs *before* the calibration stage that produces pitch coordinates.
Using them requires reordering the pipeline, which changes what every downstream
stage sees and cannot be evaluated as a single attributable change.

**Current state:** the branch `fix/f10-spatial-cap` caps the multiplier at
`min(sqrt(gap), 4.0)` (600 px) and counts how often the cap is the binding
constraint. That replaces a silent no-op with a stated constant. It is a
mitigation, not a fix — if the measurement shows no effect, the honest reading is
that the gate was never doing work and merging is appearance-only beyond a few
seconds.

---

## 2. `track: {batch_size: 64}` is inert

`sn_gamestate/configs/soccernet.yaml` carries:

```yaml
modules: # module-specific batch sizes (override the per-module yaml values)
  bbox_detector: {batch_size: 4}
  reid: {batch_size: 64}
  track: {batch_size: 64}
```

For the `track` stage the value is **silently discarded**. tracklab's
`wrappers/track/bot_sort_api.py:25-26` is:

```python
def __init__(self, cfg, device, **kwargs):
    super().__init__(batch_size=1)
```

`batch_size` is hardcoded to 1 and `cfg` is never consulted for it.
`NotebookBotSORT` overrides only `reset()` and `process()`, so it inherits this.
`ImageLevelModule.dataloader` then builds its `DataLoader` with `batch_size=1`.

**This is benign** — indeed it is what makes the stage correct, since
`process()` indexes `batch["input"][0]` and `metadatas["file_path"].values[0]`
and would silently drop 63 of every 64 frames if a real batch ever arrived.

**The comment is wrong.** "override the per-module yaml values" is false for this
module. The line is retained rather than deleted so that this note has something
to attach to, and so nobody reintroduces it believing it does something.

---

## 3. `gmc.py:283` compares `prevPoints` with itself

In tracklab's bundled `bot_sort/gmc.py`, `applySparseOptFlow` guards the affine
estimate with:

```python
if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
```

The second clause is a tautology. It was evidently intended to be
`np.size(currPoints, 0)`, i.e. a check that the two correspondence arrays have
equal length.

**Currently harmless:** both arrays are appended to in lockstep in the loop
immediately above, so their lengths cannot diverge. The intended check simply
does not exist.

**Not fixed here** because it is upstream (tracklab's vendored fork of boxmot),
and patching a dependency in place would be invisible to anyone reading this
repository and lost on the next reinstall.

**Related, and not a defect but a property to know:** when that guard fails,
`applySparseOptFlow` executes `print('Warning: not enough matching points')` and
returns `H = np.eye(2, 3)` — an identity warp, meaning camera-motion compensation
is silently off for that frame. It **prints; it does not log**, so the evidence
never reaches the log file. The Kaggle notebook tees stdout during Stage 3 for
exactly this reason, and the CMC assertion reads that capture.

---

## 4. Evaluation drops detections with a null `bbox_pitch`

`tracklab/wrappers/dataset/soccernet/soccernet_game_state.py:91-100`:

```python
# Detections with no track_id will therefore be removed and not count as FP at evaluation
dataframe.dropna(subset=["track_id", "bbox_ltwh", "bbox_pitch"], how="any", inplace=True)
```

`how="any"` means a detection is removed if **any** of the three is null — so a
calibration failure does not produce false positives, it **shrinks the evaluated
set**. Lost true positives cost recall, but silently, and a whole sequence nulled
by `_empty_outputs` in the calibration stage looks like a clean run.

**Consequence for the A/B:** if the Option A and BroadTrack arms produce
materially different non-null `bbox_pitch` counts, they are not being scored on
the same detections, and the GS-HOTA delta partly measures *willingness to
abstain* rather than calibration quality. The notebook therefore reports the
non-null count per arm as a first-class result and, when the counts differ, also
scores both arms on the intersection.

**Not "fixed":** the behaviour is upstream and arguably correct for the
challenge's scoring rules. It is the *interpretation* that has to account for it.

---

## 5. Investigated and disproved: the tracklet EMA is *not* harmed by zero features

Recorded so it is not re-raised. A static audit flagged
`gta_link_api.py` for seeding the per-tracklet appearance EMA with `gf[0]`, which
is a zero vector whenever the tracklet's first crop failed. The concern was that
at `ema_alpha = 0.9` a zero seed would bias the embedding for dozens of
detections.

**It does not.** The loop renormalises after every step:

```python
emb = self.ema_alpha * emb + (1.0 - self.ema_alpha) * v
emb /= (np.linalg.norm(emb) + 1e-6)
```

- **zero seed** — `0.9·0 + 0.1·v₁ = 0.1·v₁`, which normalises to exactly `v₁`.
  The zero is erased in a single update.
- **zero mid-sequence** — `0.9·emb + 0.1·0 = 0.9·emb`, which normalises to
  `emb`. A literal no-op.
- **all zero** — stays zero, sits at cosine distance 1.0 from everything, and is
  therefore unmergeable. Unchanged, and now reported by the zero-feature counter
  in `_extract_features`.

Measured: filtering zero rows before the EMA changes the resulting embedding by
at most **1.5e-6** across 2000 randomised tracklets (20-60 detections, 15% zeros,
half of them seeded at position 0) — float noise from the `1e-6` epsilon.

A `fix/f12-ema-seed` branch was written and then abandoned on this evidence. The
zero-feature *counters* remain valuable — they reveal crops that failed — but the
EMA needs no change.

---

## 6. The dependency tree is almost entirely unpinned

Two version-drift failures have now cost a session each: `shapely` absent
entirely, then `huggingface_hub` resolving to 1.x and removing `HfFolder`. Both
have the same structural cause, recorded here so the third one is expected.

### tracklab pins nothing

`tracklab-1.3.24`'s own `METADATA` declares **30 dependencies and bounds none of
them**. Every entry is a bare name:

```
hydra-core  lightning  pytorch_lightning  numpy  ultralytics  filterpy  torch
torchvision  soccernet  yt-dlp  gdown  pandas  matplotlib  rich  tabulate
sn-trackeval  lap  distinctipy  rtmlib  transformers  accelerate
huggingface-hub  opencv-python  tqdm  omegaconf  requests  wandb  scipy
yacs  scikit-image
```

Anything we do not bound in our own `pyproject.toml` resolves to whatever is
newest on the day the venv is built. That is why a fresh resolve on a machine
that has never run this pipeline behaves differently from one built months ago,
and why `uv.lock` being uncommitted (see the README) trades reproducibility for
correctness — the lock was stale in the other direction.

Highest remaining risk from that list, by likelihood of a breaking change on a
path this pipeline actually uses:

| Package | Why it is exposed |
|---|---|
| `transformers` | canonical `HfFolder` importer; the prime suspect for the failure that prompted this section |
| `ultralytics` | the detector. Frequent default and API changes; already forced one subclass (`YOLOUltralyticsSNFT`) to pin `imgsz`, `conf` and `iou` explicitly because the stock wrapper inherited whatever ultralytics defaulted to |
| `hydra-core` / `omegaconf` | every config in the repo resolves through these |
| `pandas` | 3.0 changes copy/view semantics; the engine contract passes DataFrames between every stage |
| `lightning` / `pytorch_lightning` | heavy, breaks often, and only the discarded TVCalib module ever needed them |
| `sn-trackeval` | produces the GS-HOTA numbers the whole A/B rests on |

### Our own two git dependencies have no revision

Worse than an unpinned PyPI package, because there is not even a version to
record:

```toml
"prtreid @ git+https://github.com/VlSomers/prtreid",
"torchreid @ git+https://github.com/VlSomers/bpbreid",
```

Both resolve to the default branch's **HEAD at install time**. `prtreid` is the
ReID model behind team clustering and role classification; `torchreid`/`bpbreid`
supplies the OSNet loader used by BoT-SORT's appearance association and by
GTA-Link. A silent upstream commit changes embedding behaviour with nothing in
the repository to point at, and no way to tell two runs apart after the fact.

**Not fixed here** because pinning them to a commit is a *behavioural* change —
it would freeze a specific model implementation, which may differ from whatever
produced any previously recorded number. It belongs on its own branch, measured,
like every other behavioural change. But it is the single largest unpinned
surface in the project, and it is in our `pyproject.toml`, not tracklab's.

**What to do about it:** capture `uv pip freeze` from the first successful Kaggle
venv as an artefact, and pin the two git dependencies to the commits that build
resolved. That converts an unbounded surface into a recorded one without
guessing which commit is "right".

### transformers disables its torch models under torch 1.13 — expected

Every run log opens with:

```
Disabling PyTorch because PyTorch >= 2.1 is required but found 1.13.1
```

That is `transformers` (pulled in unbounded by tracklab) declining to register its
PyTorch model classes. **It is harmless here.** Nothing in this pipeline uses a
transformers model: tracklab's only two importers are
`wrappers/bbox_detector/transformers_api.py` and
`wrappers/pose_estimator/transformers_api.py`, and neither is in our config.
The line is noise, not a warning — recorded so it is not mistaken for one.

---

## 7. BroadTrack removed

The BroadTrack calibration arm existed to answer one question: is Option A
better? It was, and the arm then cost more to maintain than it returned.

**Removed:** `sn_gamestate/calibration/broadtrack_api.py`,
`sn_gamestate/configs/modules/calibration/broadtrack.yaml`,
`scripts/setup_broadtrack.sh`, `scripts/verify_broadtrack_conversion.py`, and the
notebook's Stage 4 second arm. `soccernet_optiona.yaml` became the content of
`soccernet.yaml` and was deleted, so there is **one entry point**:
`tracklab -cn soccernet`.

**Last present at `8f611d1`.** Recoverable in full from history:

```bash
git show 8f611d1:sn_gamestate/calibration/broadtrack_api.py
git log --all --full-history -- sn_gamestate/calibration/broadtrack_api.py
```

**Final comparison, on SNGS-021:** Option A **GS-HOTA 54.321** (DetA 34.789,
AssA 84.824, LocA 93.914) against BroadTrack **34.544 or 41.072** — two
BroadTrack logs collided and the pairing was never resolved. That ambiguity is
recorded rather than hidden, and is no longer relevant: both candidate values sit
far below Option A, so the ordering was never in doubt even though the margin is.

Both arms scored the **identical 11,870 detections** (imbalance 0.00%), so
section 4's abstention concern does not apply to that result — see
`docs/AUDIT_FINDINGS.md`.

Lineage references to BroadTrack survive throughout `optiona_sfr/` and in
`KNOWN_LIMITATIONS.md` section 3. Those are correct: the temporal layer is a
reimplementation of BroadTrack's published objective, and `gmc.py` is still the
GMC in use. Only the *arm* is gone, not the intellectual debt.
