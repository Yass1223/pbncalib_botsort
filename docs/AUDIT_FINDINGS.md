# Audit findings — closures and open items

Provenance is marked on every line, because it decides how much a closure is
worth:

- **read-from-artifact** — measured from a saved state, log or dump produced by a
  real run. Data-dependent claims can only be closed this way.
- **read-from-source** — determined by reading the installed code. Definitive for
  *code properties*; says nothing about whether a path was exercised.

---

## Closed read-from-artifact

Source: `optiona_SNGS-021.pklz`, `run_optiona_SNGS-021.txt`,
`features_SNGS-021.npz` (SNGS-021, 750 frames, 12 996 detections).

### `bbox_pitch` completeness — section 4 does not apply to this run

`bbox_pitch` non-null **12 996 / 12 996 (100%)**. The evaluator's
`dropna(subset=["track_id","bbox_ltwh","bbox_pitch"], how="any")` leaves
**11 870**, and the binding constraint is **`track_id` (11 870)**, not
`bbox_pitch`. Calibration abstained on nothing.

Both arms scored the identical 11 870 detections — intersection 11 870,
symmetric difference 0, imbalance **0.00%** — so the final comparison measured
calibration quality, not willingness to abstain.

### Seams — the chain is exact

Index `int64 [0 .. 12995]`, unique, and `img.index.equals` holds across states.
Non-null counts chain without leakage:

| stage | column | non-null |
|---|---|---|
| bbox_detector | `bbox_ltwh`, `bbox_conf` | 12 996 |
| reid | `embeddings`, `role_detection` | 12 996 |
| track / gta_link | `track_id`, `track_bbox_*` | 11 870 |
| tracklet_agg | `role` | 11 870 |
| calibration | `bbox_pitch` | 12 996 |
| team | `team_cluster` | 10 295 |

`12 996 → 11 870 → 11 870 → 10 295` is exact at every seam.

### f–Z ambiguity — confirmed, and the expected sign was inverted

Per-frame, from the `parameters` column of `021_image.pkl` (n = 750):

| quantity | value |
|---|---|
| `corr(f, ‖C‖)` | **+0.8399** |
| `corr(f, Cy)` | **+0.8427** |
| `corr(f, Cz)` | **−0.7449** |
| `f` range | 1788.3 → 3992.3 (78.7% of mean) |
| `‖C‖` range | 46.51 → 60.66 m, travel 376.2 m |

**Positive is the confirming sign.** Holding image size fixed requires `f ∝ Z`,
so focal length and depth must move *together*; a prediction of ≈ −1 was wrong
about the sign while right about the phenomenon. On a sequence with no zoom, a
78.7% focal swing correlated at +0.84 with camera depth is calibration drift
along the optical axis, not real zoom. The projection stays self-consistent
(LocA 93.914, 0 out-of-bounds), which is exactly why no image-space metric
catches it.

**Boundary:** this does not threaten a result that only consumes `bbox_pitch`,
because the two errors cancel in the projection. It does threaten anything
consuming `parameters` directly.

### Estimator gap — STC-2025's threshold is not transferable, but ours is fine

276 pairs over 24 tracklets (≥ 20 detections), features L2-normalised (norm
1.0000 throughout, 0 zero-norm):

```
D_pairwise = 0.6218 · D_ema + 0.2234     resid std 0.0418     corr 0.8657
their 0.35  ->  D_ema equivalent 0.2037     ours 0.25     gap 0.0463
```

Gap ≈ residual noise. **Verdict: indistinguishable, keep 0.25.**

**Limitation, and it is not small:** only **46 / 276** pairs sit below 0.25, and
the dump is taken **post-merge** — every pair that actually merged is gone by
construction. The fit is therefore on a censored sample from which the
decision-relevant region has largely been removed. Treat the regression as
evidence that the two estimators are *broadly* comparable, not as a calibration
curve.

### Diagnostics from the run log

| finding | evidence |
|---|---|
| D2 class ids | `class ids present in raw output: [0]; model.names={0: 'item'}` |
| D3 imgsz | `imgsz 1280 matches` |
| ReID norms | `min=16.45 median=18.24 max=21.87`; no degenerate-feature warning ⇒ 0 zero, 0 non-finite |
| F4 CMC alive | `grep -c 'not enough matching points'` = **0** in 750 teed frames |
| F14 de-dup | `de-dup nulled 0 detection(s)` |
| F11 zero features | no warning emitted |
| calibration `s` | `min=0.151 median=0.940 max=1.000 (n=750/750)`, first lock-on `000001.jpg` |
| out-of-bounds | `0/12996 (0.0%)` — **a lower bound, not a measurement** |

---

## Closed read-from-source

Code properties, settled by reading the installed `tracklab-1.3.24` wheel. These
needed no run, and `fork_probe.json` got two of them wrong (see below).

| | conclusion | location |
|---|---|---|
| F2 | `batch_size` hardcoded to 1; the config's 64 is inert | `bot_sort_api.py:25-26` |
| F7 | state vector is **(x, y, w, h)** — correct. `tlwh_to_xyah` is defined and **never called** | `bot_sort.py:116, 127, 155` vs `:198` |
| F8 | EMA present, guarded by `if new_track.curr_feat is not None` | `bot_sort.py:42-50, 157` |
| F13 | evaluator drops NaN `track_id` | `soccernet_game_state.py:91-100` |
| F4 | the failure print exists | `gmc.py:290-291` |

### The probe was wrong twice

`fork_probe.json` from the real run reported **F4 = SILENT** and
**F7 = MIXED**. Both were defects in the probe, not findings:

- **F4** — the candidate list resolved `GMC.apply` first and stopped. `apply` is
  the *dispatcher*; the print lives in `applySparseOptFlow`. Grepping one method
  answered a question nobody asked.
- **F7** — it grepped for the *symbols* `tlwh_to_xyah` / `tlwh_to_xywh`. Both
  exist, so it said MIXED. But only `tlwh_to_xywh` is ever **called**.

Rewritten to grep whole-class source, count **call sites** rather than symbol
presence, and report **UNRESOLVED** wherever it cannot see its target — never
SILENT. Absence of evidence reported as evidence of absence is the worst failure
an instrument can have, because it looks like an answer.

---

## Open

| item | what it needs |
|---|---|
| **`team` column anomaly** | `team` non-null **12 996** while `team_cluster` is 10 295 and `role` is 11 870. 2 701 detections carry a team with no cluster; 1 126 carry one with neither `track_id` nor `role`. Neither `.loc` path in `tracklet_team_side_labeling_api.py` explains it. One targeted query on the state. |
| **`s` percentiles** | p10 / p90 / quartile medians live in the calib JSON, which is not in the state archives. The state carries `parameters`, not `s`. |
| **TM1 at nvid > 1** | The nvid=2 state was not attached. TM1 is invisible at nvid=1 by construction — the detector's global `self.id` makes the first video's index coincidentally match a fresh `RangeIndex`. |

---

## Branch order for IDSW, evidence-led

1. **DBSCAN splitter** — absent entirely. Nothing here can split a tracklet
   carrying two identities, and GTA-Link merged 35 → 24 without any counterweight.
   The only mechanism that addresses within-tracklet ID switches.
2. **F3** (`min_confidence` 0.1 → 0.05) — feeds BYTE's second association, the
   mechanism that recovers tracks through occlusion. That band is currently empty.
3. **F5** (CMC exclusion mask) — CMC is already alive (0 failure prints), so this
   improves a working component. Bounded upside, and it risks pushing the corner
   count below the 5-point threshold.
4. **F10** (spatial cap) — lowest. The gate rejected nothing detectable, and the
   long-gap regime barely occurs in one 750-frame sequence.
