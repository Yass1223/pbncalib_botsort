# STC-2025 GTA-Link parameters → this repository

Source read: `github.com/sjc042/gta-link` `refine_tracklets.py` and
`generate_tracklets.py`, `main` branch, read in-browser (not cloned).
Target: `sn_gamestate/track/gta_link_api.py` and
`sn_gamestate/configs/modules/gta_link/gta_link.yaml`.

STC-2025 reference settings:
`--use_split --min_len 50 --eps 0.65 --min_samples 15 --max_k 2`
`--use_connect --spatial_factor 1.5 --merge_dist_thres 0.35`

**Headline: five of the eight parameters have no target in this port, and the
three that do are not measured in the same units.** Copying the numbers across
would not reproduce STC-2025's behaviour; it would silently change ours.

---

## Mapping table

| Their param | Our config key | Verdict |
|---|---|---|
| `use_split` | — | **NO EQUIVALENT** |
| `min_len` 50 | `min_tracklet_len` 20 | **NO EQUIVALENT** (different target) |
| `eps` 0.65 | — | **NO EQUIVALENT** |
| `min_samples` 15 | — | **NO EQUIVALENT** |
| `max_k` 2 | — | **NO EQUIVALENT** |
| `use_connect` | (always on) | exact |
| `spatial_factor` 1.5 | `spatial_thresh` 150 | **NO EQUIVALENT** (different formula and units) |
| `merge_dist_thres` 0.35 | `appearance_thresh` 0.25 | **approximate** (same metric family, different estimator, different algorithm) |

---

## 1. Does this port implement DBSCAN splitting? **No.**

Plainly: **there is no splitter.** `gta_link_api.py` builds one embedding per
`track_id` and merges; nothing ever divides a tracklet. `sklearn.cluster.DBSCAN`
is not imported, and no code path examines within-tracklet feature structure.

The official splitter (`split_tracklets` → `detect_id_switch`) does:

1. skip any tracklet with `len(times) < min_len` — left intact;
2. `StandardScaler().fit_transform(embs)` — per-dimension standardisation;
3. `DBSCAN(eps, min_samples, metric='cosine')` on the **scaled** embeddings;
4. reassign noise (`-1`) points to the nearest cluster centre by cosine;
5. if `n_clusters > max_k`, repeatedly merge the two closest cluster centres
   until `n_clusters == max_k`;
6. if more than one cluster survives, emit each as a **new tracklet with a new
   id**.

`min_len`, `eps`, `min_samples` and `max_k` are **exclusively** splitter
parameters — every one of them is consumed only inside that path. With no
splitter, **adopting those four values is meaningless**: they have nothing to
configure. Writing them into `gta_link.yaml` would produce four keys that the
module never reads, which is worse than absent — it implies a capability that
does not exist.

Implementing the splitter is a genuine feature, not a parameter change. It would
also be the more valuable half of GTA: the splitter fixes ID switches *inside* a
tracklet, which merging cannot do, and this pipeline currently has no mechanism
for that at all.

## 2. `min_len` 50 vs `min_tracklet_len` 20 — different targets, not a tweak

They are not the same knob and the numbers are not comparable.

| | Their `min_len` | Our `min_tracklet_len` |
|---|---|---|
| Gates | **splitting** | **merging** |
| Below threshold | tracklet is kept whole, never split | tracklet gets no embedding, **never merges** |
| Above threshold | tracklet is examined for ID switches | tracklet participates in merging |

`refine_tracklets.py` applies **no length filter to merging at all** — every
tracklet, however short, enters the distance matrix and can be merged. Our
`min_tracklet_len = 20` excludes short tracklets from merging entirely
(`gta_link_api.py`, the `continue` before the embedding is built), and they then
receive fresh unique ids.

So the honest reading is: **their merger is strictly more permissive than ours,
and our length filter has no counterpart in their design.** Raising ours to 50
would exclude *more* tracklets from merging — the opposite of what adopting a
"reference setting" suggests. This is behavioural and belongs on its own branch,
measured.

## 3. `spatial_factor` 1.5 vs `spatial_thresh` 150 px — different formulas

**Theirs** (`get_spatial_constraints`, `check_spatial_constraints`):

```
x_range = (max_x_centre − min_x_centre) × factor      # over ALL boxes in the sequence
y_range = (max_y_centre − min_y_centre) × factor
reject if  dx > x_range  OR  dy > y_range
```

- computed **once per sequence** from the observed spatial extent of every
  detection — it is data-derived, not an absolute pixel value;
- **axis-separate**: `dx` and `dy` are compared independently, not as a norm;
- **no frame-gap scaling whatsoever** — a constant per sequence;
- applied between the *last box of the earlier subtrack* and the *first box of
  the later one*, walking every consecutive segment pair, and rejecting on the
  first violation.

**Ours** (`gta_link_api.py`):

```
reject if  ‖last_centre_i − first_centre_j‖₂  >  spatial_thresh × √(frame gap)
```

- absolute pixel constant, **not** data-derived;
- **Euclidean**, not axis-separate;
- **scales with √(frame gap)**, which theirs does not do at all;
- evaluated on the tracklets' overall first/last boxes, not per segment.

There is no conversion between `1.5` and `150 px`. They are different
quantities: one is a dimensionless multiplier on an observed range, the other is
a pixel distance.

**A note that matters for both:** at `factor = 1.5`, for a broadcast sequence
where detections span most of the frame, `x_range ≈ 1.5 × ~1900 ≈ 2850 px` —
wider than the image, so their gate cannot reject on `dx` either. **Their
spatial gate is also close to inert on full-width sequences**, and bites only
when the observed detection spread is small (zoomed or static views). That is
the same conclusion reached independently about ours in
`KNOWN_LIMITATIONS.md` §1 — the difference is that theirs degrades gracefully on
zoomed sequences because it is data-derived, while ours degrades with the frame
gap regardless of content.

If the goal is to adopt their *design*, the change is to replace the
`√(gap)`-scaled Euclidean constant with a per-sequence, data-derived,
axis-separate range. That is a redesign of the gate, not a re-tuning.

## 4. `merge_dist_thres` 0.35 vs `appearance_thresh` 0.25

Same metric family — cosine distance in [0, 2], with time-overlapping pairs
forced to the maximum — but **three differences that make the numbers
non-transferable**.

**a. The distance is a different estimator.**

Theirs (`get_distance`) is the **mean pairwise cosine distance over every
instance pair**:

```
D(A,B) = (1 / |A||B|) · Σ_{i∈A} Σ_{j∈B} (1 − cos(f_i, g_j))
```

Ours is the cosine distance between **two EMA-averaged, L2-normalised
embeddings**:

```
D(A,B) = 1 − cos( ema(A), ema(B) )
```

These are not equal, and not equal in a fixed way. By Jensen's inequality the
mean of pairwise distances is **≥** the distance between means, with the gap
growing as within-tracklet feature variance grows. So **their distances are
systematically larger than ours for the same pair of tracklets** — which is
precisely why their threshold (0.35) is larger than ours (0.25). Transplanting
0.35 onto our estimator would merge substantially more aggressively than
STC-2025 does, not identically.

Their EMA has no counterpart either: `ema_alpha` is ours alone, and their
per-instance mean weights every detection equally.

**b. The algorithm is different.**

Theirs is a bespoke greedy loop: while any off-diagonal distance is below the
threshold, take the global `argmin`, check the spatial constraint, merge by
**concatenating the two feature lists**, delete the row/column, and **recompute
that row exactly** from the concatenated instances. A spatially-rejected pair has
its distance *set to* the threshold, permanently blocking it.

Ours is `sklearn.cluster.AgglomerativeClustering(metric="precomputed",
linkage="average", distance_threshold=appearance_thresh)`. Average linkage uses
the Lance-Williams update on the **original tracklet-level** distances; it never
recomputes from instances. Two consequences: our merged-cluster distances are an
algebraic approximation where theirs are exact, and ours can merge
time-overlapping tracklets **transitively** (handled downstream by the
per-frame collision guard), which their sequential argmin-with-recompute cannot.

**c. Normalisation differs upstream.** Their features are L2-normalised
per-detection at generation time (`generate_tracklets.py`,
`feat /= np.linalg.norm(feat)`); so are ours. That part matches. But theirs are
then averaged *inside the distance*, ours *before* it.

**Verdict: approximate.** The metric is comparable in kind, so 0.35 is a
meaningful reference point, but it is not a drop-in value.

---

## What a numeric probe would settle

The one open quantitative question is (4a): how large is the gap between the two
estimators on real data? That determines whether an "equivalent" threshold for
our estimator is nearer 0.25, 0.30, or something else.

**Probe** — no GPU, no pipeline, pure numpy, runnable in a sandbox:

> Given `N` tracklets of `n_i` L2-normalised 512-d features each, compute for
> every pair both `D_pairwise = mean_{i,j}(1 − cos(f_i, g_j))` and
> `D_ema = 1 − cos(ema(A), ema(B))` with `ema_alpha = 0.9`, and report the
> regression of one on the other plus the residual spread.

**What settles it:** if `D_pairwise ≈ a·D_ema + b` with tight residuals, then
`0.35` maps to a definite value on our scale and the threshold can be converted.
If the residual spread is wide, the two are not inter-convertible and our
threshold must be tuned directly against GS-HOTA rather than imported.

**Input required:** real per-detection OSNet features from one sequence — i.e.
the probe needs the `feats` array `_extract_features` produces. Synthetic
Gaussian features would answer the wrong question, because the gap depends
entirely on real within-tracklet variance. So this probe runs on Kaggle after a
Stage 3 run, dumping `feats` and the `track_id` grouping to an `.npz`, not here.

## Recommended sequencing

Nothing here is a parameter tweak. In increasing order of scope:

1. **Nothing to adopt** for `eps` / `min_samples` / `max_k` / `min_len` until a
   splitter exists. Do not add the keys.
2. **`min_tracklet_len`** — behavioural, own branch, measured. Note the
   direction: their design has *no* merge-length filter, so the reference-faithful
   move is to lower or remove ours, not to raise it to 50.
3. **`merge_dist_thres`** — needs the probe above before a value can be chosen.
4. **Spatial gate redesign** to per-sequence data-derived axis-separate ranges —
   the largest change, and the one most likely to help, since it fixes the
   inertness recorded in `KNOWN_LIMITATIONS.md` §1 on zoomed sequences.
5. **Implement the splitter** — a feature, not a setting, and the only thing here
   that addresses within-tracklet ID switches.
