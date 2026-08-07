# JN pipeline -- GSR production build (maxconf)

Frozen jersey-number recognition pipeline for integration into a GSR
(SoccerNet GameState) pipeline. No ablations, no audit, no sweeps: one
configuration, one merge, four metrics.

## Frozen configuration

    legibility gate   pl > 0.9        (strict; frame_verdicts is the rule)
    det gate          ds > 0.52       (cached DBNet++ top-quad score)
    consolidation     maxconf         score(L) = exp(best single-frame
                                      log-likelihood of L) x (summed frame
                                      confidence of L)
    -1                a tracklet with no surviving ROI frame

All three are the harness defaults; nothing needs to be passed.

Chosen on the six-rule comparison run of 2026-08-02 (full harness,
GSR-2024 test, frozen gates): maxconf trk_acc 84.20%, numbered 85.01%,
-1 F1 0.83, roi kept 57.2% -- the best of the six rules on this split
(wvote 83.91/84.63, plain vote 83.72/84.37). Caveat carried over from that
run: the rule was selected on the test split itself; treat the margin over
wvote (+0.29 pts) as indicative, not as a held-out gain.

## Kaggle validation (first step)

Attach this folder as a Kaggle dataset and run `kaggle_gsr_maxconf.ipynb`
(GSR test split only): env -> hash-checked weights -> staging -> dual-GPU
worker pass -> CPU merge -> the four metrics, compared against the
reference row above. The worker is seeded and resumable; the merge is
deterministic over the cache.

## Integration surface

    from jn_recognizer import JerseyNumberRecognizer
    rec = JerseyNumberRecognizer("models")        # rule="maxconf" default
    number, confidence, n_used = rec.predict(tracklet)

`tracklet` = iterable of (pil_image_or_path, xywh) or bare pre-cropped
images; `number` is "1".."99" or "-1". Batch alternative over staged
sequences: run_eval.py worker pass + `--merge --predict-only` writes the
{player_id: int} submission dict.

`--rule vote|wvote|sum|slogconf|sumexp` stays selectable in run_eval.py for
regression checks only (jn_recognizer accepts maxconf/wvote/vote).

## Files

run_eval.py (worker + merge), jn_recognizer.py (per-tracklet API),
evaluate_jn.py (consolidation rules + metrics; `python evaluate_jn.py`
runs its offline self-tests), legibility / crop_classifier / roi_dbnet /
dbnet_infer (models), gsr_adapter + stage_data (GSR loading), fetch_weights
+ stage_weights (hash-checked checkpoints), dual_gpu (sharded worker),
common / stage_utils / setup_env / setup_kaggle (infra),
kaggle_gsr_maxconf.ipynb (validation notebook), MANIFEST.sha256
(verified by the notebook bootstrap). SN-JNR support was removed from this
build; use the full harness for cross-dataset work.
