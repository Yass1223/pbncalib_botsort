# jn_recognizer.py -- the production jersey-number module.
#
# The single integration surface for a GSR-style pipeline: one class, one
# call per tracklet. Configuration is FROZEN at the validated operating point
# and is not a constructor knob by default:
#
#     legibility > 0.9        (strictly greater; plateau member on both
#                              validation datasets)
#     DBNet++ top quad > 0.52 (strictly greater; SN-JNR test argmax; measured
#                              neutral-to-slightly-positive on both datasets)
#     maxconf consolidation  over the surviving frames' PARSeq readings:
#                             score(L) = exp(best single-frame log-lik of L)
#                             x summed frame confidence of L (production
#                             default, frozen on the six-rule GSR comparison;
#                             rule="wvote"/"vote" restore the earlier rules)
#
# Validated on GSR-2024 test (GT boxes, whole split, this exact merge path):
#     rule="maxconf"  trk_acc 84.20  numbered 85.01  -1 F1 0.83  roi_kept 57.2%
#     rule="vote"     trk_acc 83.72  numbered 84.37  -1 F1 0.83  (v6 frozen)
# kaggle_gsr_maxconf.ipynb reproduces the maxconf row and is the acceptance
# gate for this package.
#
# Usage:
#     from jn_recognizer import JerseyNumberRecognizer
#     rec = JerseyNumberRecognizer("models")            # loads all 3 nets
#     number, confidence, n_used = rec.predict(tracklet)
#
#     `tracklet` is an iterable of frames; each frame is either
#         (pil_image_or_path, xywh)   with xywh = (x, y, w, h) player box
#         pil_image_or_path           alone -> treated as an ALREADY-CROPPED
#                                     player image (xywh=None)
#     `number` is "1".."99" or "-1" (not visible); `confidence` is the
#     winning label's share of the consolidation score (maxconf by
#     default; 0.0 when none survive); `n_used` is the number of frames
#     that entered the vote.
#
# Efficiency note vs the evaluation harness: the harness runs PARSeq on every
# detected frame and gates at merge time (so thresholds can be re-applied
# without a GPU pass). Here the gates run FIRST and PARSeq reads only the
# survivors -- typically ~40% of detected frames at the frozen thresholds --
# which is the correct trade for production. Predictions are identical: the
# gates are conjunctive and the vote sees the same surviving set.
#
# The pre-cropped input mode matches the SN-JNR validation setting exactly:
# the player pad is inert on already-cropped images (there is no surrounding
# frame to pad into), so expect SN-JNR-like rather than GSR-like behaviour.

import os
from types import SimpleNamespace

import numpy as np
from PIL import Image

from common import subsample
from crop_classifier import crop_box
from evaluate_jn import (consolidate_tracklet_maxconf,
                         consolidate_tracklet_vote,
                         consolidate_tracklet_weighted_vote)
from legibility import frame_verdicts

# The frozen operating point. Provenance: legibility 0.9 chosen from the
# SN-JNR test sweep plateau (0.7-0.95 flat, cliff above); det 0.52 the SN-JNR
# test argmax at that legibility, neutral on GSR test.
LEGIBILITY_THR = 0.9
DET_THR = 0.52

# Sentinel "whole image" box for pre-cropped inputs; crop/pad code clips it.
_FULL = (0.0, 0.0, 1e9, 1e9)


class JerseyNumberRecognizer:
    """Frozen-config tracklet -> jersey number. See module docstring."""

    def __init__(self, models_dir="models", stride=5, fp16=True,
                 models=None,
                 dbnet_ckpt="best_icdar_hmean_epoch_10.pth",
                 parseq_ckpt="koshkina_sn_parseq.ckpt",
                 legibility_ckpt="sn_legibility.pth",
                 dbnet_cfg="mmocr_cfg/dbnetpp_infer.py", rule="maxconf"):
        """models_dir must hold the three validated checkpoints (fetch with
        fetch_weights.py). `models` injects prebuilt {"gate","leg","read_many"}
        -- the offline-test hook; when given, nothing is loaded and fp16 is
        ignored. stride: read every Nth frame (5 == the validated setting).
        rule: "maxconf" (PRODUCTION: score(L) = exp(best single-frame
        log-lik of L) x summed frame confidence of L, the rule frozen on
        the six-rule GSR comparison), "wvote" (the v7 weighted majority),
        or "vote" (the v6 plain majority the validated numbers in the
        header were produced under)."""
        assert rule in ("maxconf", "vote", "wvote"), rule
        self._consolidate = {"maxconf": consolidate_tracklet_maxconf,
                             "wvote": consolidate_tracklet_weighted_vote,
                             "vote": consolidate_tracklet_vote}[rule]
        self.rule = rule
        self.stride = max(1, int(stride))
        self._fp16 = bool(fp16)
        if models is not None:
            self._M = models
            return
        import run_eval  # reuse the validated builder -- no duplicate logic
        a = SimpleNamespace(
            ckpt=os.path.join(models_dir, dbnet_ckpt),
            parseq=os.path.join(models_dir, parseq_ckpt),
            legibility_weights=os.path.join(models_dir, legibility_ckpt),
            dbnet_cfg=dbnet_cfg, legibility_thr=LEGIBILITY_THR,
            legibility_size=256, det_floor=0.2, roi_pad=0.12,
            player_pad=0.18, det_cache=512)
        self._M = run_eval.build_models(a)

    # ------------------------------------------------------------------ #
    def _amp(self):
        if not self._fp16:
            import contextlib
            return contextlib.nullcontext()
        try:
            import torch
            if torch.cuda.is_available():
                return torch.autocast("cuda", dtype=torch.float16)
        except ImportError:
            pass
        import contextlib
        return contextlib.nullcontext()

    @staticmethod
    def _norm(frame):
        """Accept (img, xywh) | (img,) | bare img; path or PIL."""
        img, xywh = (frame if isinstance(frame, (tuple, list)) and
                     len(frame) == 2 else (frame, None))
        if isinstance(img, (str, os.PathLike)):
            key, im = str(img), Image.open(img).convert("RGB")
        else:
            key, im = f"mem-{id(img)}", img.convert("RGB")
        return key, im, (_FULL if xywh is None else tuple(map(float, xywh)))

    def predict(self, tracklet):
        """-> (number "1".."99" | "-1", vote_share, n_frames_in_vote)."""
        M = self._M
        frames = subsample(list(tracklet), self.stride)

        # pass 1: decode once; legibility crops + detector (score only kept)
        crops, ci, det = [], [], []
        for i, frame in enumerate(frames):
            try:
                key, im, xywh = self._norm(frame)
            except Exception:
                det.append(None)
                continue
            c, _ = crop_box(im, xywh)
            if c is not None:
                crops.append(c)
                ci.append(i)
            with self._amp():
                roi, _src, ds = M["gate"].roi_crop_scored(key, xywh, img=im)
            det.append(None if roi is None else (roi, float(ds)))

        pl = np.zeros(len(frames))
        if crops:
            with self._amp():
                scores, _ = M["leg"].score(crops)
            for j, i in enumerate(ci):
                pl[i] = float(scores[j])

        # gates FIRST (identical semantics to the harness merge: both strict)
        leg_ok = frame_verdicts(pl, LEGIBILITY_THR)
        rois = [d[0] for i, d in enumerate(det)
                if d is not None and leg_ok[i] and d[1] > DET_THR]

        # PARSeq only on survivors, then the configured vote rule
        if not rois:
            return "-1", 0.0, 0
        with self._amp():
            logls = M["read_many"](rois)
        label, share = self._consolidate(
            [t for t, _ in logls], [u for _, u in logls])
        return label, float(share), len(rois)


# ------------------------------ self-tests ------------------------------- #
if __name__ == "__main__":
    # Offline: stubbed models, no torch/GPU/checkpoints. Verifies gate order,
    # strictness at both thresholds, vote, -1 path, pre-cropped mode, stride.
    def _logl(hot):
        v = [-9.0] * 11
        v[hot] = -0.1
        return v

    class StubGate:
        def __init__(self, plan):            # key-index -> (ds | None)
            self.plan, self.calls = plan, 0

        def roi_crop_scored(self, key, xywh, img=None):
            ds = self.plan[self.calls]
            self.calls += 1
            if ds is None:
                return None, "none", 0.0
            return img, "det", ds            # roi = the image itself (stub)

    class StubLeg:
        def __init__(self, pls):
            self.pls = list(pls)

        def score(self, crops):
            out, self.pls = self.pls[:len(crops)], self.pls[len(crops):]
            return out, None

    def reader_for(digits):                  # each surviving roi -> a digit
        it = iter(digits)

        def read_many(rois):
            return [( _logl(next(it)), _logl(10)) for _ in rois]
        return read_many

    imgs = [Image.new("RGB", (40, 60), (i * 9, 50, 50)) for i in range(6)]

    # 1) gates + vote: pl (per-crop order) and ds thresholds strict
    rec = JerseyNumberRecognizer(models={
        "gate": StubGate([0.9, 0.9, 0.52, 0.9, None, 0.9]),
        "leg": StubLeg([0.99, 0.90, 0.99, 0.99, 0.99, 0.99]),
        "read_many": reader_for([7, 7, 9])}, stride=1)
    tr = [(im, (0, 0, 40, 60)) for im in imgs]
    n, conf, used = rec.predict(tr)
    # frame0 in (7); frame1 pl=0.90 NOT > 0.9 -> leg gate; frame2 ds=0.52 NOT
    # > 0.52 -> det gate; frame3 in (7); frame4 no roi; frame5 in (9).
    # DEFAULT rule is now maxconf. Equal per-frame conf c and equal joint
    # log-liks m: score(7)=exp(m)*2c, score(9)=exp(m)*c -> share 2/3
    assert (n, used) == ("7", 3) and abs(conf - 2 / 3) < 1e-9, (n, conf, used)
    # rule="wvote" restores the v7 count-weighted share 4/5
    rec_w = JerseyNumberRecognizer(models={
        "gate": StubGate([0.9, 0.9, 0.52, 0.9, None, 0.9]),
        "leg": StubLeg([0.99, 0.90, 0.99, 0.99, 0.99, 0.99]),
        "read_many": reader_for([7, 7, 9])}, stride=1, rule="wvote")
    n_w, conf_w, used_w = rec_w.predict(tr)
    assert (n_w, used_w) == ("7", 3) and abs(conf_w - 4 / 5) < 1e-9
    # rule="vote" on the same frames restores the v6 plain-count share 2/3
    rec_v = JerseyNumberRecognizer(models={
        "gate": StubGate([0.9, 0.9, 0.52, 0.9, None, 0.9]),
        "leg": StubLeg([0.99, 0.90, 0.99, 0.99, 0.99, 0.99]),
        "read_many": reader_for([7, 7, 9])}, stride=1, rule="vote")
    n_v, conf_v, used_v = rec_v.predict(tr)
    assert (n_v, used_v) == ("7", 3) and abs(conf_v - 2 / 3) < 1e-9

    # 2) nothing survives -> -1 with zero confidence
    rec = JerseyNumberRecognizer(models={
        "gate": StubGate([0.4]), "leg": StubLeg([0.99]),
        "read_many": reader_for([])}, stride=1)
    assert rec.predict([(imgs[0], (0, 0, 40, 60))]) == ("-1", 0.0, 0)

    # 3) pre-cropped mode: bare images, full-image sentinel reaches the gate
    seen = []

    class SpyGate(StubGate):
        def roi_crop_scored(self, key, xywh, img=None):
            seen.append(xywh)
            return super().roi_crop_scored(key, xywh, img)

    rec = JerseyNumberRecognizer(models={
        "gate": SpyGate([0.9, 0.9]), "leg": StubLeg([0.99, 0.99]),
        "read_many": reader_for([4, 4])}, stride=1)
    n, conf, used = rec.predict(imgs[:2])
    assert (n, conf, used) == ("4", 1.0, 2) and seen == [_FULL, _FULL]

    # 4) stride honoured: 6 frames @ stride 5 -> frames 0 and 5 only
    g = StubGate([0.9, 0.9])
    rec = JerseyNumberRecognizer(models={
        "gate": g, "leg": StubLeg([0.99, 0.99]),
        "read_many": reader_for([2, 2])}, stride=5)
    n, _, used = rec.predict(imgs)
    assert (n, used, g.calls) == ("2", 2, 2)

    # 5) two-digit decode through the real vote
    rec = JerseyNumberRecognizer(models={
        "gate": StubGate([0.9]), "leg": StubLeg([0.99]),
        "read_many": lambda rois: [(_logl(1), _logl(0))]}, stride=1)
    assert rec.predict([imgs[0]])[0] == "10"

    print("jn_recognizer: all self-tests passed (strict gates at 0.9/0.52, "
          "gate order, maxconf default + wvote + vote shares, -1 path, "
          "pre-cropped sentinel, "
          "stride, two-digit decode)")
