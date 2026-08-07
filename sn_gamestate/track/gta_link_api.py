"""GTA-Link offline tracklet stitching as a TrackLab ``VideoLevelModule``.

Faithful port of the validated notebook (§11d.5). For one video it:

1. Re-extracts a sports-OSNet appearance feature for every detection crop.
2. Builds one EMA-averaged, L2-normalised embedding per tracklet (``track_id``).
3. Computes pairwise cosine distance, then *forbids* merging any pair that overlaps in
   time, and gates temporally-disjoint pairs by a spatial cut (centre gap must be below
   ``spatial_thresh * sqrt(frame gap)``).
4. Runs agglomerative clustering on the gated distance matrix and re-assigns ``track_id``
   so each cluster shares one id.

Per-frame collision guard
-------------------------
Step 3 forbids merging tracklets that *directly* overlap in time, but agglomerative
average-linkage can still place two time-overlapping tracklets in the same cluster
*transitively* (A–B and B–C both merge, dragging A and C together even though A and C
share a frame). When that happens the relabelling would put two boxes with the same id
in one frame. Following the notebook, we resolve any such ``(image_id, track_id)``
collision by keeping the detection from the longer-lived original tracklet and setting
the colliding row's ``track_id`` to NaN (equivalent to the notebook's row drop). NaN
track_ids are handled downstream (the team visualiser skips them, pandas ``groupby``
drops them, and the SoccerNet eval encoding drops them), so this stays coherent.

Design choice: this runs as its own pipeline stage placed right after ``track`` and uses its
own OSNet (Option B), leaving the prtreid ``reid`` module — consumed by team/role/jersey —
completely untouched.

TensorRT
--------
If ``use_tensorrt`` is set and ``trt_engine`` points at a built ``.engine`` file, the OSNet
forward pass runs through a TensorRT engine (same 256×128 preprocessing; only the network
matmuls move to TRT). If the engine is missing or TensorRT can't be imported, it logs a
warning and falls back to the PyTorch OSNet, so the pipeline never hard-fails.
"""
import logging
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.cluster import AgglomerativeClustering

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.cv2 import cv2_load_image

# These ship with tracklab: its pyproject packages plugins/track as top-level packages
# (`bot_sort`, `strong_sort`, ...). Using the same loader as the BoT-SORT ReID backend
# keeps GTA-Link's appearance model byte-identical to the tracker's.
from strong_sort.deep.models import build_model
from strong_sort.deep.reid_model_factory import load_pretrained_weights

log = logging.getLogger(__name__)


class _TRTReID:
    """Minimal TensorRT runner for the sports-OSNet ReID engine.

    Consumes the *already preprocessed* [B, 3, 256, 128] tensor that GTA-Link builds
    (so preprocessing is identical to the PyTorch path) and returns [B, feat] features.
    Mirrors the TRT-8.x bindings API used by tracklab's ReIDDetectMultiBackend
    (``num_bindings`` / ``get_binding_*`` / ``set_binding_shape`` / ``execute_v2``),
    chunking to the engine's max profile batch so any GTA batch size stays valid.
    """

    def __init__(self, engine_path: str, device):
        import tensorrt as trt  # lazy import; absence handled by try_load

        self._np2torch = {np.float16: torch.float16, np.float32: torch.float32}
        self.device = (
            device if str(device) != "cpu" else torch.device("cuda:0")
        )
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.in_idx = self.engine.get_binding_index("images")
        self.out_idx = self.engine.get_binding_index("output")
        self.in_np = trt.nptype(self.engine.get_binding_dtype(self.in_idx))
        self.out_np = trt.nptype(self.engine.get_binding_dtype(self.out_idx))
        # profile: (min, opt, max) shapes for the dynamic input; take max batch.
        self.max_batch = int(self.engine.get_profile_shape(0, self.in_idx)[2][0])
        self.n_bindings = self.engine.num_bindings

    @classmethod
    def try_load(cls, engine_path, device):
        p = Path(str(engine_path or ""))
        if p.suffix != ".engine" or not p.is_file():
            log.warning(
                f"[GTA-Link] use_tensorrt=true but engine not found at '{p}'. "
                f"Using PyTorch OSNet. Build it with scripts/build_trt_engines.py."
            )
            return None
        try:
            runner = cls(str(p), device)
            log.info(f"[GTA-Link] ReID via TensorRT engine: {p}")
            return runner
        except Exception as e:  # missing tensorrt, version mismatch, bad engine, ...
            log.warning(
                f"[GTA-Link] failed to load TensorRT engine ({e}); "
                f"using PyTorch OSNet."
            )
            return None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().to(self.device)
        x = x.half() if self.in_np == np.float16 else x.float()
        out_torch_dtype = self._np2torch.get(self.out_np, torch.float32)
        outs = []
        for s in range(0, x.shape[0], self.max_batch):
            chunk = x[s:s + self.max_batch].contiguous()
            b = chunk.shape[0]
            self.context.set_binding_shape(
                self.in_idx, (b, chunk.shape[1], chunk.shape[2], chunk.shape[3])
            )
            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
            out = torch.empty(out_shape, dtype=out_torch_dtype, device=self.device)
            bindings = [0] * self.n_bindings
            bindings[self.in_idx] = int(chunk.data_ptr())
            bindings[self.out_idx] = int(out.data_ptr())
            self.context.execute_v2(bindings)
            outs.append(out.float())
        return torch.cat(outs, dim=0)


class GTALink(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id"]
    output_columns = ["track_id"]

    def __init__(self, cfg, device, tracking_dataset=None):
        self.cfg = cfg
        self.device = device
        self.fp16 = bool(getattr(cfg, "fp16", False)) and (str(device) != "cpu")

        # Build osnet_x1_0 and load the sports checkpoint the torchreid way:
        # load_pretrained_weights strips 'module.' and shape-filters the classifier,
        # matching the notebook's manual loader exactly.
        self.model = build_model(
            "osnet_x1_0", num_classes=1, pretrained=False,
            use_gpu=(str(device) != "cpu"),
        )
        load_pretrained_weights(self.model, cfg.osnet_weights)
        self.model.classifier = nn.Identity()
        self.model.to(device).eval()
        if self.fp16:
            self.model.half()

        # Optional TensorRT OSNet (falls back to the PyTorch model above on any failure).
        self._trt = None
        if bool(getattr(cfg, "use_tensorrt", False)):
            self._trt = _TRTReID.try_load(getattr(cfg, "trt_engine", None), device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.appearance_thresh = float(cfg.appearance_thresh)
        self.spatial_thresh = float(cfg.spatial_thresh)
        self.min_tracklet_len = int(cfg.min_tracklet_len)
        self.ema_alpha = float(cfg.ema_alpha)
        self.linkage = getattr(cfg, "linkage", "average")
        self.batch_size = int(getattr(cfg, "batch_size", 64))

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _extract_features(self, dets: pd.DataFrame, metadatas: pd.DataFrame) -> np.ndarray:
        """OSNet feature per detection row, aligned to ``dets.index`` (zeros on failure)."""
        feats = np.zeros((len(dets), 512), dtype=np.float32)
        id2path = (metadatas["file_path"].to_dict()
                   if "file_path" in metadatas.columns else {})
        pos = {idx: i for i, idx in enumerate(dets.index)}

        for image_id, group in dets.groupby("image_id"):
            path = id2path.get(image_id)
            if path is None:
                continue
            img = cv2_load_image(path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            batch, rows = [], []

            def _flush():
                if not batch:
                    return
                x = torch.stack(batch).to(self.device)
                if self._trt is not None:
                    raw = self._trt(x)                     # TRT handles dtype/dynamic shape
                else:
                    xx = x.half() if self.fp16 else x
                    raw = self.model(xx)
                out = raw.detach().cpu().float().numpy()
                out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
                for r, f in zip(rows, out):
                    feats[pos[r]] = f
                batch.clear()
                rows.clear()

            for idx, det in group.iterrows():
                l, t, w, h = [float(v) for v in det["bbox_ltwh"]]
                x1, y1 = max(0, int(l)), max(0, int(t))
                x2, y2 = min(w_img, int(l + w)), min(h_img, int(t + h))
                if x2 <= x1 or y2 <= y1:
                    continue  # leave zeros for degenerate boxes
                # cv2_load_image already returns RGB, and both the torchvision
                # transform and OSNet expect RGB — so crop the RGB image directly
                # (an extra BGR2RGB here would feed the model swapped channels).
                crop = img[y1:y2, x1:x2]
                batch.append(self.transform(crop))
                rows.append(idx)
                if len(batch) >= self.batch_size:
                    _flush()
            _flush()
        return feats

    # ------------------------------------------------------------------ clustering
    def _cluster(self, dist: np.ndarray):
        # sklearn renamed `affinity` -> `metric` in 1.2; support both.
        kw = dict(n_clusters=None, distance_threshold=self.appearance_thresh,
                  linkage=self.linkage)
        try:
            return AgglomerativeClustering(metric="precomputed", **kw).fit_predict(dist)
        except TypeError:
            return AgglomerativeClustering(affinity="precomputed", **kw).fit_predict(dist)

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0 or "track_id" not in detections.columns:
            return detections
        work = detections[detections["track_id"].notna()].copy()
        if work["track_id"].nunique() < 2:
            return detections

        feats = self._extract_features(work, metadatas)
        order = {idx: i for i, idx in enumerate(work.index)}

        # One EMA embedding + temporal/spatial extent per tracklet (>= min_tracklet_len).
        embs = OrderedDict()
        for tid, g in work.groupby("track_id"):
            g = g.sort_values("image_id")
            rows = [order[i] for i in g.index]
            if len(rows) < self.min_tracklet_len:
                continue
            gf = feats[rows]
            emb = gf[0].copy()
            for v in gf[1:]:
                emb = self.ema_alpha * emb + (1.0 - self.ema_alpha) * v
                emb /= (np.linalg.norm(emb) + 1e-6)
            first = np.asarray(g["bbox_ltwh"].iloc[0], dtype=float)
            last = np.asarray(g["bbox_ltwh"].iloc[-1], dtype=float)
            embs[tid] = (emb,
                         int(g["image_id"].iloc[0]), int(g["image_id"].iloc[-1]),
                         first, last)

        if len(embs) < 2:
            return detections

        tids = list(embs)
        E = np.stack([embs[t][0] for t in tids])
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        dist = 1.0 - E @ E.T
        n = len(tids)

        forbidden = np.zeros((n, n), dtype=bool)
        for i in range(n):
            _, si, ei, _, last_i = embs[tids[i]]
            for j in range(n):
                if i == j:
                    forbidden[i, j] = True
                    continue
                _, sj, ej, first_j, _ = embs[tids[j]]
                if not (ei < sj or ej < si):          # overlap in time -> never merge
                    forbidden[i, j] = True
                elif ei < sj:                          # i strictly before j -> spatial gate
                    a = last_i[:2] + last_i[2:] / 2.0  # last centre of i
                    b = first_j[:2] + first_j[2:] / 2.0  # first centre of j
                    gap = max(1.0, abs(sj - ei) ** 0.5)
                    if np.linalg.norm(a - b) > self.spatial_thresh * gap:
                        forbidden[i, j] = True

        gated = dist.copy()
        gated[forbidden] = 1.0
        np.fill_diagonal(gated, 0.0)
        labels = self._cluster(gated)

        # Collision-free relabelling: clustered tracklets share a fresh id; short/unclustered
        # tracklets each get their own fresh id. (NaN track_ids stay NaN.)
        next_id, cluster_to_new, id_map = 1, {}, {}
        for t, lab in zip(tids, labels):
            if lab not in cluster_to_new:
                cluster_to_new[lab] = next_id
                next_id += 1
            id_map[t] = cluster_to_new[lab]
        for tid in detections["track_id"].dropna().unique():
            if tid not in id_map:
                id_map[tid] = next_id
                next_id += 1

        out = detections.copy()
        out["track_id"] = detections["track_id"].map(id_map)

        # Per-frame collision guard (see module docstring): if transitive clustering put
        # two time-overlapping tracklets under one new id, keep the detection from the
        # longer-lived original tracklet in each colliding frame and NaN the rest.
        n_nulled = self._dedup_per_frame(detections, out)

        n_before = int(detections["track_id"].dropna().nunique())
        n_after = int(out["track_id"].dropna().nunique())
        log.info(
            f"[GTA-Link] tracklets {n_before} -> {n_after} "
            f"(merged {n_before - n_after}); per-frame de-dup nulled {n_nulled} detection(s)"
        )
        return out

    # ------------------------------------------------------------------ de-dup
    @staticmethod
    def _dedup_per_frame(detections: pd.DataFrame, out: pd.DataFrame) -> int:
        """NaN colliding (image_id, new track_id) rows, keeping the longest-lived original.

        Returns the number of detection rows set to NaN. Mutates ``out`` in place.
        """
        orig = detections["track_id"]
        keep = orig.notna() & out["track_id"].notna()
        if not keep.any():
            return 0

        img = detections["image_id"]
        # Original-tracklet lifetime (frame span) and support (detection count), keyed by
        # the ORIGINAL track_id, for deterministic keeper selection on collisions.
        img_int = img[orig.notna()].astype(int)
        life, count = {}, {}
        for tid, g in img_int.groupby(orig[orig.notna()]):
            life[tid] = int(g.max() - g.min() + 1)
            count[tid] = int(g.shape[0])

        frame = pd.DataFrame(
            {"orig": orig[keep].values,
             "new": out["track_id"][keep].values,
             "img": img[keep].astype(str).values},
            index=detections.index[keep],
        )

        to_nan = []
        for _, g in frame.groupby(["img", "new"]):
            if len(g) <= 1:
                continue
            best_idx, best_key = None, None
            for idx, row in g.iterrows():
                o = row["orig"]
                key = (life.get(o, 0), count.get(o, 0), -float(o))  # max life, count; min id
                if best_key is None or key > best_key:
                    best_key, best_idx = key, idx
            to_nan.extend([idx for idx in g.index if idx != best_idx])

        if to_nan:
            out.loc[to_nan, "track_id"] = np.nan
        return len(to_nan)
