"""SoccerNet-conformant YOLO wrapper (notebook §11c parity) with optional TensorRT.

tracklab's stock ``YOLOUltralytics.process`` calls ``self.model(images, verbose=False)``,
which silently runs ultralytics defaults: ``imgsz=640``, ``conf=0.25``, ``iou=0.7``.
The validated notebook (yolov11-bot-sort-cmc.ipynb §11c) ran
``model(path, conf=0.001..., iou=0.7, imgsz=1280)`` and floored the tracker feed at
``conf >= 0.1``. Two silent mismatches follow from the stock wrapper:

* **imgsz 640 vs 1280** — materially different detections on small, distant players;
* **conf floor 0.25 vs 0.1** — the BYTE low-score band [track_low_thresh,
  track_high_thresh) = [0.05..0.1, 0.3) that BoT-SORT's second association depends on
  is almost empty at a 0.25 floor.

A third one is color: tracklab feeds **RGB** numpy arrays (``cv2_load_image`` converts
BGR→RGB), while ultralytics assumes **BGR** for raw numpy input. The notebook passed
file paths, so ultralytics loaded BGR itself. We convert back to BGR before predict.

This subclass fixes all three and adds an optional TensorRT engine path
(``use_tensorrt`` + ``engine_path``): ultralytics loads ``.engine`` files natively
through the same ``YOLO()`` API. Missing engine → warning + PyTorch fallback, so the
pipeline never hard-fails.
"""
import logging

import cv2
import torch
import pandas as pd
from pathlib import Path

from ultralytics import YOLO

from tracklab.utils.coordinates import ltrb_to_ltwh
from tracklab.wrappers.bbox_detector.yolo_ultralytics_api import YOLOUltralytics

log = logging.getLogger(__name__)


class YOLOUltralyticsSNFT(YOLOUltralytics):
    """YOLO11-L SoccerNet fine-tune with notebook-parity inference settings."""

    # tracklab's Module.level is a property that derives the level from the name of
    # the DIRECT base class: ImageLevelModule -> "image", DetectionLevelModule ->
    # "detection", VideoLevelModule -> "video". Because this class subclasses the
    # concrete YOLOUltralytics rather than ImageLevelModule, that derivation yields
    # "yoloultralytics" and EngineDatapipe.__len__ raises ValueError. Declaring level
    # explicitly shadows the inherited property. YOLOUltralytics is an ImageLevelModule.
    level = "image"

    def __init__(self, cfg, device, batch_size, **kwargs):
        # Deliberately NOT calling YOLOUltralytics.__init__ (it always loads the .pt
        # and calls .to(device), which is invalid for TensorRT engines). We replicate
        # its few lines with engine-aware loading instead.
        super(YOLOUltralytics, self).__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.id = 0

        self.imgsz = int(getattr(cfg, "imgsz", 1280))          # notebook INFER_IMGSZ
        self.iou = float(getattr(cfg, "iou", 0.7))             # notebook INFER_IOU
        self.max_det = int(getattr(cfg, "max_det", 300))       # ultralytics default,
        # also what the notebook ran with (it never overrode max_det).

        weights = str(cfg.path_to_checkpoint)
        self._is_engine = False
        if bool(getattr(cfg, "use_tensorrt", False)):
            engine = Path(str(getattr(cfg, "engine_path", "") or ""))
            if engine.suffix == ".engine" and engine.is_file():
                weights = str(engine)
                self._is_engine = True
                log.info(f"[YOLO-SNFT] TensorRT engine: {engine}")
            else:
                log.warning(
                    f"[YOLO-SNFT] use_tensorrt=true but engine not found at "
                    f"'{engine}'. Falling back to PyTorch checkpoint. Build it "
                    f"with scripts/build_trt_engines.py."
                )

        self.model = YOLO(weights, task="detect")
        if not self._is_engine:
            self.model.to(device)

        self._logged_classes = False
        self._log_checkpoint_args()

    def _log_checkpoint_args(self):
        """Report the size the checkpoint was TRAINED at, against cfg.imgsz.

        Inference at a different imgsz than training degrades small-object recall
        without raising -- distant players are exactly the population at risk on
        broadcast footage, and the failure is a quieter detection set, not an
        error. ultralytics stores the training arguments on the loaded model, so
        the comparison is free; a TensorRT engine may not carry them, hence the
        broad guard.
        """
        try:
            args = getattr(getattr(self.model, "model", None), "args", None) or {}
            trained = args.get("imgsz") if hasattr(args, "get") else getattr(args, "imgsz", None)
        except Exception as exc:                      # engine / unexpected layout
            log.info(f"[YOLO-SNFT] could not read training args ({exc}); "
                     f"running at imgsz={self.imgsz}")
            return
        if trained is None:
            log.info(f"[YOLO-SNFT] checkpoint carries no training imgsz "
                     f"(TensorRT engine?); running at imgsz={self.imgsz}")
            return
        if int(trained) != int(self.imgsz):
            log.warning(
                f"[YOLO-SNFT] checkpoint was TRAINED at imgsz={trained} but "
                f"inference runs at imgsz={self.imgsz}. This does not raise; it "
                f"silently changes which small/distant players are detected. "
                f"Either set cfg.imgsz={trained} or accept the mismatch "
                f"deliberately."
            )
        else:
            log.info(f"[YOLO-SNFT] imgsz {self.imgsz} matches the checkpoint's "
                     f"training size")

    def reset(self):
        """Called once per video by the engine (offline.py:11-13, hasattr-guarded)."""
        self._logged_classes = False
        parent_reset = getattr(super(), "reset", None)
        if callable(parent_reset):
            parent_reset()

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        # tracklab images are RGB (cv2_load_image); ultralytics expects BGR numpy.
        bgr_images = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images]
        results_by_image = self.model(
            bgr_images,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.cfg.min_confidence,   # low floor: BYTE band reaches the tracker
            iou=self.iou,
            max_det=self.max_det,
        )
        if not self._logged_classes:
            # Which classes does this checkpoint actually emit? The cls == 0
            # filter below silently discards everything else, with no count. If
            # the fine-tune turns out to be multi-class (player / goalkeeper /
            # referee / ball) that drops two of the three evaluated roles and
            # still produces a completely plausible run. Logged once per video.
            try:
                seen = sorted({int(b.cls) for r in results_by_image
                               for b in r.boxes.cpu().numpy()})
                names = getattr(self.model, "names", None)
                log.info(f"[YOLO-SNFT] class ids present in raw output: {seen}; "
                         f"model.names={names}")
                if seen and seen != [0]:
                    log.warning(
                        f"[YOLO-SNFT] checkpoint emits classes {seen} but only "
                        f"cls==0 is kept. Every other class is discarded here "
                        f"without a count. If those are goalkeeper/referee, the "
                        f"pipeline is losing evaluated roles silently -- confirm "
                        f"the intended mapping before trusting any metric."
                    )
            except Exception as exc:              # never let logging kill a run
                log.debug(f"[YOLO-SNFT] class-id probe failed: {exc}")
            self._logged_classes = True

        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # Single-class person fine-tune.
                #
                # NOTE the `>=` here is cosmetic, not load-bearing: ultralytics
                # already applied `conf` inside non_max_suppression with a STRICT
                # `>`, so boxes at exactly the floor never reach this loop. The
                # effective floor is `> min_confidence`, not `>= min_confidence`.
                # Kept for symmetry with the tracker's own pre-filter, which is
                # genuinely `>=` because nothing upstream of it has filtered.
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
