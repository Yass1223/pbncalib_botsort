#!/usr/bin/env python
r"""Verify PnLCalib runs under the PIPELINE's torch, before any module is written.

PnLCalib's own requirements.txt asks for torch 2.3.1 / numpy 2.0.2 /
torchvision 0.18.1. The pipeline pins torch 1.13.1 / numpy 1.26.4 /
torchvision 0.14.1 and cannot move: the detector, reid and tracking stages
require that torch. This script tests whether PnLCalib works anyway, on the
pipeline's pins, WITHOUT changing them.

Static analysis says it should: model/cls_hrnet.py and model/cls_hrnet_l.py use
only nn.Conv2d/BatchNorm2d/ReLU/Sequential/ModuleList/Upsample/Sigmoid/Softmax,
nn.init, F.interpolate and torch.cat -- all long predating 1.13. But the two
real unknowns cannot be settled by reading:

  * the released v1.0.0 checkpoints were saved by torch 2.3.1; can 1.13.1 read
    that archive at all?
  * do the state_dict keys match the architecture built from hrnetv2_w48.yaml?

Checks (each reported independently, so a failure localises)
-----------------------------------------------------------
A. environment    - python / torch / numpy / torchvision actually in use.
                    Loud WARN if torch is not 1.13.1: the answer is then about
                    some other environment and does not transfer.
B. imports        - shapely and matplotlib (both imported at MODULE level by
                    utils/utils_optimize.py), then the decisive one:
                    `from utils.utils_calib import FramebyFrameCalib`.
C. architecture   - build cls_hrnet + cls_hrnet_l from the repo's own yamls;
                    report parameter counts and final-layer channel counts
                    against MODEL.NUM_JOINTS (58 keypoint / 24 line planes).
D. checkpoint io  - torch.load each checkpoint. This is the torch 2.3.1 ->
                    1.13.1 question; the exact exception is reported.
E. state_dict     - load_state_dict(strict=True). On mismatch, missing /
                    unexpected / wrong-shape keys are printed, not summarised.
F. forward        - one forward pass on a REAL frame, mirroring PnLCalib's own
                    inference.py preprocessing exactly. Output shapes asserted.
G. calibration    - the full detection -> heuristic_voting path. This is the
                    actual integration surface and the only check that really
                    exercises shapely. Reports detected keypoint/line counts.
H. sn round-trip  - OPTIONAL: feed cam_params straight into
                    sn_calibration_baseline.Camera.from_json_parameters. SKIPs
                    if the plugin is not installed. Confirms on real output the
                    audit's conclusion that PnLCalib's cam_params already IS an
                    sn-calibration parameter dict.

Nothing here needs a GPU; --device cuda:0 is available but cpu is the default.

Usage
-----
    python scripts/verify_pnlcalib_env.py \
        --repo /path/to/PnLCalib \
        --weights-kp weights/SV_kp \
        --weights-line weights/SV_lines \
        --frame data/SoccerNetGS/test/SNGS-116/img1/000001.jpg

Exit code 0 means PnLCalib is usable on the pipeline's pins. Non-zero is the
number of failed checks.
"""
import argparse
import sys
import traceback
from pathlib import Path

EXPECTED = {"torch": "1.13.1", "numpy": "1.26.4", "torchvision": "0.14.1"}

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[2m", "\033[1m", "\033[0m"
)

_results = []


def hdr(title):
    print(f"\n{BOLD}{title}{RESET}")


def ok(msg):
    print(f"  {GREEN}OK  {RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}WARN{RESET} {msg}")


def info(msg):
    print(f"       {DIM}{msg}{RESET}")


def fail(msg, exc=None, verbose=False):
    print(f"  {RED}FAIL{RESET} {msg}")
    if exc is not None:
        info(f"{type(exc).__name__}: {exc}")
        if verbose:
            print(f"{DIM}{traceback.format_exc()}{RESET}")


def record(name, passed):
    _results.append((name, passed))
    return passed


# --------------------------------------------------------------------------- A
def check_environment():
    hdr("A. environment")
    import numpy
    import torch
    print(f"       python      {sys.version.split()[0]}")
    print(f"       torch       {torch.__version__}")
    print(f"       numpy       {numpy.__version__}")
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:
        tv = "<not importable>"
    print(f"       torchvision {tv}")

    base = torch.__version__.split("+")[0]
    if base != EXPECTED["torch"]:
        warn(f"torch is {base}, not the pipeline's {EXPECTED['torch']}. "
             "This run does NOT answer the integration question -- rerun "
             "inside the pipeline venv (uv venv --python 3.9 .venv).")
        return record("environment", False)
    ok(f"torch {base} — the pipeline's pin")
    if numpy.__version__ != EXPECTED["numpy"]:
        warn(f"numpy is {numpy.__version__}, pipeline pins {EXPECTED['numpy']}")
    return record("environment", True)


# --------------------------------------------------------------------------- B
def check_imports(repo, verbose):
    hdr("B. imports (the new dependencies, then the entry point)")
    passed = True
    for mod, why in [("shapely", "utils_optimize: from shapely.geometry import ..."),
                     ("matplotlib", "utils_optimize: import matplotlib.pyplot")]:
        try:
            m = __import__(mod)
            ok(f"{mod} {getattr(m, '__version__', '?')}  ({why})")
        except Exception as exc:
            fail(f"{mod} missing — {why}", exc, verbose)
            passed = False

    sys.path.insert(0, str(repo))
    try:
        from utils.utils_calib import FramebyFrameCalib  # noqa: F401
        ok("from utils.utils_calib import FramebyFrameCalib")
    except Exception as exc:
        fail("utils.utils_calib is not importable — this is the entry point "
             "for every calibration call", exc, verbose)
        passed = False
    return record("imports", passed)


# --------------------------------------------------------------------------- C
def build_models(repo, verbose):
    hdr("C. architecture (built from the repo's own yamls)")
    import yaml
    # Same import spelling as PnLCalib's own inference.py.
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l

    out = {}
    for tag, yml, builder, planes in [
        ("kp", "hrnetv2_w48.yaml", get_cls_net, 58),
        ("line", "hrnetv2_w48_l.yaml", get_cls_net_l, 24),
    ]:
        try:
            cfg = yaml.safe_load(open(repo / "config" / yml))
            declared = cfg["MODEL"]["NUM_JOINTS"]
            model = builder(cfg)
            n = sum(p.numel() for p in model.parameters())
            ok(f"{tag:4s} {yml}  NUM_JOINTS={declared}  params={n/1e6:.1f}M")
            if declared != planes:
                warn(f"{tag}: expected NUM_JOINTS={planes}, yaml says {declared}")
            out[tag] = (model, cfg)
        except Exception as exc:
            fail(f"could not build {tag} model from {yml}", exc, verbose)
            out[tag] = None
    return record("architecture", all(v is not None for v in out.values())), out


# ------------------------------------------------------------------------- D+E
def load_weights(models, kp_path, line_path, device, verbose):
    hdr("D. checkpoint io (torch 2.3.1 archive -> this torch)")
    import torch
    loaded = {}
    d_ok = True
    for tag, path in [("kp", kp_path), ("line", line_path)]:
        p = Path(path)
        if not p.is_file():
            fail(f"{tag}: {p} does not exist")
            d_ok = False
            continue
        try:
            # No weights_only= : that kwarg's default differs across the torch
            # versions this must run on, and these are bare tensor state_dicts.
            sd = torch.load(str(p), map_location=device)
            if not hasattr(sd, "keys"):
                fail(f"{tag}: loaded object is {type(sd).__name__}, not a state_dict")
                d_ok = False
                continue
            if "state_dict" in sd:
                sd = sd["state_dict"]
            ok(f"{tag:4s} {p.name}  {p.stat().st_size/1e6:.1f} MB  "
               f"{len(sd)} tensors")
            loaded[tag] = sd
        except Exception as exc:
            fail(f"{tag}: torch.load failed on {p.name} — this is the "
                 "torch-version question, and the answer is no", exc, verbose)
            d_ok = False
    record("checkpoint io", d_ok)

    hdr("E. state_dict match")
    e_ok = True
    for tag in ("kp", "line"):
        if models.get(tag) is None or tag not in loaded:
            warn(f"{tag}: skipped (earlier check failed)")
            e_ok = False
            continue
        model = models[tag][0]
        try:
            model.load_state_dict(loaded[tag])
            ok(f"{tag:4s} strict load, every key matched")
        except RuntimeError as exc:
            fail(f"{tag}: state_dict does not match the built architecture", exc,
                 verbose)
            have, want = set(loaded[tag]), set(model.state_dict())
            for label, keys in [("missing", want - have), ("unexpected", have - want)]:
                if keys:
                    shown = sorted(keys)[:8]
                    info(f"{label} ({len(keys)}): {', '.join(shown)}"
                         + (" ..." if len(keys) > 8 else ""))
            msd = model.state_dict()
            bad = [k for k in have & want if msd[k].shape != loaded[tag][k].shape]
            for k in bad[:8]:
                info(f"shape {k}: ckpt {tuple(loaded[tag][k].shape)} "
                     f"vs model {tuple(msd[k].shape)}")
            e_ok = False
    return record("state_dict", e_ok)


# --------------------------------------------------------------------------- F
def forward_pass(models, frame_path, device, verbose):
    hdr("F. forward pass on a real frame")
    import cv2
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image

    p = Path(frame_path)
    if not p.is_file():
        fail(f"frame not found: {p}")
        return record("forward", False), None
    bgr = cv2.imread(str(p))
    if bgr is None:
        fail(f"cv2 could not decode {p}")
        return record("forward", False), None
    h0, w0 = bgr.shape[:2]
    info(f"{p.name}  {w0}x{h0}")

    # Mirrors PnLCalib inference.py exactly.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = TF.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
    if t.size()[-1] != 960:
        t = T.Resize((540, 960))(t)
    t = t.to(device)

    heat = {}
    for tag, planes in [("kp", 58), ("line", 24)]:
        if models.get(tag) is None:
            warn(f"{tag}: skipped (no model)")
            return record("forward", False), None
        model = models[tag][0].to(device).eval()
        try:
            with torch.no_grad():
                out = model(t)
            shape = tuple(out.shape)
            if shape[1] != planes:
                warn(f"{tag}: {shape[1]} output planes, expected {planes}")
            ok(f"{tag:4s} input {tuple(t.shape)} -> output {shape}")
            heat[tag] = out
        except Exception as exc:
            fail(f"{tag}: forward pass raised", exc, verbose)
            return record("forward", False), None
    return record("forward", True), (heat, t, (w0, h0))


# --------------------------------------------------------------------------- G
def calibrate(repo, ctx, verbose, kp_th=0.3434, line_th=0.7867):
    hdr("G. full detection -> heuristic_voting path")
    if ctx is None:
        warn("skipped (no forward output)")
        return record("calibration", False), None
    heat, t, (w0, h0) = ctx
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (complete_keypoints, coords_to_dict,
                                     get_keypoints_from_heatmap_batch_maxpool,
                                     get_keypoints_from_heatmap_batch_maxpool_l)
    try:
        _, _, hn, wn = t.size()
        kp = get_keypoints_from_heatmap_batch_maxpool(heat["kp"][:, :-1, :, :])
        ln = get_keypoints_from_heatmap_batch_maxpool_l(heat["line"][:, :-1, :, :])
        kp_d = coords_to_dict(kp, threshold=kp_th)
        ln_d = coords_to_dict(ln, threshold=line_th)
        kp_d, ln_d = complete_keypoints(kp_d[0], ln_d[0], w=wn, h=hn, normalize=True)
        ok(f"detections: {len(kp_d)} keypoints, {len(ln_d)} lines "
           f"(thresholds {kp_th} / {line_th})")
        if len(kp_d) < 4:
            warn("fewer than 4 keypoints — this frame may simply be a hard view; "
                 "that is a data question, not an environment failure")

        engine = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
        engine.update(kp_d, ln_d)
        final = engine.heuristic_voting(refine_lines=False)
        if final is None:
            warn("heuristic_voting returned None (abstained on this frame). "
                 "The path RAN — shapely and the optimiser are fine — but try "
                 "another frame to see a camera.")
            return record("calibration", True), None
        cp = final["cam_params"]
        ok(f"camera solved: mode={final['mode']} rep_err={final['rep_err']:.2f} px")
        info(f"f=({cp['x_focal_length']:.1f}, {cp['y_focal_length']:.1f})  "
             f"pp={[round(v, 1) for v in cp['principal_point']]}")
        info(f"pan/tilt/roll deg = ({cp['pan_degrees']:.2f}, "
             f"{cp['tilt_degrees']:.2f}, {cp['roll_degrees']:.2f})")
        info(f"position_meters = {[round(v, 2) for v in cp['position_meters']]}")
        return record("calibration", True), cp
    except Exception as exc:
        fail("detection/calibration path raised", exc, verbose)
        return record("calibration", False), None


# --------------------------------------------------------------------------- H
def sn_roundtrip(cam_params, verbose):
    hdr("H. sn-calibration round-trip (optional)")
    if cam_params is None:
        warn("skipped (no camera to convert)")
        return
    try:
        from sn_calibration_baseline.camera import Camera
    except Exception as exc:
        warn(f"sn_calibration_baseline not importable ({type(exc).__name__}) — "
             "skipped. Install the calibration plugin to enable this check.")
        return
    import numpy as np
    try:
        cam = Camera()
        cam.from_json_parameters(cam_params)
        pt = cam.unproject_point_on_planeZ0(
            np.array([cam_params["principal_point"][0],
                      cam_params["principal_point"][1], 1]))
        ok("cam_params consumed by Camera.from_json_parameters unchanged")
        info(f"image centre unprojects to pitch ({pt[0]:.2f}, {pt[1]:.2f}) m "
             "— expect |x|<60, |y|<40 for a main-camera view")
        if abs(pt[0]) > 60 or abs(pt[1]) > 40:
            warn("that is off the pitch; suspect this frame's camera, not the "
                 "conversion")
    except Exception as exc:
        fail("round-trip raised", exc, verbose)


# ------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path,
                    help="PnLCalib clone (the dir containing inference.py)")
    ap.add_argument("--weights-kp", required=True, help="v1.0.0 SV_kp checkpoint")
    ap.add_argument("--weights-line", required=True, help="v1.0.0 SV_lines checkpoint")
    ap.add_argument("--frame", required=True, help="a real broadcast frame (.jpg/.png)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda:0")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="full traceback for each failure")
    args = ap.parse_args(argv)

    if not (args.repo / "inference.py").is_file():
        sys.exit(f"not a PnLCalib checkout (no inference.py): {args.repo}")

    check_environment()
    check_imports(args.repo, args.verbose)
    _, models = build_models(args.repo, args.verbose)
    load_weights(models, args.weights_kp, args.weights_line, args.device,
                 args.verbose)
    _, ctx = forward_pass(models, args.frame, args.device, args.verbose)
    _, cam_params = calibrate(args.repo, ctx, args.verbose)
    sn_roundtrip(cam_params, args.verbose)

    hdr("summary")
    failed = [n for n, p in _results if not p]
    for name, passed in _results:
        print(f"  {GREEN + 'PASS' + RESET if passed else RED + 'FAIL' + RESET}  {name}")
    print()
    if not failed:
        print(f"{GREEN}PnLCalib runs on the pipeline's pins.{RESET} "
              "No torch change needed; proceed to the module.")
        return 0
    print(f"{RED}{len(failed)} check(s) failed: {', '.join(failed)}{RESET}")
    print(f"{YELLOW}Do NOT bump the pipeline's torch to fix this — the "
          f"detector, reid and tracking stages require 1.13.1. Report the "
          f"failure and decide deliberately.{RESET}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
