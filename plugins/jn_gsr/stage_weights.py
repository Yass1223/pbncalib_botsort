#!/usr/bin/env python3
"""stage_weights.py -- resolve the PARSeq warm-start checkpoint, verifiably.

One canonical output (default models/koshkina_sn_parseq.ckpt -- the name
warm_start.py's --init has always defaulted to), produced by the first source
that works:

  1. --attached <path>   a file the user attached as a Kaggle dataset (fastest,
                         reproducible, immune to outages) -- preferred;
  2. HF  Ynniss/parseq_soccernet_epoch24
         'parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt'
         (the exact upstream filename);
  3. HF  Ynniss/paresq_soccernet  'parseq_epoch_24.ckpt'
         (byte-identical second mirror: both HF blobs publish the same SHA256,
         verified 2026-07-26 -- guards a single-repo outage);
  4. Google Drive id 1uRln22tlhneVt3P6MePmVxBWSLMsL3bm via gdown
         (Koshkina's canonical link; last because Drive publishes no checksum,
         rate-limits large public files, and gdown breaks when the confirm
         interstitial changes).

HF is tried before Drive deliberately: huggingface_hub is already a pipeline
dependency proven working on Kaggle (it fetches the SoccerNet YOLO), and HF
publishes a SHA256 the download can be verified against; Drive offers neither.

After ANY source wins:
  * sha256 is computed and compared against the published upstream hash --
    reported, NOT fatal on mismatch (a mismatch means the user supplied a
    legitimately different checkpoint, e.g. a private fine-tune; the run must
    say so, not refuse);
  * a LOAD GATE: the file must load through strhub's loader (the exact code
    path warm_start.py and stage_test.py use). A file that fails here would
    have crashed the pipeline hours later mid-stage -- fail now instead;
  * the checkpoint's own internal metadata (epoch, global_step, whether
    optimizer state is present, PARSeq hparams) is read FROM THE FILE and
    written to weights_provenance.json next to the output. Filenames lie;
    the file's contents are the ground truth about what is being warm-started.

    $PY stage_weights.py --attached "/kaggle/input/mydata/parseq*.ckpt-match" \\
        --out models/koshkina_sn_parseq.ckpt
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

# Before ANY import that can reach matplotlib (strhub -> pytorch_lightning ->
# torchmetrics -> matplotlib). Kaggle presets MPLBACKEND to its inline backend,
# whose module is absent from this venv; matplotlib raises at import time.
os.environ["MPLBACKEND"] = "Agg"

# Published by both HF mirrors (verified from the blob pages, 2026-07-26).
UPSTREAM_SHA256 = "14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879"
UPSTREAM_NAME = ("parseq_epoch=24-step=2575-val_accuracy=95.6044"
                 "-val_NED=96.3255.ckpt")
HF_A = ("Ynniss/parseq_soccernet_epoch24", UPSTREAM_NAME)
HF_B = ("Ynniss/paresq_soccernet", "parseq_epoch_24.ckpt")
DRIVE_URL = "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm"


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ sources --
def src_attached(a):
    if not a.attached:
        raise RuntimeError("no --attached path given")
    if not os.path.exists(a.attached):
        raise RuntimeError(f"attached path does not exist: {a.attached}")
    shutil.copyfile(a.attached, a.out)
    return f"attached: {a.attached}"


def src_hf(repo, fname):
    def fetch(a):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, fname)
        shutil.copyfile(p, a.out)
        return f"huggingface: {repo}/{fname}"
    return fetch


def src_gdown(a):
    import gdown
    got = gdown.download(DRIVE_URL, a.out, quiet=False)   # user-specified snippet
    if not got or not os.path.exists(a.out):
        raise RuntimeError("gdown returned nothing (Drive quota/interstitial?)")
    return f"gdrive: {DRIVE_URL}"


# ---------------------------------------------------------------- load gate --
def inspect_and_gate(path):
    """Load through the SAME strhub path the pipeline uses, then read the
    checkpoint's own metadata. Everything here is best-effort-defensive:
    metadata fields vary by Lightning version, but the LOAD must succeed."""
    from strhub.models.utils import load_from_checkpoint
    model = load_from_checkpoint(path)            # raises on a broken/wrong file
    meta = {}
    try:
        import torch
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict):
            meta["epoch"] = raw.get("epoch")
            meta["global_step"] = raw.get("global_step")
            meta["has_optimizer_state"] = bool(raw.get("optimizer_states"))
            hp = raw.get("hyper_parameters") or {}
            meta["hparams"] = {k: hp[k] for k in
                               ("charset_train", "img_size", "max_label_length")
                               if k in hp}
    except Exception as e:                        # metadata is optional; load isn't
        meta["metadata_note"] = f"unreadable ({type(e).__name__}: {e})"
    del model
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attached", default="",
                    help="path to a user-attached checkpoint (tried first)")
    ap.add_argument("--out", default="models/koshkina_sn_parseq.ckpt")
    ap.add_argument("--skip-load-gate", action="store_true",
                    help="hash + provenance only (used by offline tests where "
                         "the stub loader would accept any bytes anyway)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if os.path.exists(a.out):
        print(f"[weights] {a.out} already present -- verifying, not re-fetching")
        source = "already-present"
    else:
        chain = [("attached dataset", src_attached),
                 (f"HF {HF_A[0]}", src_hf(*HF_A)),
                 (f"HF {HF_B[0]}", src_hf(*HF_B)),
                 ("Google Drive (gdown)", src_gdown)]
        source, errors = None, []
        for name, fn in chain:
            try:
                print(f"[weights] trying {name} ...")
                source = fn(a)
                print(f"[weights] OK from {name}")
                break
            except Exception as e:
                msg = f"{name}: {type(e).__name__}: {e}"
                errors.append(msg)
                print(f"[weights] {msg}")
        if source is None:
            sys.exit("[weights] every source failed:\n  " + "\n  ".join(errors) +
                     "\n  Attach the checkpoint as a Kaggle dataset "
                     "(Add Input) and re-run.")

    digest = sha256_of(a.out)
    size = os.path.getsize(a.out)
    match = digest == UPSTREAM_SHA256
    print(f"[weights] sha256 {digest}  ({size/1e6:.1f} MB)")
    if match:
        print(f"[weights] MATCHES the published upstream checkpoint "
              f"({UPSTREAM_NAME})")
    else:
        print(f"[weights] does NOT match the published upstream hash "
              f"{UPSTREAM_SHA256[:12]}... -- this is a DIFFERENT checkpoint. "
              f"Not an error (a private fine-tune is legitimate), but the "
              f"provenance record will say so and results carry it.")

    meta = {}
    if a.skip_load_gate:
        print("[weights] load gate SKIPPED (--skip-load-gate)")
    else:
        meta = inspect_and_gate(a.out)
        print(f"[weights] load gate PASSED; checkpoint metadata: {meta}")

    prov = {"path": a.out, "source": source, "sha256": digest, "bytes": size,
            "matches_upstream": match, "upstream_name": UPSTREAM_NAME,
            "upstream_sha256": UPSTREAM_SHA256, **meta}
    pp = os.path.join(os.path.dirname(a.out) or ".", "weights_provenance.json")
    with open(pp, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[weights] provenance -> {pp}")


if __name__ == "__main__":
    main()
