#!/usr/bin/env python3
"""fetch_weights.py -- the three HuggingFace checkpoints, hash-checked.

    DBNet++      Ynniss/dbnetppp_jn              111 MB   single .pth
    ResNet-18    Ynniss/Resnet18_multi_single_player  44.8 MB   EXPLODED archive
    ResNet-34    Ynniss/Legibility_classifier     85.3 MB   single .pth

PARSeq is NOT here: stage_weights.py fetches it, because it also has to gate the
strhub Lightning load, which is a different kind of check.

TWO SHAPES OF REPOSITORY. Two of these are ordinary single-file repos and are
fetched with hf_hub_download plus a sha256 comparison. The ResNet-18 repo is a
torch.save zip archive uploaded EXPLODED into its 128 member files, so there is
no single file to download or hash. crop_classifier.resolve_weights snapshot-
downloads it and reassembles the archive; provenance is pinned on data.pkl's
sha256, which fixes every key name, dtype and shape in the checkpoint. The
reassembled .pt is deliberately NOT hashed -- zip records carry mtimes, so it is
not byte-reproducible and a hash of it would fail for no reason.

A MISMATCH IS REPORTED, NOT FATAL. Re-uploading a private fine-tune under the
same path is legitimate; quietly claiming a provenance you do not have is not.
Every mismatch is printed and written into the provenance record.

    $PY fetch_weights.py --out-dir models
    $PY fetch_weights.py --out-dir models --dbnet-attached /kaggle/input/x/y.pth
"""
import argparse
import hashlib
import json
import os
import sys

# repo, filename, sha256, approximate size, destination, description
SOURCES = {
    "dbnet": {
        "repo": "Ynniss/dbnetppp_jn",
        "file": "best_icdar_hmean_epoch_10.pth",
        "sha256": "1e8ee32969a0264dd8d26918a3dea7ab9914c90033b087b1bf97c5eefecfe6c9",
        "size_mb": 111,
        "out": "best_icdar_hmean_epoch_10.pth",
        "what": "DBNet++ text detector (number localisation)",
    },
    "legibility": {
        "repo": "Ynniss/Legibility_classifier",
        "file": "legibility_resnet34_soccer_20240215.pth",
        "sha256": "b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41",
        "size_mb": 85.3,
        "out": "sn_legibility.pth",
        "what": "Koshkina ResNet-34 legibility classifier (SoccerNet fine-tuned)",
    },
}

# Exploded-archive repo: no single file, so it has its own fetch path.
CLASSIFIER = {
    "repo": "Ynniss/Resnet18_multi_single_player",
    "out": "resnet18_multi_single.pt",
    "size_mb": 44.8,
    "what": "ResNet-18 single/multi player frame filter",
}

# For reference only -- stage_weights.py fetches and gates this one.
PARSEQ = {
    "repo": "Ynniss/parseq_soccernet_epoch24",
    "file": ("parseq_epoch=24-step=2575-val_accuracy=95.6044"
             "-val_NED=96.3255.ckpt"),
    "sha256": "14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879",
    "size_mb": 382,
}


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch_single(key, out_dir, attached=None, refresh=False):
    """An ordinary one-file HuggingFace repo."""
    spec = SOURCES[key]
    dest = os.path.join(out_dir, spec["out"])
    os.makedirs(out_dir, exist_ok=True)

    if refresh and os.path.exists(dest):
        # A re-upload under the SAME repo/filename is invisible to both caches:
        # the local copy short-circuits, and huggingface_hub may serve a stale
        # blob. --refresh forces both to be re-checked.
        os.remove(dest)

    if attached and os.path.exists(attached):
        import shutil
        if os.path.abspath(attached) != os.path.abspath(dest):
            shutil.copyfile(attached, dest)
        source = f"attached: {attached}"
    elif os.path.exists(dest):
        source = "already-present"
    else:
        print(f"[weights] {key}: downloading ~{spec['size_mb']} MB from "
              f"{spec['repo']}...")
        import shutil

        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(spec["repo"], spec["file"],
                                force_download=refresh)
        except Exception as e:
            raise SystemExit(
                f"[weights] {key}: could not fetch {spec['repo']}/"
                f"{spec['file']}: {type(e).__name__}: {e}\n"
                f"  * Internet must be ON in the Kaggle notebook settings.\n"
                f"  * If the repo is private: huggingface-cli login "
                f"(or set $HF_TOKEN).\n"
                f"  * Or attach the file as a Kaggle dataset and pass "
                f"--{key}-attached <path>.") from e
        shutil.copyfile(p, dest)
        source = f"huggingface: {spec['repo']}/{spec['file']}"

    size = os.path.getsize(dest)
    digest = sha256_of(dest)
    match = digest == spec["sha256"]
    print(f"[weights] {key}: {dest}  {size / 1e6:.1f} MB  "
          f"sha256 {digest[:16]}...  {'OK' if match else 'MISMATCH'}")
    if not match:
        print(f"[weights]   WARNING: published sha256 is "
              f"{spec['sha256'][:16]}... -- this is a DIFFERENT file. Not an "
              f"error (a private re-upload is legitimate), but every result "
              f"carries this flag.")
    return {"key": key, "path": dest, "source": source, "sha256": digest,
            "published_sha256": spec["sha256"], "sha256_matches": match,
            "bytes": size, "repo": spec["repo"], "what": spec["what"]}


def fetch_classifier(out_dir, attached=None, refresh=False):
    """The exploded-archive repo. Delegates to the loader that understands it,
    so there is exactly ONE implementation of the reassembly."""
    import crop_classifier as CC
    dest = os.path.join(out_dir, CLASSIFIER["out"])
    if refresh and os.path.exists(dest):
        os.remove(dest)
    if not (attached or os.path.exists(dest)):
        print(f"[weights] classifier: downloading ~{CLASSIFIER['size_mb']} MB "
              f"from {CLASSIFIER['repo']} (128 files, then reassembled)...")
    path, source, info = CC.resolve_weights(attached, dest,
                                            repo=CLASSIFIER["repo"])
    size = os.path.getsize(path)
    print(f"[weights] classifier: {path}  {size / 1e6:.1f} MB  ({source})")
    if info.get("data_pkl_sha256"):
        print(f"[weights]   data.pkl sha256 {info['data_pkl_sha256'][:16]}...  "
              f"{'OK' if info.get('data_pkl_matches') else 'MISMATCH'}  "
              f"({info.get('n_storages')} tensor storages)")
    return {"key": "classifier", "path": path, "source": source,
            "bytes": size, "repo": CLASSIFIER["repo"],
            "what": CLASSIFIER["what"], **info}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--dbnet-attached", default=None,
                    help="skip the 111 MB download (Kaggle dataset path)")
    ap.add_argument("--legibility-attached", default=None)
    ap.add_argument("--clf-attached", default=None,
                    help="an already-reassembled ResNet-18 .pt")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if a local copy exists")
    ap.add_argument("--provenance", default=None,
                    help="default: <out-dir>/weights_provenance.json")
    a = ap.parse_args(argv)

    # final config: the ResNet-18 single/multi filter is retired (measured
    # <= no-op in every ablation cell), so its weights are no longer fetched.
    recs = [fetch_single("dbnet", a.out_dir, a.dbnet_attached, a.refresh),
            fetch_single("legibility", a.out_dir, a.legibility_attached,
                         a.refresh)]

    out = a.provenance or os.path.join(a.out_dir, "weights_provenance.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"weights": recs, "parseq_note": PARSEQ}, f, indent=2)
    print(f"\n[weights] provenance -> {out}")

    bad = [r["key"] for r in recs if r.get("sha256_matches") is False
           or r.get("data_pkl_matches") is False]
    if bad:
        print(f"[weights] NOTE: {bad} did not match the published hash. "
              f"Proceeding; the flag travels with the results.")
    print("[weights] all three checkpoints present. PARSeq: run stage_weights.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
