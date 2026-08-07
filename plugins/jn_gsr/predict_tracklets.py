#!/usr/bin/env python3
"""predict_tracklets.py -- GSR-pipeline worker: manifest of PREDICTED tracklets
-> jersey numbers, shardable across GPUs.

This is the bridge between sn-gamestate-lightning (python 3.9 / torch 1.13) and
this package's frozen recognizer (python 3.10 / torch 2.0.1): the host pipeline
never imports this code -- it writes a manifest, launches one copy of this
script per GPU inside .venv_jn, and merges the shard outputs.

Manifest (written by the host module):
    {"tracklets": {"<tid>": [["/abs/frame.jpg", [x, y, w, h]], ...], ...}}
frames in chronological order, xywh = top-left player box (the recognizer's
native input: it pads/crops internally; stride is applied by the recognizer).

Output (one file per shard):
    {"shard": i, "num_shards": n, "rule": ..., "stride": ...,
     "results": {"<tid>": {"number": "1".."99"|"-1",
                            "confidence": float, "n_used": int}}}

Sharding: sorted(tids)[shard::num_shards] -- deterministic, disjoint, complete.
Each worker is pinned to one GPU by the launcher via CUDA_VISIBLE_DEVICES (the
same isolation dual_gpu.py uses); inside the process that GPU is cuda:0.

Configuration is the package's FROZEN operating point (legibility > 0.9,
DBNet > 0.52, maxconf, stride 5); --rule/--stride exist for regression checks
only. --self-test exercises sharding, merge-completeness, the -1 path and the
number normalisation with a stubbed recognizer -- no torch, GPU or checkpoints.
"""
import argparse
import json
import os
import sys
import time

# Before ANY import that can reach matplotlib (see setup_env.py header).
os.environ["MPLBACKEND"] = "Agg"


def shard_tids(tids, shard, num_shards):
    """Deterministic disjoint complete partition of the tracklet ids."""
    return sorted(tids)[shard::num_shards]


def run(manifest_path, out_path, models_dir, shard, num_shards,
        stride, rule, fp16, recognizer=None):
    with open(manifest_path) as f:
        manifest = json.load(f)
    tracklets = manifest["tracklets"]
    mine = shard_tids(tracklets.keys(), shard, num_shards)
    print(f"[jn-worker {shard}/{num_shards}] {len(mine)} of "
          f"{len(tracklets)} tracklets", flush=True)

    if recognizer is None:
        from jn_recognizer import JerseyNumberRecognizer
        recognizer = JerseyNumberRecognizer(
            models_dir, stride=stride, fp16=fp16, rule=rule)

    results, t0 = {}, time.time()
    for k, tid in enumerate(mine):
        frames = [(fp, tuple(xywh)) for fp, xywh in tracklets[tid]]
        try:
            number, conf, n_used = recognizer.predict(frames)
        except Exception as e:  # one broken tracklet must not sink the shard
            print(f"[jn-worker {shard}] tracklet {tid} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            number, conf, n_used = "-1", 0.0, 0
        results[tid] = {"number": str(number), "confidence": float(conf),
                        "n_used": int(n_used)}
        if (k + 1) % 20 == 0 or k + 1 == len(mine):
            print(f"[jn-worker {shard}] {k + 1}/{len(mine)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"shard": shard, "num_shards": num_shards, "rule": rule,
                   "stride": stride, "results": results}, f)
    print(f"[jn-worker {shard}] wrote {out_path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--stride", type=int, default=5)     # frozen default
    ap.add_argument("--rule", default="maxconf",
                    choices=["maxconf", "wvote", "vote"])  # frozen default
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--stub", action="store_true",
                    help="TEST ONLY: deterministic stub recognizer (no torch); "
                         "number = (n_frames %% 99) + 1, or -1 for empty "
                         "tracklets. Used by the host module's unit tests.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.manifest or not a.out:
        ap.error("--manifest and --out are required (or use --self-test)")
    rec = _StubRecognizer() if a.stub else None
    run(a.manifest, a.out, a.models_dir, a.shard, a.num_shards,
        a.stride, a.rule, fp16=not a.no_fp16, recognizer=rec)
    return 0


class _StubRecognizer:
    """Deterministic, torch-free stand-in mirroring predict()'s contract."""

    def predict(self, frames):
        n = len(list(frames))
        if n == 0:
            return "-1", 0.0, 0
        return str((n % 99) + 1), 0.75, n


# --------------------------------- self-tests --------------------------------
def self_test():
    import tempfile

    # sharding: deterministic, disjoint, complete, order-independent
    tids = [f"SNGS-1_{i}" for i in range(11)]
    for n in (1, 2, 4):
        parts = [shard_tids(tids, i, n) for i in range(n)]
        assert sorted(sum(parts, [])) == sorted(tids)
        assert sum(len(p) for p in parts) == len(tids)
        for i in range(n):
            for j in range(i + 1, n):
                assert not set(parts[i]) & set(parts[j])
    assert shard_tids(reversed(tids), 0, 2) == shard_tids(tids, 0, 2)

    # end-to-end with the stub: 2 shards over 5 tracklets, then merged
    with tempfile.TemporaryDirectory() as d:
        man = {"tracklets": {
            "a": [["f1.jpg", [0, 0, 10, 20]]],
            "b": [["f1.jpg", [0, 0, 10, 20]], ["f2.jpg", [1, 1, 10, 20]]],
            "c": [["f1.jpg", [0, 0, 10, 20]]] * 3,
            "d": [],                                   # empty -> -1
            "e": [["f9.jpg", [5, 5, 8, 16]]] * 4,
        }}
        mp = os.path.join(d, "m.json")
        json.dump(man, open(mp, "w"))
        merged = {}
        for i in range(2):
            op = os.path.join(d, f"s{i}.json")
            run(mp, op, "models", i, 2, 5, "maxconf", True,
                recognizer=_StubRecognizer())
            out = json.load(open(op))
            assert out["shard"] == i and out["num_shards"] == 2
            merged.update(out["results"])
        assert set(merged) == set(man["tracklets"])
        assert merged["d"] == {"number": "-1", "confidence": 0.0, "n_used": 0}
        assert merged["b"]["number"] == "3" and merged["b"]["n_used"] == 2
        assert merged["e"]["number"] == "5" and merged["e"]["confidence"] == 0.75

    print("predict_tracklets: all self-tests passed (deterministic sharding, "
          "disjoint+complete partition, shard files, merge, -1 path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
