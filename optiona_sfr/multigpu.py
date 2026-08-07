"""Multi-GPU sharding for the detection stage + CPU-parallel experiments.

Detection (Stage 2) is embarrassingly parallel at sequence granularity: each
worker owns one GPU (Kaggle: 2x T4), loads its own model replicas, and caches
its round-robin shard of sequences. Falls back transparently to a single GPU
or CPU. Workers are spawned processes (CUDA-safe) and the cache is
resume-safe, so a crashed worker loses nothing already written.

The experiment stage (Stage 4) is CPU-only (runs from the detection cache);
`parallel_run_experiments` fans sequences out over CPU processes.
"""
from __future__ import annotations
import os
import traceback
import multiprocessing as mp
from pathlib import Path
from dataclasses import asdict


def gpu_inventory():
    """-> (n_gpus, [names]). Never raises."""
    try:
        import torch
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return n, names
    except Exception:
        return 0, []


def _shard(items, n):
    return [items[i::n] for i in range(n)]


def _cache_worker(gpu_id, seq_dirs, det_cfg_dict, repo_dir, cache_dir, q):
    """One process per GPU. Rebuilds config, pins its device, caches its shard."""
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        from optiona_sfr.config import DetectorCfg
        from optiona_sfr.pnl_adapter import load_models
        from optiona_sfr.detection import cache_sequence
        from optiona_sfr.experiments import iter_frames

        cfg = DetectorCfg(**{k: v for k, v in det_cfg_dict.items()
                             if k in DetectorCfg.__dataclass_fields__})
        cfg.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        cfg.weights_kp_path = det_cfg_dict["weights_kp_path"]
        cfg.weights_line_path = det_cfg_dict["weights_line_path"]
        models = load_models(cfg, repo_dir)
        for sd in seq_dirs:
            sd = Path(sd)
            n = cache_sequence(iter_frames(sd), sd.name, models, cfg,
                               repo_dir, cache_dir)
            q.put(("done", gpu_id, sd.name, n))
        q.put(("exit", gpu_id, None, None))
    except Exception:
        q.put(("error", gpu_id, traceback.format_exc(), None))


def shard_cache_all(seq_dirs, detector_cfg, repo_dir, cache_dir,
                    max_gpus=None, verbose=True):
    """Cache detections for all sequences using every available GPU.

    - >=2 GPUs: one spawn-worker per GPU, sequences sharded round-robin.
    - exactly 1 GPU: sequential on that GPU (no process overhead).
    - 0 GPUs: sequential on CPU (slow — warn).
    Returns {seq_id: n_new_frames}.
    """
    n_gpus, names = gpu_inventory()
    if max_gpus is not None:
        n_gpus = min(n_gpus, max_gpus)
    if verbose:
        print(f"[multigpu] {n_gpus} GPU(s) available: {names or 'none'}")

    det_dict = asdict(detector_cfg)
    for attr in ("weights_kp_path", "weights_line_path"):
        p = getattr(detector_cfg, attr, None)
        if p is None:
            raise RuntimeError(
                f"detector_cfg.{attr} is not set — run "
                "kaggle_setup.setup_pnlcalib(paths, detector_cfg) first "
                "(Stage 0) so weight files are downloaded and resolved.")
        det_dict[attr] = str(p)
    seq_dirs = [str(s) for s in seq_dirs]
    results = {}

    if n_gpus <= 1:
        gpu_id = 0 if n_gpus == 1 else -1
        if gpu_id < 0 and verbose:
            print("[multigpu] no GPU found — running detection on CPU "
                  "(consider enabling a GPU accelerator).")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        # Run in-process for the single-device case (simpler tracebacks).
        _cache_worker(gpu_id, seq_dirs, det_dict, str(repo_dir),
                      str(cache_dir), q)
        while not q.empty():
            kind, g, a, b = q.get()
            if kind == "done":
                results[a] = b
                if verbose:
                    print(f"[gpu{g}] {a}: +{b} frames")
            elif kind == "error":
                raise RuntimeError(a)
        return results

    shards = _shard(seq_dirs, n_gpus)
    ctx = mp.get_context("spawn")           # required for CUDA in children
    q = ctx.Queue()
    procs = []
    for g in range(n_gpus):
        p = ctx.Process(target=_cache_worker,
                        args=(g, shards[g], det_dict, str(repo_dir),
                              str(cache_dir), q))
        p.start()
        procs.append(p)
        if verbose:
            print(f"[multigpu] worker on cuda:{g} -> {len(shards[g])} sequences")
    alive = n_gpus
    errors = []
    while alive > 0:
        kind, g, a, b = q.get()
        if kind == "done":
            results[a] = b
            if verbose:
                print(f"[gpu{g}] {a}: +{b} frames")
        elif kind == "exit":
            alive -= 1
        elif kind == "error":
            errors.append(f"[gpu{g}]\n{a}")
            alive -= 1
    for p in procs:
        p.join()
    if errors:
        # Cache is resume-safe: report and let the caller rerun the stage.
        raise RuntimeError("Worker error(s) — cached frames are kept, rerun "
                           "the stage to resume:\n" + "\n".join(errors))
    return results


# ----------------------------------------------------------------------------- CPU-parallel experiments
def _exp_worker(args):
    exp_cfg_bytes, seq_dir, kp_world, membership, convention = args
    import pickle
    from optiona_sfr.experiments import run_sequence, set_pitch_convention
    # Spawned workers start with a FRESH module state, so the pitch convention
    # installed in the parent (Stage 3c) does not carry over. Re-install it
    # here or every worker silently uses the default guess.
    if convention is not None:
        set_pitch_convention(*convention)
    exp_cfg = pickle.loads(exp_cfg_bytes)
    df = run_sequence(Path(seq_dir), exp_cfg, kp_world, membership)
    return (exp_cfg.name, Path(seq_dir).name, len(df))


def parallel_run_experiments(grid, seq_dirs, kp_world, membership,
                             max_workers=None, verbose=True):
    """Fan (experiment, sequence) jobs over CPU processes. Result CSVs are the
    checkpoint, so this is fully resume-safe as well."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed
    max_workers = max_workers or max(2, (os.cpu_count() or 4) - 1)
    from optiona_sfr.experiments import _CONVENTION
    if verbose:
        print(f"[experiments] pitch convention propagated to workers: "
              f"right_x={'+' if _CONVENTION[0] else '-'}, "
              f"top_y={'+' if _CONVENTION[1] else '-'}")
    jobs = [(pickle.dumps(g), str(s), kp_world, membership, tuple(_CONVENTION))
            for g in grid for s in seq_dirs]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_exp_worker, j) for j in jobs]
        for f in as_completed(futs):
            name, seq, n = f.result()
            if verbose:
                print(f"[{name}] {seq}: {n} frames")
