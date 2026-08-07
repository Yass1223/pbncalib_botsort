"""One-shot environment setup for Kaggle sessions.

* Clones the GPL-2.0 PnLCalib repo at runtime (never vendored — license
  hygiene) and downloads the verified v1.0.0 release weights.
* Downloads SoccerNet data via the official pip package. SoccerNet-GSR
  (task="gamestate-2024") needs NO password; a password is optional and only
  used for password-gated tasks. Never hardcode it (use the Kaggle secret or
  the SOCCERNET_PASSWORD env var), and never re-upload SoccerNet data
  publicly.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

PNLCALIB_GIT = "https://github.com/mguti97/PnLCalib.git"
WEIGHTS_URLS = {  # verified release asset URLs (repo releases/tag/v1.0.0)
    "SV_kp": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
    "SV_lines": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
}


def sh(cmd, check=True):
    """Run a shell command, translating the exit codes that actually bite us
    on Kaggle into actionable messages instead of a bare CalledProcessError."""
    print("+", cmd)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        hint = {
            50: ("DISK FULL during extraction. /kaggle/working is capped at "
                 "20 GiB; use scratch space (/kaggle/tmp, ~60 GiB) and stage "
                 "one split at a time — see ensure_gsr_data(root=...)."),
            9: "Archive is corrupt or truncated — delete it and re-download.",
            2: "Archive not found or unreadable.",
        }.get(r.returncode, "")
        raise RuntimeError(
            f"Command failed (exit {r.returncode}): {cmd}\n{hint}".rstrip())
    return r.returncode


def disk_free_gb(path):
    """Free space in GiB on the filesystem holding `path` (nearest existing
    parent, so it works before the directory is created)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    st = os.statvfs(str(p))
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def pick_scratch_root(name="SoccerNetGS", min_free_gb=25.0):
    """Choose where to put the dataset.

    /kaggle/working is persisted but capped at 20 GiB — too small for a full
    GSR split plus its archive. Prefer scratch (/kaggle/tmp) when it exists
    and has room; fall back to working with a loud warning.
    """
    cands = [Path("/kaggle/tmp"), Path("/kaggle/temp"), Path("/tmp"),
             Path("/kaggle/working")]
    for base in cands:
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        free = disk_free_gb(base)
        if free >= min_free_gb:
            print(f"[disk] using scratch {base} ({free:.1f} GiB free)")
            return base / name
    base = Path("/kaggle/working")
    print(f"[disk] WARNING: no location with >={min_free_gb} GiB free; "
          f"falling back to {base} ({disk_free_gb(base):.1f} GiB). "
          "Stage a single split and use chunked processing.")
    return base / name


def _zip_size_gb(z):
    return Path(z).stat().st_size / (1024 ** 3)


def verify_checkpoint(path, min_mb=5.0):
    """Is this a usable torch checkpoint? Existence is NOT validity: a failed
    or interrupted download leaves a truncated file (or an HTML error page)
    with the right name, and every later run then reuses it. Symptom seen on
    Kaggle: 'PytorchStreamReader failed reading zip archive: failed finding
    central directory'."""
    import zipfile
    path = Path(path)
    if not path.exists():
        return False, "missing"
    mb = path.stat().st_size / 1e6
    if mb < min_mb:
        return False, f"too small ({mb:.2f} MB)"
    head = path.read_bytes()[:512].lstrip()[:64].lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        return False, "HTML error page, not a checkpoint"
    # torch >=1.6 checkpoints are zip archives; legacy ones start with a pickle
    if not zipfile.is_zipfile(path):
        if not path.read_bytes()[:2] == b"\x80\x02":
            return False, f"not a torch archive ({mb:.1f} MB)"
    return True, f"ok ({mb:.1f} MB)"


def download_checkpoint(url, dst, retries=3, verbose=True):
    """Atomic, verified download: fetch to <dst>.part, validate, then rename.
    A partially written file can therefore never masquerade as a good one."""
    dst = Path(dst)
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        if part.exists():
            part.unlink()
        if verbose:
            print(f"[weights] downloading {dst.name} (attempt {attempt}/{retries})")
        sh(f"wget -q --tries=3 --timeout=60 -O {part} {url}", check=False)
        ok, why = verify_checkpoint(part)
        if ok:
            part.replace(dst)
            if verbose:
                print(f"[weights] {dst.name}: {why}")
            return True
        print(f"[weights] attempt {attempt} failed: {why}")
    if part.exists():
        part.unlink()
    return False


def setup_pnlcalib(paths, detector_cfg, force_weights=False):
    """Clone the PnLCalib repo and ensure VALID weights are present.

    Each checkpoint is verified (not merely present); anything corrupt is
    deleted and re-downloaded, and the download is atomic so an interrupted
    fetch cannot leave a plausible-looking file behind.
    """
    repo = Path(paths.pnlcalib_repo)
    if not (repo / "inference.py").exists():
        sh(f"git clone --depth 1 {PNLCALIB_GIT} {repo}")
    wdir = Path(paths.weights_dir); wdir.mkdir(parents=True, exist_ok=True)

    for name, attr in [(detector_cfg.weights_kp, "weights_kp_path"),
                       (detector_cfg.weights_line, "weights_line_path")]:
        dst = wdir / name
        url = WEIGHTS_URLS.get(name)
        if force_weights and dst.exists():
            dst.unlink()
        ok, why = verify_checkpoint(dst)
        if not ok:
            if dst.exists():
                print(f"[weights] {name} is unusable ({why}) - re-downloading")
                dst.unlink()
            if url is None:
                raise RuntimeError(f"No download URL known for weight '{name}'.")
            if not download_checkpoint(url, dst):
                raise RuntimeError(
                    f"Could not obtain a valid checkpoint for '{name}' from "
                    f"{url}. Check Internet is enabled in this session, then "
                    "retry; or download it manually and place it at "
                    f"{dst}.")
        else:
            print(f"[weights] {name}: {why}")
        setattr(detector_cfg, attr, dst)

    sh(f"{sys.executable} -m pip -q install timm einops 2>/dev/null || true",
       check=False)
    return repo


def split_is_ready(root, split, min_seqs=1):
    """A split counts as staged only if it holds real sequence dirs (each with
    an img1/ folder). Checking that the parent directory merely *exists* is
    not enough — a failed run leaves an empty dir behind and would silently
    skip the download."""
    d = Path(root) / split
    if not d.is_dir():
        return False
    seqs = [p for p in d.iterdir() if p.is_dir() and (p / "img1").is_dir()]
    return len(seqs) >= min_seqs


def _flatten_if_nested(root, split):
    """Some archives contain a top-level folder, giving <root>/<split>/<split>/
    SNGS-xxx. Move sequence dirs up one level so the layout is uniform."""
    d = Path(root) / split
    if split_is_ready(root, split):
        return
    for sub in [p for p in d.iterdir() if p.is_dir()] if d.is_dir() else []:
        inner = [p for p in sub.iterdir() if p.is_dir() and (p / "img1").is_dir()]
        if inner:
            for p in inner:
                target = d / p.name
                if not target.exists():
                    p.rename(target)
            print(f"[soccernet] flattened nested layout under {sub}")
            return


def ensure_gsr_data(root=None, splits=("test",), task="gamestate-2024",
                    password=None, verbose=True, delete_zip_after=True,
                    max_sequences=None):
    """Idempotent, resume-capable staging of SoccerNet-GSR.

    Per split: (1) already staged -> skip; (2) zip present -> unzip;
    (3) otherwise download that split, then unzip. Disk is checked BEFORE
    extraction; archives can be kept or deleted; max_sequences enables pilot
    runs via selective extraction. Safe to re-run after any interruption.
    Returns the root that contains <split>/ directories.
    """
    root = Path(root) if root is not None else pick_scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[soccernet] root={root} ({disk_free_gb(root):.1f} GiB free)")
    todo = []
    for s in splits:
        if split_is_ready(root, s):
            if verbose:
                n = len([p for p in (root / s).iterdir() if p.is_dir()])
                print(f"[soccernet] {s}: already staged ({n} sequences)")
            continue
        todo.append(s)

    if todo:
        if password is None:
            password = os.environ.get("SOCCERNET_PASSWORD")
            if password is None:
                try:
                    from kaggle_secrets import UserSecretsClient
                    password = UserSecretsClient().get_secret("SOCCERNET_PASSWORD")
                except Exception:
                    pass            # GSR needs no password
        need_download = [s for s in todo
                         if not (root / task / f"{s}.zip").exists()]
        if need_download:
            # Only touch pip when a real download is required, so a pure
            # unzip/resume works in any environment.
            sh(f"{sys.executable} -m pip -q install SoccerNet "
               "--break-system-packages 2>/dev/null || "
               f"{sys.executable} -m pip -q install SoccerNet", check=False)
            from SoccerNet.Downloader import SoccerNetDownloader
            dl = SoccerNetDownloader(LocalDirectory=str(root))
            if password:
                dl.password = password
            if verbose:
                print(f"[soccernet] downloading task={task} splits={need_download}")
            dl.downloadDataTask(task=task, split=list(need_download))
        for s in todo:
            z = root / task / f"{s}.zip"
            if not z.exists():                       # tolerate other layouts
                cands = list(root.rglob(f"{s}.zip"))
                z = cands[0] if cands else None
            if z is None:
                print(f"[error] no archive found for split '{s}'. Contents:")
                sh(f"find {root} -maxdepth 3 | head -40", check=False)
                continue
            dest = root / s
            dest.mkdir(exist_ok=True)
            need = _zip_size_gb(z) * 1.05      # JPEGs barely compress
            free = disk_free_gb(dest)
            if verbose:
                print(f"[disk] {s}: archive {need/1.05:.1f} GiB, "
                      f"free {free:.1f} GiB")
            if free < need:
                raise RuntimeError(
                    f"Not enough space to extract {z.name}: need ~{need:.1f} "
                    f"GiB, have {free:.1f} GiB on {dest}.\n"
                    "Options: (a) put the dataset on scratch — "
                    "kaggle_setup.pick_scratch_root() picks /kaggle/tmp; "
                    "(b) stage ONE split (splits=('test',)); (c) use "
                    "max_sequences=N for a pilot; (d) extract_sequences + "
                    "prune_images per chunk.")
            if max_sequences:
                extract_sequences(z, dest, limit=max_sequences, verbose=verbose)
            else:
                sh(f"unzip -qn {z} -d {dest}")
            _flatten_if_nested(root, s)
            if delete_zip_after and split_is_ready(root, s):
                if verbose:
                    print(f"[disk] removing {z.name} "
                          f"(frees {_zip_size_gb(z):.1f} GiB)")
                z.unlink()

    ok = [s for s in splits if split_is_ready(root, s)]
    bad = [s for s in splits if s not in ok]
    if bad:
        print(f"[error] splits still not staged: {bad}. Tree under {root}:")
        sh(f"find {root} -maxdepth 3 -type d | head -40", check=False)
        raise RuntimeError(f"Staging failed for {bad}; see the tree above.")
    check_gsr_version(root, ok)
    return root


def download_soccernet(task="gamestate-2024", local_dir="data/SoccerNetGS",
                       splits=("valid", "test"), password=None):
    """Thin wrapper kept for backwards compatibility — prefer ensure_gsr_data,
    which is resume-capable.

    IMPORTANT (verified against the sn-gamestate README, 2026-08-02): the
    official snippet for task="gamestate-2024" uses NO password — SoccerNet-GSR
    downloads without the NDA. A password is optional and only used if passed,
    set in SOCCERNET_PASSWORD, or present as a Kaggle secret.
    """
    return ensure_gsr_data(root=local_dir, splits=splits, task=task,
                           password=password)


def check_gsr_version(root, splits, minimum=1.3):
    """The README requires dataset version >= 1.3 (bbox-pitch temporal
    consistency fix). Verify rather than assume."""
    import json
    root = Path(root)
    for split in splits:
        labels = sorted((root / split).rglob("Labels-GameState.json"))
        if not labels:
            print(f"[warn] no Labels-GameState.json found under {root/split}")
            continue
        try:
            v = json.loads(labels[0].read_text()).get("info", {}).get("version")
            print(f"[soccernet] {split}: {len(labels)} sequences, version {v}")
            if v is not None and float(v) < minimum:
                print(f"[WARN] dataset version {v} < {minimum} — re-download; "
                      "older versions have inconsistent bbox-pitch labels.")
        except Exception as e:
            print(f"[warn] could not read version from {labels[0]}: {e}")


# ----------------------------------------------------------------------------- selective extraction / pruning
def zip_sequence_names(zip_path):
    """Sequence directory names inside a split archive, without extracting."""
    import zipfile
    names = set()
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            parts = Path(n).parts
            for i, p in enumerate(parts):
                if p.startswith("SNGS-"):
                    names.add(p)
                    break
            else:
                if len(parts) > 1:
                    names.add(parts[0])
    return sorted(names)


def extract_sequences(zip_path, dest, names=None, limit=None, verbose=True):
    """Extract only selected sequences from a split archive — the basis of
    chunked staging, so peak disk usage is one chunk instead of a whole split."""
    import zipfile
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        all_names = zip_sequence_names(zip_path)
        picked = names if names is not None else all_names[:limit]
        picked = set(picked)
        members = [n for n in z.namelist()
                   if any(f"{p}/" in n or n.startswith(f"{p}/") for p in picked)]
        if verbose:
            print(f"[soccernet] extracting {len(picked)} sequences "
                  f"({len(members)} members) from {Path(zip_path).name}")
        z.extractall(dest, members=members)
    return sorted(picked)


def prune_images(seq_dir, verbose=False):
    """Delete img1/ for a sequence but KEEP Labels-GameState.json (needed for
    metrics). Safe once detections and flow are cached — the optimization
    stages never touch pixels again."""
    import shutil
    img = Path(seq_dir) / "img1"
    if img.is_dir():
        n = len(list(img.glob("*")))
        shutil.rmtree(img)
        if verbose:
            print(f"[disk] pruned {n} images from {Path(seq_dir).name}")
        return n
    return 0


# ----------------------------------------------------------------------------- split-at-a-time workflow
def harvest_labels(root, split, dest_dir, verbose=True):
    """Copy every Labels-GameState.json into persisted storage BEFORE the
    split is released.

    Critical on Kaggle: the dataset lives on scratch (/kaggle/tmp), which is
    wiped when the session ends, while the detection cache lives in
    /kaggle/working and survives. Annotations are what the reprojection metric
    scores against, so they must travel with the cache — they are tiny
    (a few MB per sequence) compared with the images.
    """
    import shutil
    root, dest_dir = Path(root), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for seq in sorted((root / split).iterdir()):
        src = seq / "Labels-GameState.json"
        if src.exists():
            d = dest_dir / seq.name
            d.mkdir(exist_ok=True)
            shutil.copy2(src, d / "Labels-GameState.json")
            n += 1
    if verbose:
        mb = sum(f.stat().st_size for f in dest_dir.rglob("*.json")) / 1e6
        print(f"[labels] harvested {n} annotation files -> {dest_dir} "
              f"({mb:.1f} MB)")
    return n


def release_split(root, split, keep_labels=True, verbose=True):
    """Free the disk used by a split once its detections are cached.

    keep_labels=True removes only the images (default, safest). Pass False to
    delete the split directory entirely — only do that AFTER harvest_labels().
    """
    import shutil
    root = Path(root)
    d = root / split
    if not d.is_dir():
        return 0.0
    before = disk_free_gb(d)
    if keep_labels:
        for seq in sorted(p for p in d.iterdir() if p.is_dir()):
            prune_images(seq)
    else:
        shutil.rmtree(d)
    for z in (root / "gamestate-2024").glob(f"{split}.zip"):
        z.unlink()
    freed = disk_free_gb(root) - before
    if verbose:
        print(f"[disk] released '{split}' (keep_labels={keep_labels}); "
              f"freed ~{freed:.1f} GiB, now {disk_free_gb(root):.1f} GiB free")
    return freed
