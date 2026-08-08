#!/usr/bin/env python
r"""Settle the static-audit findings that need the installed forks on disk.

The `track` / `gta_link` audit was performed against a checkout in which
`tracklab`, `bot_sort` and `strong_sort` were NOT present -- they ship inside the
installed tracklab package. Five findings were therefore classified UNVERIFIABLE
STATICALLY. The first moment those sources exist is immediately after the venv is
built, which is why this runs as Kaggle Stage 1.5, before anything expensive.

It reads source; it executes nothing from the pipeline and changes no state.

  F2  Does the engine force batch_size=1 for an ImageLevelModule?
      NotebookBotSORT.process takes batch["input"][0] and metadatas[...][0] --
      strictly one image -- while soccernet.yaml sets track: {batch_size: 64}.
      If the engine honours 64, 63 of every 64 frames are silently dropped.
      The reference notebook calls tracker.update(det, img) once per frame, so
      one-frame-per-call is the required semantics, not a preference.
  F4  Does GMC.apply log or print when motion estimation fails, and what does it
      return? An identity warp means camera-motion compensation is dead; the
      CMC ablation puts that at HOTA 0.6315 vs 0.6687.
  F7  Kalman state vector: BoT-SORT's (x, y, w, h) or ByteTrack's (x, y, a, h)?
      The classic transplant error.
  F8  Does STrack.update apply the EMA appearance update?
  F13 Does the SoccerNet encoder drop detections whose track_id is NaN?
      GTA-Link's collision guard produces those deliberately.

Usage
-----
    python scripts/probe_forks.py            # human-readable verdicts
    python scripts/probe_forks.py --json f   # also write a machine-readable summary

Exit code is 0 whether or not a defect is found -- this is an instrument, not a
gate. Read the verdicts.
"""
import argparse
import importlib
import inspect
import json
import re
import sys

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[2m", "\033[1m", "\033[0m"
)
RESULTS = {}


def hdr(t):
    print(f"\n{BOLD}{t}{RESET}")


def verdict(tag, answer, detail, evidence=(), concern=False):
    RESULTS[tag] = {"answer": answer, "detail": detail}
    colour = RED if concern else GREEN
    print(f"  {colour}{answer}{RESET}  {detail}")
    for ln in list(evidence)[:12]:
        print(f"       {DIM}| {ln.rstrip()[:110]}{RESET}")


def get_source(dotted):
    """'pkg.mod:Attr.method' -> source text, or None.

    DEFECT 1 THIS GUARDS AGAINST (UNRESOLVED, never SILENT). The first version
    of this probe reported F4 as "SILENT -- 0 print() / 0 log call(s)". That was
    not a finding about the fork; it was this function returning nothing and the
    caller reading absence-of-evidence as evidence-of-absence. A probe that
    cannot see its target must say UNRESOLVED. Reporting SILENT is worse than
    reporting nothing, because it looks like an answer.
    """
    mod_path, _, attr = dotted.partition(":")
    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    obj = mod
    for part in filter(None, attr.split(".")):
        obj = getattr(obj, part, None)
        if obj is None:
            return None, f"{mod_path} has no {attr}"
    try:
        return inspect.getsource(obj), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def get_class_source(dotted):
    """Whole-CLASS source, not one method.

    DEFECT 2a THIS GUARDS AGAINST. F4 read SILENT partly because the candidate
    list resolved `GMC.apply` first and stopped there -- but `apply` is only the
    DISPATCHER. The `print('Warning: not enough matching points')` lives in
    `applySparseOptFlow`, one level down. Grepping a single method answers a
    question nobody asked. Always grep the whole class.
    """
    return get_source(dotted)


def grep(src, pattern, flags=re.I):
    return [ln.strip() for ln in src.splitlines() if re.search(pattern, ln, flags)]


def call_sites(src, name):
    """Lines where `name` is CALLED, excluding its own definition.

    DEFECT 2b THIS GUARDS AGAINST (presence != call site). F7 read "MIXED --
    both conversions present" because it grepped for the SYMBOLS
    `tlwh_to_xyah` and `tlwh_to_xywh`. Both exist. But `tlwh_to_xyah` is defined
    at bot_sort.py:198 and **never called**: every state path -- activate:116,
    re_activate:127, update:155 -- uses `tlwh_to_xywh`. A symbol being defined
    says nothing about whether the code path is taken. This counts invocations
    (`name(`) and skips `def name`.
    """
    out = []
    for ln in src.splitlines():
        t = ln.strip()
        if re.match(rf"\s*def\s+{re.escape(name)}\b", ln):
            continue
        if re.search(rf"\b{re.escape(name)}\s*\(", t):
            out.append(t)
    return out


# --------------------------------------------------------------------------- F2
def probe_f2():
    hdr("F2 — does the engine batch images for an ImageLevelModule?")
    for dotted in ("tracklab.pipeline.imagelevel_module:ImageLevelModule",
                   "tracklab.pipeline.imagelevel_module:ImageLevelModule.dataloader"):
        src, err = get_source(dotted)
        if src:
            hits = grep(src, r"batch_size|DataLoader|collate|num_workers")
            if hits:
                verdict("F2", "READ", f"{dotted} references batching:", hits)
    eng, err = get_source("tracklab.engine.offline:OfflineTrackingEngine")
    if eng is None:
        eng, err = get_source("tracklab.engine:TrackingEngine")
    if eng is None:
        verdict("F2", "UNRESOLVED", f"engine source unavailable ({err})", concern=True)
        return
    hits = grep(eng, r"batch_size|DataLoader|image_filepath|for .*batch|model\.process")
    verdict("F2", "READ", "engine dispatch lines — confirm one image per process() call:",
            hits)
    print(f"       {YELLOW}Decide from the lines above: if batch_size flows from the "
          f"config into a DataLoader for image-level modules, track: "
          f"{{batch_size: 64}} in soccernet.yaml is a correctness bug.{RESET}")


# --------------------------------------------------------------------------- F4
def probe_f4():
    hdr("F4 — GMC failure path: does it log, and what does it return?")
    # WHOLE CLASS, not GMC.apply. apply is the dispatcher; the print lives in
    # applySparseOptFlow. The fork is installed as `bot_sort` (not `boxmot`),
    # so that name is tried first and boxmot is only a fallback.
    src = None
    for dotted in ("bot_sort.gmc:GMC", "boxmot.motion.cmc.sof:SOF"):
        s, err = get_class_source(dotted)
        if s:
            src = s
            print(f"       {DIM}source: {dotted} (whole class, "
                  f"{len(s.splitlines())} lines){RESET}")
            break
    if src is None:
        # UNRESOLVED, never SILENT -- absence of evidence is not evidence.
        verdict("F4", "UNRESOLVED",
                "GMC class source not importable; this probe cannot see the "
                "failure path and refuses to call it silent", concern=True)
        return
    if "applySparseOptFlow" not in src:
        verdict("F4", "UNRESOLVED",
                "applySparseOptFlow not in the class source -- probing the wrong "
                "object", concern=True)
        return
    prints = grep(src, r"\bprint\s*\(")
    logs = grep(src, r"\b(log|logger|LOGGER|warnings)\.")
    identity = grep(src, r"np\.eye|return\s+H\b|H\s*=\s*np\.eye")
    if not prints and not logs:
        verdict("F4", "UNRESOLVED",
                "no print() and no log call found anywhere in the class -- "
                "implausible for this fork; suspect the probe, not the code",
                concern=True)
        return
    concern = bool(prints) and not logs
    verdict("F4",
            "PRINTS, DOES NOT LOG" if concern else "LOGS",
            f"{len(prints)} print() / {len(logs)} log call(s) in the class",
            prints + logs, concern=concern)
    if identity:
        print(f"       {DIM}identity-return sites:{RESET}")
        for ln in identity[:6]:
            print(f"       {DIM}| {ln[:110]}{RESET}")
    print(f"       {YELLOW}If it prints rather than logs, the evidence will be in "
          f"captured STDOUT, not the log file — Stage 3 tees stdout for this "
          f"reason.{RESET}")


# --------------------------------------------------------------------------- F7
def probe_f7():
    """CALL SITES, not symbol presence. See call_sites() for why."""
    hdr("F7 — Kalman state vector: (x, y, w, h) or (x, y, a, h)?")
    src, err = get_class_source("bot_sort.bot_sort:STrack")
    if src is None:
        verdict("F7", "UNRESOLVED", f"STrack source not importable ({err})",
                concern=True)
        return
    xyah = call_sites(src, "tlwh_to_xyah")
    xywh = call_sites(src, "tlwh_to_xywh")
    defined = grep(src, r"def\s+tlwh_to_(xyah|xywh)")
    print(f"       {DIM}defined: {len(defined)} conversion(s); "
          f"CALLED: xyah {len(xyah)}, xywh {len(xywh)}{RESET}")
    for ln in defined:
        print(f"       {DIM}| def  {ln[:100]}{RESET}")
    if not xyah and not xywh:
        verdict("F7", "UNRESOLVED",
                "conversions may be defined but NO call site found -- the state "
                "path is not visible to this probe", concern=True)
        return
    if xywh and not xyah:
        verdict("F7", "(w, h) — CORRECT",
                f"{len(xywh)} call site(s), all tlwh_to_xywh; tlwh_to_xyah is "
                f"{'defined but never called' if defined else 'absent'}", xywh)
    elif xyah and not xywh:
        verdict("F7", "(a, h) — MISMATCH",
                "ByteTrack state vector in a BoT-SORT tracker: the classic "
                "transplant error", xyah, concern=True)
    else:
        verdict("F7", "MIXED — GENUINELY",
                f"both are CALLED ({len(xywh)} xywh / {len(xyah)} xyah), which is "
                f"a real inconsistency rather than a symbol-presence artefact",
                xywh + xyah, concern=True)


# --------------------------------------------------------------------------- F8
def probe_f8():
    hdr("F8 — is the EMA appearance update applied?")
    src, err = get_source("bot_sort.bot_sort:STrack.update_features")
    if src is None:
        s2, _ = get_source("bot_sort.bot_sort:STrack")
        src = s2 or ""
    ema = grep(src, r"smooth_feat\s*=|alpha\s*\*|self\.alpha")
    guard = grep(src, r"if\s+feat\s+is\s+None|feat\s+is\s+not\s+None")
    if ema:
        verdict("F8", "PRESENT", "EMA update found", ema + guard)
        if not guard:
            print(f"       {YELLOW}No explicit None-guard seen. The second BYTE "
                  f"association builds STracks WITHOUT features "
                  f"(bot_sort_notebook.py:205), so a low-confidence match must not "
                  f"reach this unguarded.{RESET}")
    else:
        verdict("F8", "NOT FOUND", "no EMA update located — appearance may not "
                "accumulate", concern=True)


# -------------------------------------------------------------------------- F13
def probe_f13():
    hdr("F13 — does the SoccerNet encoder drop NaN track_id?")
    found = False
    for dotted in ("tracklab.datastruct.tracker_state:TrackerState",
                   "tracklab.wrappers.eval.soccernet_gs:SoccerNetGSEvaluator",
                   "tracklab.wrappers.eval.soccernet:SoccerNetEvaluator"):
        s, _ = get_source(dotted)
        if not s:
            continue
        hits = grep(s, r"track_id.*(dropna|notna|isna|fillna)|dropna\(.*track_id")
        if hits:
            verdict("F13", "DROPS", f"{dotted} filters NaN track_id", hits)
            found = True
            break
    if not found:
        verdict("F13", "UNRESOLVED",
                "no explicit NaN track_id filter found in the evaluator; a "
                "NaN-id detection may reach the encoder as a spurious unmatched "
                "detection and depress GS-HOTA precision", concern=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write a machine-readable summary here")
    args = ap.parse_args(argv)

    print(f"{BOLD}Fork probe — settling the UNVERIFIABLE-STATICALLY audit findings"
          f"{RESET}")
    for name in ("tracklab", "bot_sort", "strong_sort", "boxmot"):
        try:
            m = importlib.import_module(name)
            print(f"       {name:12s} {getattr(m, '__version__', '?'):>10}  "
                  f"{getattr(m, '__file__', '?')}")
        except Exception as exc:
            print(f"       {name:12s} {RED}not importable{RESET} ({type(exc).__name__})")

    for fn in (probe_f2, probe_f4, probe_f7, probe_f8, probe_f13):
        try:
            fn()
        except Exception as exc:
            print(f"  {RED}PROBE ERROR{RESET} {fn.__name__}: "
                  f"{type(exc).__name__}: {exc}")

    hdr("summary")
    for tag in ("F2", "F4", "F7", "F8", "F13"):
        r = RESULTS.get(tag)
        print(f"  {tag:>4}  {r['answer'] if r else 'UNRESOLVED'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(RESULTS, f, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
