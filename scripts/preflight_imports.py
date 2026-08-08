#!/usr/bin/env python
"""Import every pipeline stage up front.

Hydra instantiates modules lazily, one stage at a time, so a stale transitive
dependency in stage 7 is only discovered after stages 1-6 have already run --
turning dependency debugging into one failure per (slow) evaluation run.

This walks the same `_target_` list the configs declare and imports each one,
so every breakage surfaces in a single pass that takes seconds and needs no GPU.

    python scripts/preflight_imports.py
    python scripts/preflight_imports.py -v     # full traceback for each failure

Exit code 0 means every stage is importable and `tracklab` will get as far as
actually running. Non-zero is the count of broken stages.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import traceback
from pathlib import Path

# APPLY the precondition, do not advise it. This script used to detect that every
# stage failed at `import matplotlib` and then print "Fix by: export
# MPLBACKEND=Agg" -- a check that could state the precondition but left the human
# to satisfy it. Setting it here is the same one line, and it must happen before
# any import pulls matplotlib in transitively.
# setdefault, not assignment: an explicit backend from the caller wins.
os.environ.setdefault("MPLBACKEND", "Agg")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "sn_gamestate" / "configs"
TARGET_RE = re.compile(r"^\s*_target_:\s*(\S+)\s*$", re.MULTILINE)

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[2m", "\033[0m"
)


def discover_targets() -> list[str]:
    """Collect every _target_ declared under configs/ (deduped, sorted)."""
    if not CONFIG_DIR.is_dir():
        sys.exit(f"config directory not found: {CONFIG_DIR}")
    targets: set[str] = set()
    for yaml_file in CONFIG_DIR.rglob("*.yaml"):
        targets.update(TARGET_RE.findall(yaml_file.read_text(encoding="utf-8")))
    return sorted(targets)


def is_matplotlib_backend_error(reason: str) -> bool:
    """True for the ValueError matplotlib raises over an unusable MPLBACKEND.

    matplotlib assigns rcParams['backend'] from MPLBACKEND at import time, so
    this fires during `import matplotlib` itself, before any pipeline code
    runs. Match on the invariant fragment only -- the tail of the message is a
    version-dependent list of supported backends.

    The same ValueError is reachable without MPLBACKEND at all (a bad backend
    in a matplotlibrc, or an explicit matplotlib.use() inside a dependency),
    and "export MPLBACKEND=Agg" would be useless advice there. So require the
    variable to be set AND to be the value the message is complaining about.
    """
    backend = os.environ.get("MPLBACKEND")
    return (bool(backend)
            and reason.startswith("ValueError:")
            and "is not a valid value for backend" in reason
            and "'%s'" % backend in reason)


def try_import(dotted: str) -> tuple[bool, str, str]:
    """Import 'pkg.mod.Class'. Returns (ok, short_reason, full_traceback)."""
    module_path, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - report anything, don't crash
        return False, f"{type(exc).__name__}: {exc}", traceback.format_exc()
    if not hasattr(module, attr):
        return False, f"module '{module_path}' has no attribute '{attr}'", ""
    return True, "", ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the full traceback for each failure")
    args = ap.parse_args(argv)

    targets = discover_targets()
    print(f"Importing {len(targets)} pipeline stages "
          f"(python {sys.version.split()[0]})\n")

    failures: list[tuple[str, str, str]] = []
    for dotted in targets:
        ok, reason, tb = try_import(dotted)
        if ok:
            print(f"  {GREEN}OK  {RESET} {dotted}")
        else:
            print(f"  {RED}FAIL{RESET} {dotted}")
            print(f"       {DIM}{reason}{RESET}")
            failures.append((dotted, reason, tb))

    print()
    if not failures:
        print(f"{GREEN}All {len(targets)} stages import cleanly.{RESET} "
              "tracklab should reach execution.")
        return 0

    print(f"{RED}{len(failures)} of {len(targets)} stages failed to import.{RESET}")

    # A uniform failure across EVERY stage is not dependency drift. Drift breaks
    # the stages that share the drifted package and leaves the rest importable;
    # losing all of them at once, with byte-identical errors, means the failure
    # is upstream of the pipeline entirely.
    reasons = {reason for _, reason, _ in failures}
    if (len(failures) == len(targets) and len(reasons) == 1
            and is_matplotlib_backend_error(next(iter(reasons)))):
        # The cause is a notebook kernel exporting its own inline backend into
        # a venv that does not have it. ipykernel does this unconditionally --
        # kernelapp.py defaults MPLBACKEND to
        # module://matplotlib_inline.backend_inline -- so it applies to any
        # Jupyter host, Kaggle included, and every subprocess inherits it.
        #
        # That name looks like it should be exempt: matplotlib accepts any
        # `module://` backend without importing it. But
        # BackendRegistry.is_valid_backend (matplotlib/backends/registry.py,
        # checked against 3.10.8) first rewrites the two known long names for
        # backward compatibility --
        #     module://matplotlib_inline.backend_inline -> inline
        #     module://ipympl.backend_nbagg             -> widget
        # -- and the rewritten name no longer starts with `module://`, so it
        # loses that exemption and has to resolve as an ordinary named backend
        # through entry points. matplotlib_inline supplies those, so the venv
        # not having it is exactly what makes the name unresolvable. Hence a
        # ValueError at `import matplotlib`, before any of our code runs.
        print(f"{YELLOW}Every stage failed identically, at `import matplotlib`."
              f"{RESET}")
        print("This is the ENVIRONMENT, not dependency drift -- pyproject.toml "
              "is not the problem.")
        print(f"    {DIM}{next(iter(reasons))}{RESET}")
        # This script already sets MPLBACKEND=Agg at import time, so reaching
        # here means the backend was NOT the cause, or something overrode it.
        print(f"MPLBACKEND is {os.environ.get('MPLBACKEND')!r} in this process, "
              f"so a missing backend is not the explanation -- read the reason "
              f"above.\n")
    else:
        print(f"{YELLOW}These are almost always unpinned transitive "
              f"dependencies that drifted forward.{RESET}")
        print("Fix by adding an upper bound to [project.dependencies] in "
              "pyproject.toml, then:")
        print("    rm -rf .venv && uv venv --clear --python 3.9 .venv "
              "&& uv pip install --python .venv -e .\n")

    if args.verbose:
        for dotted, _, tb in failures:
            if tb:
                print(f"{DIM}{'-' * 70}{RESET}\n{dotted}\n{tb}")

    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
