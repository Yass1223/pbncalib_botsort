"""Path bootstrap — paste as the first Kaggle cell (or import if already on path).

Finds the directory that CONTAINS the `optiona_sfr` package (the one holding
`optiona_sfr/__init__.py`) anywhere under /kaggle/input or /kaggle/working,
puts it on sys.path, and returns (pkg_parent, project_root).

Why this is needed: the deliverable zip has the project root inside it, so a
dataset upload nests as
    /kaggle/input/<slug>/optiona_sfr/            <- project root
    /kaggle/input/<slug>/optiona_sfr/optiona_sfr <- the actual package
and `sys.path` must contain the FIRST of those, not the second. Different
upload routes (zip auto-extract, folder upload, GitHub import) nest
differently, so we search rather than assume.
"""
from __future__ import annotations
import sys
from pathlib import Path

SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working", "."]


def bootstrap(extra_roots=(), verbose=True):
    seen = []
    for root in list(extra_roots) + SEARCH_ROOTS:
        root = Path(root)
        if not root.exists():
            continue
        # depth-limited search: dataset mounts are shallow
        for init in sorted(root.glob("*/optiona_sfr/__init__.py")) + \
                sorted(root.glob("*/*/optiona_sfr/__init__.py")) + \
                sorted(root.glob("*/*/*/optiona_sfr/__init__.py")) + \
                sorted(root.glob("optiona_sfr/__init__.py")):
            pkg = init.parent            # .../optiona_sfr  (the package)
            parent = pkg.parent          # must go on sys.path
            seen.append(pkg)
            # sanity: the package must contain our modules
            if not (pkg / "config.py").exists():
                continue
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            project_root = parent if (parent / "tests").exists() else parent
            if verbose:
                print(f"[bootstrap] package : {pkg}")
                print(f"[bootstrap] sys.path: {parent}")
            return parent, project_root
    msg = ["Could not locate the optiona_sfr package.",
           f"Searched under: {SEARCH_ROOTS + list(extra_roots)}"]
    if seen:
        msg.append(f"Found candidates without config.py: {seen}")
    msg.append("Check that the dataset is attached, then run:")
    msg.append("  !find /kaggle/input -name '__init__.py' -path '*optiona_sfr*'")
    raise ModuleNotFoundError("\n".join(msg))


if __name__ == "__main__":
    bootstrap()
