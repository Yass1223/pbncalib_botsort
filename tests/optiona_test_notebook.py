"""Structural check of the Kaggle notebook.

A previous release shipped a notebook whose Stage 1 never called
shard_cache_all, so the detection cache stayed empty and Stage 3 aborted.
Prose cannot be trusted to catch that; this test asserts the invariants.
"""
import json, pathlib, sys

NB = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / \
     "optiona_kaggle_pipeline.ipynb"


def _src():
    nb = json.loads(NB.read_text())
    return [''.join(c.get("source", [])) for c in nb["cells"]]


def test_notebook_is_valid_json_and_has_all_stages():
    src = _src()
    required = ["_bootstrap", "ensure_gsr_data", "shard_cache_all",
                "harvest_labels", "estimate_tripod",
                "parallel_run_experiments", "summarize"]
    for r in required:
        assert any(r in s for s in src), f"notebook is missing '{r}'"


def test_caching_precedes_consumers():
    src = _src()
    i_cache = next(i for i, s in enumerate(src) if "shard_cache_all" in s)
    for consumer in ["estimate_tripod", "parallel_run_experiments"]:
        i = next(i for i, s in enumerate(src) if consumer in s)
        assert i_cache < i, f"'{consumer}' runs before detections are cached"


def test_empty_cache_is_asserted_not_silently_accepted():
    src = _src()
    assert any("assert ids" in s or "assert seq_ids" in s for s in src), \
        "no guard against an empty detection cache"




def test_suite_runners_are_last_in_file():
    """Tests appended after the __main__ block never run. This silently hid
    newly added regressions more than once."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    for f in sorted(root.glob("test_*.py")):
        src = f.read_text()
        i = src.find('if __name__ ==')
        if i < 0:
            continue
        tail = src[i:]
        assert "\ndef test_" not in tail, \
            f"{f.name}: tests defined AFTER the __main__ runner will not run"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
