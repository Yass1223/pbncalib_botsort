"""Environment-setup regressions.

Real-data failure (Kaggle): both GPU workers died with
'PytorchStreamReader failed reading zip archive: failed finding central
directory' — the line-detector checkpoint was truncated. Because
setup_pnlcalib treated EXISTENCE as validity, no run ever re-downloaded it.
"""
import io, os, sys, pathlib, tempfile, zipfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from optiona_sfr.kaggle_setup import verify_checkpoint


def _tmp():
    return pathlib.Path(tempfile.mkdtemp())


def test_rejects_every_realistic_corruption_mode():
    d = _tmp()
    assert not verify_checkpoint(d / "does_not_exist")[0]

    (d / "tiny").write_bytes(b"x" * 100)
    assert not verify_checkpoint(d / "tiny")[0]

    (d / "html").write_bytes(b"<!DOCTYPE html><html>404</html>" + b" " * 6_000_000)
    ok, why = verify_checkpoint(d / "html")
    assert not ok and "HTML" in why

    # the exact observed failure: PK header present, archive truncated
    (d / "trunc").write_bytes(b"PK\x03\x04" + os.urandom(6_000_000))
    assert not verify_checkpoint(d / "trunc")[0]


def test_accepts_a_valid_archive():
    d = _tmp()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.pkl", b"0" * 6_000_000)
    (d / "good").write_bytes(buf.getvalue())
    ok, why = verify_checkpoint(d / "good")
    assert ok, why


def test_partial_download_cannot_masquerade_as_valid():
    """download_checkpoint writes to <dst>.part and only renames after the
    file verifies, so an interrupted fetch leaves no plausible-looking file."""
    from optiona_sfr.kaggle_setup import download_checkpoint
    d = _tmp()
    dst = d / "SV_lines"
    ok = download_checkpoint("http://127.0.0.1:9/nonexistent", dst, retries=1,
                             verbose=False)
    assert ok is False
    assert not dst.exists(), "a failed download must not leave the target file"
    assert not (d / "SV_lines.part").exists(), "temp file must be cleaned up"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
