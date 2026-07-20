"""Content-addressable storage for attachment payloads.

Layout::

    <attachments_root>/<sha256[0:2]>/<sha256><ext>

The 2-char shard prefix prevents flat-directory pathologies once the
store grows past a few thousand blobs. Writes are atomic via
temp-file + rename, lifted from
:func:`althing.persistence._atomic_write`. A blob whose sha256 is
already on disk is left untouched — CAS dedups across panel runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def _data_dir() -> Path:
    return Path(os.environ.get("SYNTH_PANEL_DATA_DIR", "~/.althing")).expanduser()


def attachments_dir() -> Path:
    """Return the CAS root, creating it if needed.

    Honors ``SYNTH_PANEL_ATTACHMENT_DIR`` for users wanting CAS on a
    different volume (e.g. faster scratch disk), then falls back to
    ``$SYNTH_PANEL_DATA_DIR/attachments`` (default
    ``~/.althing/attachments``).
    """
    override = os.environ.get("SYNTH_PANEL_ATTACHMENT_DIR", "").strip()
    root = Path(override).expanduser() if override else _data_dir() / "attachments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def refs_path(results_dir: Path, result_id: str) -> Path:
    """Return the per-result refs.json path for *result_id*.

    The parent directory is *not* created here; callers create it only
    when there are refs to write.
    """
    return results_dir / f"{result_id}.attachments" / "refs.json"


def _shard_path(root: Path, sha256: str, ext: str) -> Path:
    if len(sha256) < 2:
        raise ValueError(f"Invalid sha256 (too short): {sha256!r}")
    suffix = ext if not ext or ext.startswith(".") else f".{ext}"
    return root / sha256[:2] / f"{sha256}{suffix}"


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        return ""
    if not ext.startswith("."):
        ext = "." + ext
    if any(c in ext for c in "/\\"):
        raise ValueError(f"Invalid attachment extension: {ext!r}")
    return ext


def write_blob(content: bytes, *, ext: str = "") -> str:
    """Write *content* into the CAS and return its sha256 hex digest.

    The blob path is determined by the digest, so calling ``write_blob``
    twice with identical bytes is a no-op on the second call (dedup
    across runs). Writes go to a sibling temp file with a ``.sp-`` prefix
    and atomically replace the final path so a crash mid-write can never
    leave a half-blob in CAS.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("attachment content must be bytes-like")
    digest = hashlib.sha256(bytes(content)).hexdigest()
    safe_ext = _normalize_ext(ext)
    final = _shard_path(attachments_dir(), digest, safe_ext)
    if final.exists():
        return digest

    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(final.parent), suffix=".tmp", prefix=".sp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(content))
        shutil.move(tmp, str(final))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return digest


def read_blob(sha256: str, *, ext: str = "") -> bytes:
    """Return the bytes of the CAS-stored blob with digest *sha256*.

    Raises :class:`FileNotFoundError` when the blob is absent.
    """
    safe_ext = _normalize_ext(ext)
    path = _shard_path(attachments_dir(), sha256, safe_ext)
    if not path.exists() and not safe_ext:
        match = next(iter(path.parent.glob(f"{sha256}*")), None) if path.parent.exists() else None
        if match is None:
            raise FileNotFoundError(f"Attachment not found: {sha256}")
        return match.read_bytes()
    return path.read_bytes()
