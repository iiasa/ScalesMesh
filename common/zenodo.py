"""Download model checkpoints published on Zenodo.

Model-agnostic: used by ``scales.dit`` today, and by ``scales.ssm``/``mesh``
once their checkpoints are published too. Zenodo deposits are addressed by
record ID; the file listing and per-file MD5 checksums come from the public
Zenodo REST API, so a checkpoint is verified against its published checksum
on download and again on every cache hit.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api/records"


def _record_id(record: str | int) -> str:
    """Accept a bare record ID, a Zenodo URL, or a Zenodo DOI, and return the ID."""
    m = re.search(r"(\d+)\s*$", str(record))
    if not m:
        raise ValueError(f"could not find a Zenodo record ID in {record!r}")
    return m.group(1)


def _default_cache_dir() -> Path:
    return Path(os.environ.get("SCALESMESH_CACHE", Path.home() / ".cache" / "scalesmesh")) / "zenodo"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(record: str | int) -> list[dict]:
    """List the files (key, size, checksum, download link) in a Zenodo record."""
    resp = requests.get(f"{ZENODO_API}/{_record_id(record)}", timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def download_from_zenodo(
    record: str | int,
    filename: str | None = None,
    dest_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download one file from a Zenodo record, caching it locally.

    ``record`` is a Zenodo record ID, DOI (e.g. ``"10.5281/zenodo.1234567"``)
    or URL. ``filename`` selects which file to fetch when the record holds
    more than one; if omitted and the record has exactly one file, that file
    is used. A cached copy is reused (and re-verified) unless ``force=True``.

    Returns the local path to the downloaded file.
    """
    record_id = _record_id(record)
    files = list_files(record_id)
    if not files:
        raise ValueError(f"Zenodo record {record_id} has no files")

    if filename is None:
        if len(files) != 1:
            names = [f["key"] for f in files]
            raise ValueError(f"Zenodo record {record_id} has multiple files, pass `filename`: {names}")
        entry = files[0]
    else:
        matches = [f for f in files if f["key"] == filename]
        if not matches:
            names = [f["key"] for f in files]
            raise ValueError(f"{filename!r} not found in Zenodo record {record_id}; available: {names}")
        entry = matches[0]

    dest_dir = Path(dest_dir) if dest_dir is not None else _default_cache_dir() / record_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / entry["key"]

    checksum = entry.get("checksum", "")
    expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None

    if dest_path.exists() and not force:
        if expected_md5 is None or _md5(dest_path) == expected_md5:
            return dest_path
        print(f"[zenodo] cached file {dest_path} failed checksum verification, re-downloading")

    url = entry["links"]["self"]
    print(f"[zenodo] downloading {entry['key']} ({entry.get('size', 0) / 1e6:.1f} MB) from record {record_id}")
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if expected_md5 is not None and _md5(tmp_path) != expected_md5:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"downloaded {entry['key']} failed checksum verification against the Zenodo record")
    tmp_path.rename(dest_path)
    print(f"[zenodo] saved to {dest_path}")
    return dest_path
