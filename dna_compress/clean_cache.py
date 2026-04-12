from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .fasta_cleaning import sanitize_fasta_bytes


_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CleanSequenceResult:
    payload: bytes
    observed_bytes: set[int]
    cache_status: str
    cache_path: str | None


def resolve_clean_cache_root(dataset_dir: Path, configured_dir: str | None) -> Path:
    if configured_dir is None:
        return dataset_dir / ".dna_cache" / "clean"
    configured_path = Path(configured_dir)
    if configured_path.is_absolute():
        return configured_path
    return dataset_dir / configured_path


def _cache_paths(cache_root: Path, dataset_dir: Path, source_path: Path) -> tuple[Path, Path, str]:
    source_relpath = str(source_path.relative_to(dataset_dir))
    relative_path = Path(source_relpath)
    sequence_path = cache_root / relative_path.parent / f"{relative_path.name}.clean.bin"
    metadata_path = cache_root / relative_path.parent / f"{relative_path.name}.clean.json"
    return sequence_path, metadata_path, source_relpath


def _source_identity(source_path: Path, source_relpath: str, alphabet: str) -> dict[str, Any]:
    stat = source_path.stat()
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "source_relpath": source_relpath,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "alphabet": alphabet,
    }


def _load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.tmp.", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    os.replace(temp_path, path)


def _write_atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f"{path.name}.tmp.",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    os.replace(temp_path, path)


def load_or_build_clean_sequence(
    *,
    source_path: Path,
    dataset_dir: Path,
    alphabet: str,
    cache_enabled: bool,
    clean_cache_dir: str | None,
) -> CleanSequenceResult:
    if not cache_enabled:
        payload, observed_bytes = sanitize_fasta_bytes(source_path, alphabet)
        return CleanSequenceResult(
            payload=payload,
            observed_bytes=observed_bytes,
            cache_status="disabled",
            cache_path=None,
        )

    cache_root = resolve_clean_cache_root(dataset_dir, clean_cache_dir)
    sequence_path, metadata_path, source_relpath = _cache_paths(cache_root, dataset_dir, source_path)
    identity = _source_identity(source_path, source_relpath, alphabet)
    existing_cache_files = sequence_path.exists() or metadata_path.exists()
    existing_metadata = _load_metadata(metadata_path)

    if sequence_path.exists() and existing_metadata is not None:
        metadata_matches = all(existing_metadata.get(key) == value for key, value in identity.items())
        if metadata_matches:
            payload = sequence_path.read_bytes()
            observed_raw = existing_metadata.get("observed_bytes", [])
            observed_bytes = {int(item) for item in observed_raw}
            return CleanSequenceResult(
                payload=payload,
                observed_bytes=observed_bytes,
                cache_status="hit",
                cache_path=str(sequence_path),
            )

    payload, observed_bytes = sanitize_fasta_bytes(source_path, alphabet)
    metadata = {
        **identity,
        "cleaned_length": len(payload),
        "observed_bytes": sorted(observed_bytes),
    }
    _write_atomic_bytes(sequence_path, payload)
    _write_atomic_text(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False))
    cache_status = "rebuilt" if existing_cache_files else "created"
    return CleanSequenceResult(
        payload=payload,
        observed_bytes=observed_bytes,
        cache_status=cache_status,
        cache_path=str(sequence_path),
    )
