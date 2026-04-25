from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable

import numpy as np

from .config import DataConfig, ModelConfig
from .dnagpt_loader import build_dnagpt_tokenizer, get_variant_spec
from .dnagpt_tokenization import resolve_species_prefix_token
from .tokenization import normalize_alphabet, resolve_vocab_and_special_ids, tokenize_source_array


NUGGET_TOKENIZERS = ("byte", "fixed_kmer", "dnagpt_kmer")


@dataclass(frozen=True)
class NuggetTokenizerSpec:
    name: str
    vocab_size: int
    pad_id: int
    eos_id: int | None
    decoder_start_token_id: int
    token_merge_size: int
    token_merge_alphabet: str
    dnagpt_variant: str | None = None
    dnagpt_kmer_size: int | None = None
    dnagpt_dynamic_kmer: bool = False
    tokenizer: object | None = None


@dataclass(frozen=True)
class NuggetTokenizedSource:
    species: str
    source_name: str | None
    prefix_token: str | None
    prefix_ids: list[int]
    dna_token_ids: np.ndarray
    dna_token_base_lengths: np.ndarray
    tail_sequence: str
    total_bases: int


@dataclass(frozen=True)
class NuggetTokenCacheStats:
    enabled: bool
    cache_dir: str | None
    hits: int
    created: int
    rebuilt: int
    disabled: int
    load_seconds: float = 0.0
    build_seconds: float = 0.0
    write_seconds: float = 0.0


_NUGGET_TOKEN_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class NuggetCacheSourceDescriptor:
    species: str
    source_name: str | None
    payload: bytes
    clean_cache_path: str | None = None
    source_path: str | None = None
    split_start: int | None = None
    split_length: int | None = None


def build_nugget_tokenizer_spec(data_config: DataConfig, model_config: ModelConfig) -> NuggetTokenizerSpec:
    tokenizer_name = data_config.nugget_tokenizer
    if tokenizer_name not in NUGGET_TOKENIZERS:
        raise ValueError(
            f"data.nugget_tokenizer must be one of: {', '.join(NUGGET_TOKENIZERS)}"
        )

    if tokenizer_name == "byte":
        return NuggetTokenizerSpec(
            name="byte",
            vocab_size=259,
            pad_id=257,
            eos_id=258,
            decoder_start_token_id=257,
            token_merge_size=1,
            token_merge_alphabet=data_config.token_merge_alphabet,
        )

    if tokenizer_name == "fixed_kmer":
        resolved = resolve_vocab_and_special_ids(
            data_config.token_merge_size,
            data_config.token_merge_alphabet,
        )
        if resolved is None:
            raise ValueError("fixed_kmer tokenizer requires data.token_merge_size > 1")
        vocab_size, pad_id, eos_id = resolved
        return NuggetTokenizerSpec(
            name="fixed_kmer",
            vocab_size=vocab_size,
            pad_id=pad_id,
            eos_id=eos_id,
            decoder_start_token_id=pad_id,
            token_merge_size=data_config.token_merge_size,
            token_merge_alphabet=normalize_alphabet(data_config.token_merge_alphabet),
        )

    tokenizer = build_dnagpt_tokenizer(model_config.variant)
    spec = get_variant_spec(model_config.variant)
    return NuggetTokenizerSpec(
        name="dnagpt_kmer",
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id,
        eos_id=None,
        decoder_start_token_id=tokenizer.pad_id,
        token_merge_size=spec.kmer_size,
        token_merge_alphabet="NAGCT",
        dnagpt_variant=spec.variant,
        dnagpt_kmer_size=spec.kmer_size,
        dnagpt_dynamic_kmer=spec.dynamic_kmer,
        tokenizer=tokenizer,
    )


def apply_nugget_tokenizer_to_model_config(model_config: ModelConfig, spec: NuggetTokenizerSpec) -> None:
    model_config.vocab_size = spec.vocab_size
    model_config.pad_id = spec.pad_id
    if spec.eos_id is not None:
        model_config.eos_id = spec.eos_id


def _normalize_sequence(source: bytes) -> str:
    return source.decode("ascii").strip().upper()


def _compact_unsigned_dtype(max_value: int):
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _token_id_dtype(spec: NuggetTokenizerSpec):
    return _compact_unsigned_dtype(max(spec.vocab_size - 1, 0))


def _base_length_dtype(max_length: int):
    return _compact_unsigned_dtype(max(max_length, 1))


def _tokenize_fixed_kmer_source(
    *,
    species: str,
    source_name: str | None,
    source: bytes,
    spec: NuggetTokenizerSpec,
) -> NuggetTokenizedSource:
    alphabet = normalize_alphabet(spec.token_merge_alphabet)
    allowed = set(alphabet)
    sequence = "".join(base for base in _normalize_sequence(source) if base in allowed)
    full_base_count = (len(sequence) // spec.token_merge_size) * spec.token_merge_size
    payload = sequence[:full_base_count].encode("ascii")
    tail_sequence = sequence[full_base_count:]
    ids = tokenize_source_array(payload, spec.token_merge_size, alphabet).astype(_token_id_dtype(spec), copy=False)
    return NuggetTokenizedSource(
        species=species,
        source_name=source_name,
        prefix_token=None,
        prefix_ids=[],
        dna_token_ids=ids,
        dna_token_base_lengths=np.full(ids.shape, spec.token_merge_size, dtype=_base_length_dtype(spec.token_merge_size)),
        tail_sequence=tail_sequence,
        total_bases=len(sequence),
    )


def _tokenize_dnagpt_source(
    *,
    species: str,
    source_name: str | None,
    source: bytes,
    spec: NuggetTokenizerSpec,
    species_prefix_map: dict[str, str] | None,
    drop_tail_to_full_kmer: bool,
) -> NuggetTokenizedSource:
    if spec.tokenizer is None or spec.dnagpt_kmer_size is None:
        raise ValueError("dnagpt_kmer tokenizer spec is missing DNAGPT tokenizer metadata")
    prefix_token = resolve_species_prefix_token(species, species_prefix_map)
    prefix_ids: list[int] = []
    if prefix_token is not None:
        token_text = prefix_token if prefix_token.startswith("<") else f"<{prefix_token}>"
        token_id = spec.tokenizer.piece_to_id(token_text)
        if token_id == spec.tokenizer.unk_id and token_text != spec.tokenizer.id_to_piece(spec.tokenizer.unk_id):
            raise ValueError(
                f"Unknown DNAGPT special token '{prefix_token}' configured for species '{species}'."
            )
        prefix_ids.append(int(token_id))

    sequence = _normalize_sequence(source)
    pieces = [
        sequence[index : index + spec.dnagpt_kmer_size]
        for index in range(0, len(sequence), spec.dnagpt_kmer_size)
    ]
    tail_sequence = ""
    if drop_tail_to_full_kmer and pieces and len(pieces[-1]) < spec.dnagpt_kmer_size:
        tail_sequence = pieces[-1]
        pieces = pieces[:-1]
    token_ids = [int(spec.tokenizer.piece_to_id(piece)) for piece in pieces if piece]
    base_lengths = [len(piece) for piece in pieces if piece]
    return NuggetTokenizedSource(
        species=species,
        source_name=source_name,
        prefix_token=prefix_token,
        prefix_ids=prefix_ids,
        dna_token_ids=np.asarray(token_ids, dtype=_token_id_dtype(spec)),
        dna_token_base_lengths=np.asarray(base_lengths, dtype=_base_length_dtype(spec.dnagpt_kmer_size)),
        tail_sequence=tail_sequence,
        total_bases=len(sequence),
    )


def tokenize_nugget_source(
    *,
    species: str,
    source_name: str | None = None,
    source: bytes,
    spec: NuggetTokenizerSpec,
    species_prefix_map: dict[str, str] | None = None,
    drop_tail_to_full_kmer: bool = False,
) -> NuggetTokenizedSource:
    if spec.name == "byte":
        ids = np.frombuffer(source, dtype=np.uint8).astype(_token_id_dtype(spec), copy=True)
        return NuggetTokenizedSource(
            species=species,
            source_name=source_name,
            prefix_token=None,
            prefix_ids=[],
            dna_token_ids=ids,
            dna_token_base_lengths=np.ones(ids.shape, dtype=np.uint8),
            tail_sequence="",
            total_bases=len(source),
        )
    if spec.name == "fixed_kmer":
        return _tokenize_fixed_kmer_source(species=species, source_name=source_name, source=source, spec=spec)
    if spec.name == "dnagpt_kmer":
        return _tokenize_dnagpt_source(
            species=species,
            source_name=source_name,
            source=source,
            spec=spec,
            species_prefix_map=species_prefix_map,
            drop_tail_to_full_kmer=drop_tail_to_full_kmer,
        )
    raise ValueError(f"Unsupported Nugget tokenizer '{spec.name}'.")


def tokenize_nugget_sources(
    *,
    source_descriptors: Iterable[NuggetCacheSourceDescriptor],
    spec: NuggetTokenizerSpec,
    species_prefix_map: dict[str, str] | None = None,
    drop_tail_to_full_kmer: bool = False,
) -> list[NuggetTokenizedSource]:
    return [
        tokenize_nugget_source(
            species=descriptor.species,
            source_name=descriptor.source_name,
            source=descriptor.payload,
            spec=spec,
            species_prefix_map=species_prefix_map,
            drop_tail_to_full_kmer=drop_tail_to_full_kmer,
        )
        for descriptor in source_descriptors
    ]


def resolve_nugget_token_cache_root(dataset_dir: Path, configured_dir: str | None) -> Path:
    if configured_dir is None:
        return dataset_dir / ".dna_cache" / "nugget_tokens"
    configured_path = Path(configured_dir)
    if configured_path.is_absolute():
        return configured_path / "nugget_tokens"
    return dataset_dir / configured_path / "nugget_tokens"


def _write_atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.tmp.", suffix=".npz", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _write_atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f"{path.name}.tmp.",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(payload)
    os.replace(tmp_path, path)


def _load_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _spec_signature(spec: NuggetTokenizerSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "vocab_size": spec.vocab_size,
        "pad_id": spec.pad_id,
        "eos_id": spec.eos_id,
        "decoder_start_token_id": spec.decoder_start_token_id,
        "token_merge_size": spec.token_merge_size,
        "token_merge_alphabet": spec.token_merge_alphabet,
        "dnagpt_variant": spec.dnagpt_variant,
        "dnagpt_kmer_size": spec.dnagpt_kmer_size,
        "dnagpt_dynamic_kmer": spec.dnagpt_dynamic_kmer,
    }


def _file_identity(path_value: str | None) -> dict[str, object] | None:
    if path_value is None:
        return None
    path = Path(path_value)
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_key(
    *,
    descriptor: NuggetCacheSourceDescriptor,
    spec: NuggetTokenizerSpec,
    species_prefix_map: dict[str, str] | None,
    drop_tail_to_full_kmer: bool,
    split_scope: str | None,
) -> tuple[str | None, str, str | None, str]:
    clean_identity = _file_identity(descriptor.clean_cache_path)
    source_identity = _file_identity(descriptor.source_path)
    source_sha: str | None = None
    if clean_identity is None and source_identity is None:
        source_sha = hashlib.sha256(descriptor.payload).hexdigest()
    prefix_token = resolve_species_prefix_token(descriptor.species, species_prefix_map) if spec.name == "dnagpt_kmer" else None
    key_payload = {
        "schema_version": _NUGGET_TOKEN_CACHE_SCHEMA_VERSION,
        "species": descriptor.species,
        "source_name": descriptor.source_name,
        "source_sha256": source_sha,
        "source_len": len(descriptor.payload),
        "source_path": descriptor.source_path,
        "source_identity": source_identity,
        "clean_cache_path": descriptor.clean_cache_path,
        "clean_cache_identity": clean_identity,
        "split_scope": split_scope,
        "split_start": descriptor.split_start,
        "split_length": descriptor.split_length,
        "spec": _spec_signature(spec),
        "prefix_token": prefix_token,
        "drop_tail_to_full_kmer": bool(drop_tail_to_full_kmer),
    }
    key_json = json.dumps(key_payload, ensure_ascii=True, sort_keys=True)
    key_sha = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    return source_sha, key_sha, prefix_token, key_json


def _cache_paths(cache_root: Path, species: str, key_sha: str) -> tuple[Path, Path]:
    safe_species = species.replace("/", "_")
    data_path = cache_root / safe_species / f"{key_sha}.npz"
    metadata_path = cache_root / safe_species / f"{key_sha}.json"
    return data_path, metadata_path


def _load_cached_source(data_path: Path, metadata: dict[str, object]) -> NuggetTokenizedSource:
    with np.load(data_path, allow_pickle=False) as payload:
        prefix_ids = payload["prefix_ids"].astype(np.int64, copy=False).tolist()
        dna_token_ids = payload["dna_token_ids"].copy()
        dna_token_base_lengths = payload["dna_token_base_lengths"].copy()
    return NuggetTokenizedSource(
        species=str(metadata["species"]),
        source_name=metadata.get("source_name"),
        prefix_token=metadata.get("prefix_token"),
        prefix_ids=[int(item) for item in prefix_ids],
        dna_token_ids=dna_token_ids,
        dna_token_base_lengths=dna_token_base_lengths,
        tail_sequence=str(metadata.get("tail_sequence", "")),
        total_bases=int(metadata["total_bases"]),
    )


def tokenize_nugget_sources_with_cache(
    *,
    source_descriptors: Iterable[NuggetCacheSourceDescriptor],
    spec: NuggetTokenizerSpec,
    dataset_dir: Path,
    cache_enabled: bool,
    cache_dir: str | None,
    species_prefix_map: dict[str, str] | None = None,
    drop_tail_to_full_kmer: bool = False,
    split_scope: str | None = None,
) -> tuple[list[NuggetTokenizedSource], NuggetTokenCacheStats]:
    cache_root = resolve_nugget_token_cache_root(dataset_dir, cache_dir)
    tokenized: list[NuggetTokenizedSource] = []
    hits = 0
    created = 0
    rebuilt = 0
    disabled = 0
    load_seconds = 0.0
    build_seconds = 0.0
    write_seconds = 0.0

    for descriptor in source_descriptors:
        source_sha, key_sha, prefix_token, key_json = _cache_key(
            descriptor=descriptor,
            spec=spec,
            species_prefix_map=species_prefix_map,
            drop_tail_to_full_kmer=drop_tail_to_full_kmer,
            split_scope=split_scope,
        )
        data_path, metadata_path = _cache_paths(cache_root, descriptor.species, key_sha)

        if cache_enabled:
            metadata = _load_metadata(metadata_path)
            if (
                metadata is not None
                and int(metadata.get("schema_version", -1)) == _NUGGET_TOKEN_CACHE_SCHEMA_VERSION
                and data_path.exists()
                and metadata.get("key") == key_json
            ):
                try:
                    started = time.perf_counter()
                    tokenized.append(_load_cached_source(data_path, metadata))
                    load_seconds += time.perf_counter() - started
                    hits += 1
                    continue
                except Exception:
                    # Fall through and rebuild cache entry when payload is corrupted.
                    pass

        started = time.perf_counter()
        built = tokenize_nugget_source(
            species=descriptor.species,
            source_name=descriptor.source_name,
            source=descriptor.payload,
            spec=spec,
            species_prefix_map=species_prefix_map,
            drop_tail_to_full_kmer=drop_tail_to_full_kmer,
        )
        build_seconds += time.perf_counter() - started
        tokenized.append(built)

        if not cache_enabled:
            disabled += 1
            continue

        existed = data_path.exists() or metadata_path.exists()
        started = time.perf_counter()
        _write_atomic_npz(
            data_path,
            prefix_ids=np.asarray(built.prefix_ids, dtype=_token_id_dtype(spec)),
            dna_token_ids=built.dna_token_ids,
            dna_token_base_lengths=built.dna_token_base_lengths,
        )
        metadata = {
            "schema_version": _NUGGET_TOKEN_CACHE_SCHEMA_VERSION,
            "species": descriptor.species,
            "source_name": descriptor.source_name,
            "source_sha256": source_sha,
            "source_len": len(descriptor.payload),
            "source_path": descriptor.source_path,
            "clean_cache_path": descriptor.clean_cache_path,
            "split_scope": split_scope,
            "split_start": descriptor.split_start,
            "split_length": descriptor.split_length,
            "prefix_token": prefix_token,
            "tail_sequence": built.tail_sequence,
            "total_bases": built.total_bases,
            "key": key_json,
            "data_path": str(data_path),
        }
        _write_atomic_text(metadata_path, json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2))
        write_seconds += time.perf_counter() - started
        if existed:
            rebuilt += 1
        else:
            created += 1

    return tokenized, NuggetTokenCacheStats(
        enabled=bool(cache_enabled),
        cache_dir=str(cache_root),
        hits=hits,
        created=created,
        rebuilt=rebuilt,
        disabled=disabled,
        load_seconds=load_seconds,
        build_seconds=build_seconds,
        write_seconds=write_seconds,
    )
