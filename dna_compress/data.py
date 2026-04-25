from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re

import torch
from torch.utils.data import Dataset

from .clean_cache import load_or_build_clean_sequence, resolve_clean_cache_root
from .config import DataConfig
from .tokenization import normalize_alphabet, tokenize_source_array, tokenize_source_bytes


@dataclass
class LoadedSplits:
    train_sources: list[bytes]
    val_sources: list[bytes]
    test_sources: list[bytes]
    summary: dict[str, object]


@dataclass
class SourceRecord:
    species: str
    source_name: str
    source_mode: str
    source_path: str | None
    sequence_keys: list[str]
    sequence_files: list[str]
    observed_bytes: set[int]
    cache_status: str
    cache_counts: dict[str, int]
    cache_path: str | None
    raw_bytes: bytes


_CHROMOSOME_RE = re.compile(r"\.dna\.chromosome\.([^.]+)\.fa(?:\.gz)?$", re.IGNORECASE)
_PRIMARY_RE = re.compile(r"\.dna\.primary_assembly\.([^.]+)\.fa(?:\.gz)?$", re.IGNORECASE)
_NONCHROM_RE = re.compile(r"\.dna\.nonchromosomal\.fa(?:\.gz)?$", re.IGNORECASE)


def _validate_data_config(config: DataConfig) -> None:
    if config.sequence_source_mode not in {"auto", "flat_file", "fasta_dir"}:
        raise ValueError("data.sequence_source_mode must be one of: auto, flat_file, fasta_dir")
    if config.multi_sequence_mode not in {"separate", "concat"}:
        raise ValueError("data.multi_sequence_mode must be one of: separate, concat")
    if not isinstance(config.clean_cache_enabled, bool):
        raise ValueError("data.clean_cache_enabled must be a bool")
    if config.clean_cache_dir is not None and (not isinstance(config.clean_cache_dir, str) or not config.clean_cache_dir.strip()):
        raise ValueError("data.clean_cache_dir must be null or a non-empty string")
    if not isinstance(config.sequence_include_map, dict):
        raise ValueError("data.sequence_include_map must be a dict[str, list[str]]")
    for species, values in config.sequence_include_map.items():
        if not isinstance(species, str) or not species:
            raise ValueError("data.sequence_include_map keys must be non-empty species strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"data.sequence_include_map[{species!r}] must be a non-empty list")
        if any((not isinstance(item, str) or not item.strip()) for item in values):
            raise ValueError(f"data.sequence_include_map[{species!r}] must contain only non-empty strings")


def _sequence_key_from_path(path: Path) -> str:
    name = path.name
    match = _CHROMOSOME_RE.search(name)
    if match is not None:
        return match.group(1)
    match = _PRIMARY_RE.search(name)
    if match is not None:
        return match.group(1)
    if _NONCHROM_RE.search(name) is not None:
        return "nonchromosomal"
    return path.name.removesuffix(".gz").removesuffix(".fa")


def _boundary_bytes(config: DataConfig, seq_length: int | None) -> bytes:
    alphabet = normalize_alphabet(config.token_merge_alphabet)
    if "N" not in alphabet:
        return b""
    boundary_length = max(1, config.token_merge_size) * max(1, seq_length or 1)
    return b"N" * boundary_length


def _discover_fasta_files(species_path: Path) -> list[Path]:
    dna_dir = species_path / "dna"
    candidates: list[Path] = []
    if dna_dir.is_dir():
        candidates.extend(sorted(dna_dir.glob("*.fa")))
        candidates.extend(sorted(dna_dir.glob("*.fa.gz")))
    if not candidates and species_path.is_dir():
        candidates.extend(sorted(species_path.glob("*.fa")))
        candidates.extend(sorted(species_path.glob("*.fa.gz")))
    return sorted(candidates)


def _resolve_sequence_source_mode(species_path: Path, config: DataConfig) -> str:
    if config.sequence_source_mode == "flat_file":
        if not species_path.is_file():
            raise FileNotFoundError(f"Expected flat file for species at {species_path}")
        return "flat_file"
    if config.sequence_source_mode == "fasta_dir":
        if not _discover_fasta_files(species_path):
            raise FileNotFoundError(f"Expected FASTA directory for species at {species_path}")
        return "fasta_dir"

    if species_path.is_file():
        return "flat_file"
    if _discover_fasta_files(species_path):
        return "fasta_dir"
    raise FileNotFoundError(f"Could not resolve data source for species at {species_path}")


def _load_source_records_for_species(config: DataConfig, species: str, seq_length: int | None) -> list[SourceRecord]:
    species_path = Path(config.dataset_dir) / species
    source_mode = _resolve_sequence_source_mode(species_path, config)

    if source_mode == "flat_file":
        raw_bytes = species_path.read_bytes()
        return [
            SourceRecord(
                species=species,
                source_name=species,
                source_mode="flat_file",
                source_path=str(species_path),
                sequence_keys=[species],
                sequence_files=[species_path.name],
                observed_bytes=set(raw_bytes),
                cache_status="not_applicable",
                cache_counts={},
                cache_path=None,
                raw_bytes=raw_bytes,
            )
        ]

    fasta_files = _discover_fasta_files(species_path)
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found for species {species} under {species_path}")

    selected_keys = config.sequence_include_map.get(species)
    selected_set = {item.strip() for item in selected_keys} if selected_keys is not None else None
    sequence_entries: list[tuple[str, Path, bytes, set[int], str, str | None]] = []
    matched_keys: set[str] = set()
    for fasta_path in fasta_files:
        sequence_key = _sequence_key_from_path(fasta_path)
        if selected_set is not None and sequence_key not in selected_set:
            continue
        clean_result = load_or_build_clean_sequence(
            source_path=fasta_path,
            dataset_dir=Path(config.dataset_dir),
            alphabet=config.token_merge_alphabet,
            cache_enabled=config.clean_cache_enabled,
            clean_cache_dir=config.clean_cache_dir,
        )
        payload = clean_result.payload
        observed_bytes = clean_result.observed_bytes
        if len(payload) == 0:
            continue
        sequence_entries.append(
            (
                sequence_key,
                fasta_path,
                payload,
                observed_bytes,
                clean_result.cache_status,
                clean_result.cache_path,
            )
        )
        matched_keys.add(sequence_key)

    if selected_set is not None:
        missing = sorted(selected_set - matched_keys)
        if missing:
            raise ValueError(f"{species} requested sequence keys not found: {', '.join(missing)}")

    if not sequence_entries:
        raise ValueError(f"{species} does not have any usable FASTA sequence after filtering")

    if config.multi_sequence_mode == "separate":
        records: list[SourceRecord] = []
        for sequence_key, fasta_path, payload, observed_bytes, cache_status, cache_path in sequence_entries:
            records.append(
                SourceRecord(
                    species=species,
                    source_name=f"{species}:{sequence_key}",
                    source_mode="fasta_dir_separate",
                    source_path=str(fasta_path),
                    sequence_keys=[sequence_key],
                    sequence_files=[str(fasta_path.relative_to(species_path))],
                    observed_bytes=set(observed_bytes),
                    cache_status=cache_status,
                    cache_counts={cache_status: 1},
                    cache_path=cache_path,
                    raw_bytes=payload,
                )
            )
        return records

    boundary = _boundary_bytes(config, seq_length)
    joined_parts: list[bytes] = []
    sequence_keys: list[str] = []
    sequence_files: list[str] = []
    observed_bytes: set[int] = set()
    cache_statuses: set[str] = set()
    cache_counts = {
        "hit": 0,
        "created": 0,
        "rebuilt": 0,
        "disabled": 0,
    }
    cache_paths: list[str] = []
    for index, (sequence_key, fasta_path, payload, payload_observed_bytes, cache_status, cache_path) in enumerate(sequence_entries):
        if index > 0 and boundary:
            joined_parts.append(boundary)
        joined_parts.append(payload)
        sequence_keys.append(sequence_key)
        sequence_files.append(str(fasta_path.relative_to(species_path)))
        observed_bytes.update(payload_observed_bytes)
        cache_statuses.add(cache_status)
        if cache_status in cache_counts:
            cache_counts[cache_status] += 1
        if cache_path is not None:
            cache_paths.append(cache_path)
    if boundary:
        observed_bytes.update(boundary)
    if "rebuilt" in cache_statuses:
        combined_cache_status = "rebuilt"
    elif "created" in cache_statuses:
        combined_cache_status = "created"
    elif "disabled" in cache_statuses:
        combined_cache_status = "disabled"
    else:
        combined_cache_status = "hit"
    return [
        SourceRecord(
            species=species,
            source_name=species,
            source_mode="fasta_dir_concat",
            source_path=None,
            sequence_keys=sequence_keys,
            sequence_files=sequence_files,
            observed_bytes=observed_bytes,
            cache_status=combined_cache_status,
            cache_counts=cache_counts,
            cache_path=cache_paths[0] if cache_paths else None,
            raw_bytes=b"".join(joined_parts),
        )
    ]


def _read_slice(path: Path, start: int, length: int | None) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read() if length is None else handle.read(length)


def _split_points(length: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")
    train_end = int(length * train_ratio)
    val_end = train_end + int(length * val_ratio)
    return train_end, val_end


def load_splits(config: DataConfig, seq_length: int | None = None) -> LoadedSplits:
    _validate_data_config(config)
    train_sources: list[bytes] = []
    val_sources: list[bytes] = []
    test_sources: list[bytes] = []
    species_summary: list[dict[str, int | str]] = []
    all_observed: set[int] = set()
    clean_cache_summary = {
        "enabled": config.clean_cache_enabled,
        "cache_dir": None,
        "applicable_sources": 0,
        "hits": 0,
        "created": 0,
        "rebuilt": 0,
        "disabled": 0,
    }
    if config.clean_cache_enabled:
        clean_cache_summary["cache_dir"] = str(resolve_clean_cache_root(Path(config.dataset_dir), config.clean_cache_dir))
    cache_summary_key_map = {
        "hit": "hits",
        "created": "created",
        "rebuilt": "rebuilt",
        "disabled": "disabled",
    }

    for species in config.species:
        for record in _load_source_records_for_species(config, species, seq_length):
            total_size = len(record.raw_bytes)
            train_end, val_end = _split_points(
                total_size,
                config.train_ratio,
                config.val_ratio,
                config.test_ratio,
            )

            train_size = min(
                train_end,
                config.max_train_bytes_per_species or train_end,
            )
            val_available = max(val_end - train_end, 0)
            val_size = min(
                val_available,
                config.max_val_bytes_per_species or val_available,
            )
            test_available = max(total_size - val_end, 0)
            test_size = min(
                test_available,
                config.max_test_bytes_per_species or test_available,
            )

            train_bytes = record.raw_bytes[:train_size]
            val_bytes = record.raw_bytes[train_end : train_end + val_size]
            test_bytes = record.raw_bytes[val_end : val_end + test_size]

            if len(train_bytes) < 2:
                raise ValueError(f"{record.source_name} does not have enough train bytes for sampling")
            if len(val_bytes) < 2 or len(test_bytes) < 2:
                raise ValueError(f"{record.source_name} does not have enough validation/test bytes")

            train_sources.append(train_bytes)
            val_sources.append(val_bytes)
            test_sources.append(test_bytes)
            all_observed.update(record.observed_bytes)
            if record.source_mode != "flat_file":
                record_applicable = sum(record.cache_counts.values()) if record.cache_counts else 1
                clean_cache_summary["applicable_sources"] += record_applicable
                for cache_status, count in record.cache_counts.items():
                    summary_key = cache_summary_key_map.get(cache_status)
                    if summary_key is not None:
                        clean_cache_summary[summary_key] += count
            species_summary.append(
                {
                    "species": record.species,
                    "source_name": record.source_name,
                    "source_mode": record.source_mode,
                    "source_path": record.source_path,
                    "sequence_keys": list(record.sequence_keys),
                    "sequence_files": list(record.sequence_files),
                    "selected_sequence_count": len(record.sequence_keys),
                    "total_size": total_size,
                    "train_start": 0,
                    "train_bytes": len(train_bytes),
                    "val_start": train_end,
                    "val_bytes": len(val_bytes),
                    "test_start": val_end,
                    "test_bytes": len(test_bytes),
                    "clean_cache_status": record.cache_status,
                    "clean_cache_path": record.cache_path,
                }
            )

    return LoadedSplits(
        train_sources=train_sources,
        val_sources=val_sources,
        test_sources=test_sources,
        summary={
            "species": species_summary,
            "alphabet_bytes": sorted(all_observed),
            "clean_cache": clean_cache_summary,
        },
    )


class RandomWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[bytes],
        seq_length: int,
        samples_per_epoch: int,
        seed: int,
        sampling_strategy: str = "proportional",
        token_merge_size: int = 1,
        token_merge_alphabet: str = "ACGTN",
    ) -> None:
        self.sources = [
            tokenize_source_array(source, token_merge_size, token_merge_alphabet)
            for source in sources
        ]
        self.sources = [source for source in self.sources if source.shape[0] >= seq_length]
        if not self.sources:
            raise ValueError("no train sources are long enough for the configured seq_length")
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.available = [int(source.shape[0]) - seq_length + 1 for source in self.sources]
        self.sampling_strategy = sampling_strategy
        if self.sampling_strategy == "proportional":
            self.source_weights = [float(count) for count in self.available]
        elif self.sampling_strategy == "uniform":
            self.source_weights = [1.0 for _ in self.available]
        elif self.sampling_strategy == "sqrt":
            self.source_weights = [float(count) ** 0.5 for count in self.available]
        else:
            raise ValueError(
                "sampling_strategy must be one of: proportional, uniform, sqrt"
            )

        self.total_weight = sum(self.source_weights)
        if self.total_weight <= 0:
            raise ValueError("sampling source weights must sum to > 0")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index)
        source_index = rng.choices(range(len(self.sources)), weights=self.source_weights, k=1)[0]
        source = self.sources[source_index]
        start = rng.randrange(self.available[source_index])
        window = source[start : start + self.seq_length]
        ids = torch.as_tensor(window, dtype=torch.long)
        return {"input_ids": ids}


class SequentialWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[bytes],
        seq_length: int,
        pad_id: int,
        token_merge_size: int = 1,
        token_merge_alphabet: str = "ACGTN",
    ) -> None:
        tokenized_sources = [
            tokenize_source_array(source, token_merge_size, token_merge_alphabet)
            for source in sources
        ]
        self.sources = [source for source in tokenized_sources if source.shape[0] > 0]
        self.seq_length = seq_length
        self.pad_id = pad_id
        self.index: list[tuple[int, int]] = []

        for source_idx, source in enumerate(self.sources):
            if source.shape[0] <= seq_length:
                self.index.append((source_idx, 0))
                continue
            for start in range(0, int(source.shape[0]), seq_length):
                self.index.append((source_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_idx, start = self.index[index]
        source = self.sources[source_idx]
        chunk = source[start : start + self.seq_length]
        ids = torch.full((self.seq_length,), self.pad_id, dtype=torch.long)
        ids[: chunk.shape[0]] = torch.as_tensor(chunk, dtype=torch.long)
        return {"input_ids": ids}


def build_compression_sample(sources: list[bytes], requested_bytes: int) -> bytes:
    for source in sources:
        if len(source) >= requested_bytes:
            return source[:requested_bytes]
    longest = max(sources, key=len)
    return longest
