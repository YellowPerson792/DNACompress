from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .tokenization import tokenize_source_bytes


@dataclass
class LoadedSplits:
    train_sources: list[bytes]
    val_sources: list[bytes]
    test_sources: list[bytes]
    summary: dict[str, object]


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


def load_splits(config: DataConfig) -> LoadedSplits:
    dataset_dir = Path(config.dataset_dir)
    train_sources: list[bytes] = []
    val_sources: list[bytes] = []
    test_sources: list[bytes] = []
    species_summary: list[dict[str, int | str]] = []

    for species in config.species:
        species_path = dataset_dir / species
        total_size = species_path.stat().st_size
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

        train_bytes = _read_slice(species_path, 0, train_size)
        val_bytes = _read_slice(species_path, train_end, val_size)
        test_bytes = _read_slice(species_path, val_end, test_size)

        if len(train_bytes) < 2:
            raise ValueError(f"{species_path} does not have enough train bytes for sampling")
        if len(val_bytes) < 2 or len(test_bytes) < 2:
            raise ValueError(f"{species_path} does not have enough validation/test bytes")

        train_sources.append(train_bytes)
        val_sources.append(val_bytes)
        test_sources.append(test_bytes)
        species_summary.append(
            {
                "species": species,
                "total_size": total_size,
                "train_bytes": len(train_bytes),
                "val_bytes": len(val_bytes),
                "test_bytes": len(test_bytes),
            }
        )

    all_observed = set()
    for source in train_sources + val_sources + test_sources:
        all_observed.update(source)

    return LoadedSplits(
        train_sources=train_sources,
        val_sources=val_sources,
        test_sources=test_sources,
        summary={
            "species": species_summary,
            "alphabet_bytes": sorted(all_observed),
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
            tokenize_source_bytes(source, token_merge_size, token_merge_alphabet)
            for source in sources
        ]
        self.sources = [source for source in self.sources if len(source) >= seq_length]
        if not self.sources:
            raise ValueError("no train sources are long enough for the configured seq_length")
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.available = [len(source) - seq_length + 1 for source in self.sources]
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
        ids = torch.tensor(window, dtype=torch.long)
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
            tokenize_source_bytes(source, token_merge_size, token_merge_alphabet)
            for source in sources
        ]
        self.sources = [source for source in tokenized_sources if len(source) > 0]
        self.seq_length = seq_length
        self.pad_id = pad_id
        self.index: list[tuple[int, int]] = []

        for source_idx, source in enumerate(self.sources):
            if len(source) <= seq_length:
                self.index.append((source_idx, 0))
                continue
            for start in range(0, len(source), seq_length):
                self.index.append((source_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_idx, start = self.index[index]
        source = self.sources[source_idx]
        chunk = source[start : start + self.seq_length]
        ids = torch.full((self.seq_length,), self.pad_id, dtype=torch.long)
        ids[: len(chunk)] = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": ids}


def build_compression_sample(sources: list[bytes], requested_bytes: int) -> bytes:
    for source in sources:
        if len(source) >= requested_bytes:
            return source[:requested_bytes]
    longest = max(sources, key=len)
    return longest
