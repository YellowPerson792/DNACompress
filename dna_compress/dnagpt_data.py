from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.utils.data import Dataset

from .dnagpt_tokenization import TokenizedDNASource


IGNORE_INDEX = -100


def max_target_tokens(seq_length: int, prefix_length: int) -> int:
    target_capacity = seq_length - prefix_length
    if target_capacity <= 0:
        raise ValueError(
            f"seq_length={seq_length} is too small for DNAGPT prefix length {prefix_length}. "
            "Increase model.seq_length or remove the configured prefix."
        )
    return target_capacity


@dataclass(frozen=True)
class DNAGPTWindow:
    input_ids: torch.Tensor
    labels: torch.Tensor
    base_lengths: torch.Tensor


def build_dnagpt_window(
    source: TokenizedDNASource,
    *,
    start: int,
    target_length: int,
    seq_length: int,
    pad_id: int,
) -> DNAGPTWindow:
    if target_length <= 0:
        raise ValueError("target_length must be > 0")

    prefix_length = len(source.prefix_ids)
    target_capacity = max_target_tokens(seq_length, prefix_length)
    target_length = min(target_length, target_capacity, len(source.dna_token_ids) - start)
    if target_length <= 0:
        raise ValueError("Requested DNAGPT window does not contain any target tokens.")

    input_ids = torch.full((seq_length,), pad_id, dtype=torch.long)
    labels = torch.full((seq_length,), IGNORE_INDEX, dtype=torch.long)
    base_lengths = torch.zeros((seq_length,), dtype=torch.long)

    for prefix_index, prefix_id in enumerate(source.prefix_ids):
        labels[prefix_index] = int(prefix_id)
        input_ids[prefix_index + 1] = int(prefix_id)

    for offset in range(target_length):
        label_index = prefix_length + offset
        token_index = start + offset
        labels[label_index] = int(source.dna_token_ids[token_index])
        base_lengths[label_index] = int(source.dna_token_base_lengths[token_index])
        if offset + 1 < target_length:
            input_ids[prefix_length + offset + 1] = int(source.dna_token_ids[token_index])

    if prefix_length > 0:
        labels[:prefix_length] = IGNORE_INDEX

    return DNAGPTWindow(
        input_ids=input_ids,
        labels=labels,
        base_lengths=base_lengths,
    )


class RandomDNAGPTWindowDataset(Dataset):
    def __init__(
        self,
        *,
        sources: list[TokenizedDNASource],
        seq_length: int,
        samples_per_epoch: int,
        seed: int,
        sampling_strategy: str = "proportional",
        pad_id: int,
    ) -> None:
        self.sources = [source for source in sources if len(source.dna_token_ids) > 0]
        if not self.sources:
            raise ValueError("No DNAGPT train sources contain any DNA tokens.")
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.pad_id = pad_id
        self.target_capacities = [
            max_target_tokens(seq_length, len(source.prefix_ids))
            for source in self.sources
        ]
        self.available = [
            max(1, len(source.dna_token_ids) - target_capacity + 1)
            for source, target_capacity in zip(self.sources, self.target_capacities)
        ]
        self.sampling_strategy = sampling_strategy
        if sampling_strategy == "proportional":
            self.source_weights = [float(count) for count in self.available]
        elif sampling_strategy == "uniform":
            self.source_weights = [1.0 for _ in self.available]
        elif sampling_strategy == "sqrt":
            self.source_weights = [float(count) ** 0.5 for count in self.available]
        else:
            raise ValueError("sampling_strategy must be one of: proportional, uniform, sqrt")

        self.total_weight = sum(self.source_weights)
        if self.total_weight <= 0:
            raise ValueError("DNAGPT sampling weights must sum to > 0")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index)
        source_index = rng.choices(range(len(self.sources)), weights=self.source_weights, k=1)[0]
        source = self.sources[source_index]
        target_capacity = self.target_capacities[source_index]
        max_start_count = self.available[source_index]
        start = 0 if max_start_count == 1 else rng.randrange(max_start_count)
        target_length = min(target_capacity, len(source.dna_token_ids) - start)
        window = build_dnagpt_window(
            source,
            start=start,
            target_length=target_length,
            seq_length=self.seq_length,
            pad_id=self.pad_id,
        )
        return {
            "input_ids": window.input_ids,
            "labels": window.labels,
            "base_lengths": window.base_lengths,
        }


class SequentialDNAGPTWindowDataset(Dataset):
    def __init__(
        self,
        *,
        sources: list[TokenizedDNASource],
        seq_length: int,
        pad_id: int,
    ) -> None:
        self.sources = [source for source in sources if len(source.dna_token_ids) > 0]
        self.seq_length = seq_length
        self.pad_id = pad_id
        self.index: list[tuple[int, int, int]] = []

        for source_index, source in enumerate(self.sources):
            target_capacity = max_target_tokens(seq_length, len(source.prefix_ids))
            for start in range(0, len(source.dna_token_ids), target_capacity):
                target_length = min(target_capacity, len(source.dna_token_ids) - start)
                self.index.append((source_index, start, target_length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index, start, target_length = self.index[index]
        window = build_dnagpt_window(
            self.sources[source_index],
            start=start,
            target_length=target_length,
            seq_length=self.seq_length,
            pad_id=self.pad_id,
        )
        return {
            "input_ids": window.input_ids,
            "labels": window.labels,
            "base_lengths": window.base_lengths,
        }
