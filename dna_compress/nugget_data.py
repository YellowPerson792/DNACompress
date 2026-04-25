from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .nugget_tokenization import NuggetTokenizedSource


IGNORE_INDEX = -100


@dataclass(frozen=True)
class NuggetWindow:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    decoder_attention_mask: torch.Tensor
    base_lengths: torch.Tensor


def max_nugget_target_tokens(seq_length: int, prefix_length: int) -> int:
    target_capacity = seq_length - prefix_length
    if target_capacity <= 0:
        raise ValueError(
            f"seq_length={seq_length} is too small for Nugget prefix length {prefix_length}."
        )
    return target_capacity


def build_nugget_window(
    source: NuggetTokenizedSource,
    *,
    start: int,
    target_length: int,
    seq_length: int,
    pad_id: int,
) -> NuggetWindow:
    prefix_length = len(source.prefix_ids)
    target_capacity = max_nugget_target_tokens(seq_length, prefix_length)
    target_length = min(target_length, target_capacity, len(source.dna_token_ids) - start)
    if target_length <= 0:
        raise ValueError("Requested Nugget window does not contain DNA target tokens.")

    input_ids = torch.full((seq_length,), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((seq_length,), dtype=torch.long)
    labels = torch.full((seq_length,), IGNORE_INDEX, dtype=torch.long)
    decoder_attention_mask = torch.zeros((seq_length,), dtype=torch.long)
    base_lengths = torch.zeros((seq_length,), dtype=torch.long)

    token_slice = np.ascontiguousarray(source.dna_token_ids[start : start + target_length])
    base_length_slice = np.ascontiguousarray(source.dna_token_base_lengths[start : start + target_length])
    window_length = prefix_length + target_length
    if prefix_length > 0:
        input_ids[:prefix_length] = torch.tensor(source.prefix_ids, dtype=torch.long)
    input_ids[prefix_length:window_length] = torch.as_tensor(token_slice, dtype=torch.long)
    attention_mask[:window_length] = 1
    decoder_attention_mask[:window_length] = 1
    if prefix_length > 0:
        labels[prefix_length:window_length] = torch.as_tensor(token_slice, dtype=torch.long)
    else:
        labels[:window_length] = torch.as_tensor(token_slice, dtype=torch.long)
    base_lengths[prefix_length:window_length] = torch.as_tensor(base_length_slice, dtype=torch.long)

    return NuggetWindow(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
        base_lengths=base_lengths,
    )


class RandomNuggetWindowDataset(Dataset):
    def __init__(
        self,
        *,
        sources: list[NuggetTokenizedSource],
        seq_length: int,
        samples_per_epoch: int,
        seed: int,
        sampling_strategy: str,
        pad_id: int,
    ) -> None:
        self.sources = [source for source in sources if len(source.dna_token_ids) > 0]
        if not self.sources:
            raise ValueError("No Nugget train sources contain DNA tokens.")
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.pad_id = pad_id
        self.target_capacities = [
            max_nugget_target_tokens(seq_length, len(source.prefix_ids))
            for source in self.sources
        ]
        self.available = [
            max(1, len(source.dna_token_ids) - capacity + 1)
            for source, capacity in zip(self.sources, self.target_capacities)
        ]
        if sampling_strategy == "proportional":
            self.source_weights = [float(count) for count in self.available]
        elif sampling_strategy == "uniform":
            self.source_weights = [1.0 for _ in self.available]
        elif sampling_strategy == "sqrt":
            self.source_weights = [float(count) ** 0.5 for count in self.available]
        else:
            raise ValueError("sampling_strategy must be one of: proportional, uniform, sqrt")
        if sum(self.source_weights) <= 0:
            raise ValueError("Nugget sampling source weights must sum to > 0")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index)
        source_index = rng.choices(range(len(self.sources)), weights=self.source_weights, k=1)[0]
        source = self.sources[source_index]
        capacity = self.target_capacities[source_index]
        max_start_count = self.available[source_index]
        start = 0 if max_start_count == 1 else rng.randrange(max_start_count)
        target_length = min(capacity, len(source.dna_token_ids) - start)
        window = build_nugget_window(
            source,
            start=start,
            target_length=target_length,
            seq_length=self.seq_length,
            pad_id=self.pad_id,
        )
        return {
            "input_ids": window.input_ids,
            "attention_mask": window.attention_mask,
            "labels": window.labels,
            "decoder_attention_mask": window.decoder_attention_mask,
            "base_lengths": window.base_lengths,
        }


class SequentialNuggetWindowDataset(Dataset):
    def __init__(
        self,
        *,
        sources: list[NuggetTokenizedSource],
        seq_length: int,
        pad_id: int,
    ) -> None:
        self.sources = [source for source in sources if len(source.dna_token_ids) > 0]
        self.seq_length = seq_length
        self.pad_id = pad_id
        self.index: list[tuple[int, int, int]] = []
        for source_index, source in enumerate(self.sources):
            capacity = max_nugget_target_tokens(seq_length, len(source.prefix_ids))
            for start in range(0, len(source.dna_token_ids), capacity):
                target_length = min(capacity, len(source.dna_token_ids) - start)
                if target_length > 0:
                    self.index.append((source_index, start, target_length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index, start, target_length = self.index[index]
        window = build_nugget_window(
            self.sources[source_index],
            start=start,
            target_length=target_length,
            seq_length=self.seq_length,
            pad_id=self.pad_id,
        )
        return {
            "input_ids": window.input_ids,
            "attention_mask": window.attention_mask,
            "labels": window.labels,
            "decoder_attention_mask": window.decoder_attention_mask,
            "base_lengths": window.base_lengths,
        }
