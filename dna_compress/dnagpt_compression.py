from __future__ import annotations

import math
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .compression import ArithmeticEncoder, baseline_sizes, probabilities_to_cumulative_batch
from .compression_eval import NON_OVERLAP_MODE, SLIDING_TOKEN_MODE
from .dnagpt_data import max_target_tokens
from .dnagpt_tokenization import TokenizedDNASource, tokenize_dna_source
from .experiment import autocast_context


SUPPORTED_DNAGPT_COMPRESSION_MODES = (
    SLIDING_TOKEN_MODE,
    NON_OVERLAP_MODE,
)


def sample_payload(source: bytes, requested_bytes: int | None) -> bytes:
    if requested_bytes is None or requested_bytes <= 0 or len(source) <= requested_bytes:
        return source
    return source[:requested_bytes]


def _finalize_metrics(
    *,
    payload: bytes,
    tokenized_source: TokenizedDNASource,
    total_bits: float,
    encoded: bytes,
    mode: str,
    probability_compute_seconds: float,
    arithmetic_encode_seconds: float,
    mode_details: dict[str, object] | None = None,
) -> dict[str, object]:
    sample_bytes = len(payload)
    sample_bases = tokenized_source.total_bases
    compression_process_seconds = probability_compute_seconds + arithmetic_encode_seconds
    return {
        "mode": mode,
        "sample_bytes": sample_bytes,
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": len(tokenized_source.dna_token_ids),
        "uses_eos": False,
        "prefix_token_count": len(tokenized_source.prefix_ids),
        "theoretical_bits": total_bits,
        "theoretical_bits_per_base": total_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_base": (len(encoded) * 8) / max(sample_bases, 1),
        "probability_compute_seconds": probability_compute_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "compression_process_seconds": compression_process_seconds,
        "compression_bytes_per_second": sample_bytes / max(compression_process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(compression_process_seconds, 1e-12),
        "compression_symbols_per_second": len(tokenized_source.dna_token_ids) / max(compression_process_seconds, 1e-12),
        **baseline_sizes(payload),
        **(mode_details or {}),
    }


def compress_dnagpt_sequence_sliding(
    *,
    model: torch.nn.Module,
    payload: bytes,
    tokenized_source: TokenizedDNASource,
    seq_length: int,
    pad_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    dna_tokens = tokenized_source.dna_token_ids
    if not dna_tokens:
        raise ValueError("DNAGPT compression requires at least one DNA token.")

    encoder = ArithmeticEncoder()
    total_bits = 0.0
    probability_compute_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    total_batches = max(1, math.ceil(len(dna_tokens) / batch_size))
    processed_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(dna_tokens), batch_size):
            batch_targets = dna_tokens[batch_start : batch_start + batch_size]
            contexts: list[list[int]] = []
            used_lengths: list[int] = []

            for offset, _ in enumerate(batch_targets):
                target_index = batch_start + offset
                context_tokens = [pad_id] + tokenized_source.prefix_ids + dna_tokens[:target_index]
                if len(context_tokens) > seq_length:
                    context_tokens = context_tokens[-seq_length:]
                used_lengths.append(len(context_tokens))
                contexts.append(context_tokens)

            batch_input = torch.full((len(batch_targets), seq_length), pad_id, dtype=torch.long)
            for row_index, context_tokens in enumerate(contexts):
                batch_input[row_index, : len(context_tokens)] = torch.tensor(context_tokens, dtype=torch.long)

            prob_started = perf_counter()
            batch_input = batch_input.to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                logits = model(batch_input)
            gather_index = torch.tensor([length - 1 for length in used_lengths], device=device, dtype=torch.long)
            row_index = torch.arange(len(batch_targets), device=device)
            next_token_logits = logits[row_index, gather_index, :]
            probability_compute_seconds += perf_counter() - prob_started

            targets_device = torch.tensor(batch_targets, dtype=torch.long, device=device)
            target_log_probs = torch.log_softmax(next_token_logits, dim=-1).gather(1, targets_device[:, None]).squeeze(1)
            total_bits += float((-target_log_probs / math.log(2)).sum().item())

            encode_started = perf_counter()
            probs_np = torch.softmax(next_token_logits.float(), dim=-1).cpu().numpy()
            cumulative_batch = probabilities_to_cumulative_batch(probs_np)
            for cumulative, target in zip(cumulative_batch, batch_targets):
                encoder.update(cumulative, int(target))
            arithmetic_encode_seconds += perf_counter() - encode_started

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    encoded = encoder.finish()
    return _finalize_metrics(
        payload=payload,
        tokenized_source=tokenized_source,
        total_bits=total_bits,
        encoded=encoded,
        mode=SLIDING_TOKEN_MODE,
        probability_compute_seconds=probability_compute_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        mode_details={
            "window_stride": 1,
            "window_policy": "left_context_sliding",
            "cache_reuse": False,
        },
    )


def compress_dnagpt_sequence_train_windows(
    *,
    model: torch.nn.Module,
    payload: bytes,
    tokenized_source: TokenizedDNASource,
    seq_length: int,
    pad_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    dna_tokens = tokenized_source.dna_token_ids
    if not dna_tokens:
        raise ValueError("DNAGPT compression requires at least one DNA token.")

    prefix_length = len(tokenized_source.prefix_ids)
    target_capacity = max_target_tokens(seq_length, prefix_length)
    starts = list(range(0, len(dna_tokens), target_capacity))
    encoder = ArithmeticEncoder()
    total_bits = 0.0
    probability_compute_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    total_batches = max(1, math.ceil(len(starts) / batch_size))
    processed_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            batch_input = torch.full((len(batch_starts), seq_length), pad_id, dtype=torch.long)
            chunk_lengths: list[int] = []
            chunk_targets: list[list[int]] = []

            for row_index, start in enumerate(batch_starts):
                chunk = dna_tokens[start : start + target_capacity]
                chunk_lengths.append(len(chunk))
                chunk_targets.append(chunk)
                for prefix_index, prefix_id in enumerate(tokenized_source.prefix_ids):
                    batch_input[row_index, prefix_index + 1] = int(prefix_id)
                for offset, token_id in enumerate(chunk[:-1]):
                    batch_input[row_index, prefix_length + offset + 1] = int(token_id)

            prob_started = perf_counter()
            batch_input = batch_input.to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                logits = model(batch_input)
            probability_compute_seconds += perf_counter() - prob_started

            encode_started = perf_counter()
            for row_index, (chunk, chunk_length) in enumerate(zip(chunk_targets, chunk_lengths)):
                if chunk_length <= 0:
                    continue
                row_logits = logits[row_index, prefix_length : prefix_length + chunk_length, :]
                targets_device = torch.tensor(chunk, dtype=torch.long, device=device)
                target_log_probs = torch.log_softmax(row_logits, dim=-1).gather(1, targets_device[:, None]).squeeze(1)
                total_bits += float((-target_log_probs / math.log(2)).sum().item())
                probs_np = torch.softmax(row_logits.float(), dim=-1).cpu().numpy()
                cumulative_batch = probabilities_to_cumulative_batch(probs_np)
                for cumulative, target in zip(cumulative_batch, chunk):
                    encoder.update(cumulative, int(target))
            arithmetic_encode_seconds += perf_counter() - encode_started

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    encoded = encoder.finish()
    return _finalize_metrics(
        payload=payload,
        tokenized_source=tokenized_source,
        total_bits=total_bits,
        encoded=encoded,
        mode=NON_OVERLAP_MODE,
        probability_compute_seconds=probability_compute_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        mode_details={
            "window_stride": target_capacity,
            "window_policy": "contiguous_train_style",
            "cache_reuse": False,
        },
    )


def compress_dnagpt_source(
    *,
    model: torch.nn.Module,
    species: str,
    source: bytes,
    tokenizer,
    kmer_size: int,
    species_prefix_map: dict[str, str] | None,
    seq_length: int,
    pad_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    requested_bytes: int | None,
    mode: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    payload = sample_payload(source, requested_bytes)
    tokenized_source = tokenize_dna_source(
        species=species,
        source=payload,
        tokenizer=tokenizer,
        kmer_size=kmer_size,
        species_prefix_map=species_prefix_map,
    )
    if mode == SLIDING_TOKEN_MODE:
        metrics = compress_dnagpt_sequence_sliding(
            model=model,
            payload=payload,
            tokenized_source=tokenized_source,
            seq_length=seq_length,
            pad_id=pad_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
    elif mode == NON_OVERLAP_MODE:
        metrics = compress_dnagpt_sequence_train_windows(
            model=model,
            payload=payload,
            tokenized_source=tokenized_source,
            seq_length=seq_length,
            pad_id=pad_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unsupported DNAGPT compression mode '{mode}'.")

    metrics["sample_bytes"] = len(payload)
    metrics["sample_bases"] = len(payload)
    return metrics
