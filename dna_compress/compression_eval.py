from __future__ import annotations

import math
from contextlib import nullcontext
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import torch

from .compression import ArithmeticEncoder, baseline_sizes, probabilities_to_cumulative
from .tokenization import tokenize_source_bytes


CompressionMode = str

SLIDING_TOKEN_MODE = "sliding_token"
NON_OVERLAP_MODE = "train_windows_nonoverlap"
OVERLAP_MODE = "train_windows_overlap"
SUPPORTED_COMPRESSION_MODES = (
    SLIDING_TOKEN_MODE,
    NON_OVERLAP_MODE,
    OVERLAP_MODE,
)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def sample_payload(source: bytes, requested_bytes: int | None) -> bytes:
    if requested_bytes is None or requested_bytes <= 0 or len(source) <= requested_bytes:
        return source
    return source[:requested_bytes]


def _symbols_with_optional_eos(
    payload: bytes,
    eos_id: int | None,
    token_merge_size: int,
    token_merge_alphabet: str,
) -> list[int]:
    symbols = tokenize_source_bytes(payload, token_merge_size, token_merge_alphabet)
    if eos_id is not None:
        symbols.append(eos_id)
    return symbols


def _finalize_metrics(
    *,
    payload: bytes,
    symbols: list[int],
    symbol_count_without_eos: int,
    token_merge_size: int,
    total_bits: float,
    encoded: bytes,
    mode: CompressionMode,
    probability_compute_seconds: float,
    arithmetic_encode_seconds: float,
    mode_details: dict[str, object] | None = None,
) -> dict[str, object]:
    sample_bytes = len(payload)
    sample_bases = symbol_count_without_eos * token_merge_size
    compression_process_seconds = probability_compute_seconds + arithmetic_encode_seconds
    return {
        "mode": mode,
        "sample_bytes": sample_bytes,
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": len(symbols),
        "theoretical_bits": total_bits,
        "theoretical_bits_per_base": total_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_base": (len(encoded) * 8) / max(sample_bases, 1),
        "probability_compute_seconds": probability_compute_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "compression_process_seconds": compression_process_seconds,
        "compression_bytes_per_second": sample_bytes / max(compression_process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(compression_process_seconds, 1e-12),
        "compression_symbols_per_second": len(symbols) / max(compression_process_seconds, 1e-12),
        **baseline_sizes(payload),
        **(mode_details or {}),
    }


def compress_sequence_sliding_token(
    *,
    model: torch.nn.Module,
    payload: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    symbols = _symbols_with_optional_eos(payload, eos_id, token_merge_size, token_merge_alphabet)
    total_bits = 0.0
    probability_rows: list[np.ndarray] = []
    targets_for_encoding: list[int] = []
    total_batches = max(1, math.ceil(len(symbols) / batch_size))
    processed_batches = 0

    model.eval()
    compute_started = perf_counter()
    with torch.no_grad():
        for start in range(0, len(symbols), batch_size):
            current = symbols[start : start + batch_size]
            windows = torch.full((len(current), seq_length), pad_id, dtype=torch.long)
            for row_index, symbol_index in enumerate(range(start, start + len(current))):
                history_start = max(0, symbol_index - seq_length + 1)
                history = symbols[history_start : symbol_index + 1]
                windows[row_index, -len(history) :] = torch.tensor(history, dtype=torch.long)

            batch = windows.to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                output = model(batch, return_loss=False)
                log_probs = torch.log_softmax(output.lm_logits[:, -1, :], dim=-1)

            rows_cpu = log_probs.float().cpu()
            for row_cpu, target in zip(rows_cpu, current):
                total_bits += float(-row_cpu[target].item() / math.log(2))
                probability_rows.append(row_cpu.exp().numpy())
                targets_for_encoding.append(target)

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    probability_compute_seconds = perf_counter() - compute_started
    encode_started = perf_counter()
    encoder = ArithmeticEncoder()
    for target, probs in zip(targets_for_encoding, probability_rows):
        encoder.update(probabilities_to_cumulative(probs), target)
    encoded = encoder.finish()
    arithmetic_encode_seconds = perf_counter() - encode_started

    return _finalize_metrics(
        payload=payload,
        symbols=symbols,
        symbol_count_without_eos=len(symbols) - (1 if eos_id is not None else 0),
        token_merge_size=token_merge_size,
        total_bits=total_bits,
        encoded=encoded,
        mode=SLIDING_TOKEN_MODE,
        probability_compute_seconds=probability_compute_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        mode_details={
            "window_stride": 1,
            "window_policy": "right_aligned_sliding_context",
            "cache_reuse": False,
        },
    )


def _window_starts_for_overlap(total_symbols: int, seq_length: int, stride: int) -> list[int]:
    if total_symbols <= 0:
        return [0]
    if total_symbols <= seq_length:
        return [0]

    extra = total_symbols - seq_length
    num_extra_windows = math.ceil(extra / stride)
    return [0] + [stride * index for index in range(1, num_extra_windows + 1)]


def compress_sequence_train_windows(
    *,
    model: torch.nn.Module,
    payload: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    overlap_stride: int | None = None,
    token_merge_size: int,
    token_merge_alphabet: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    symbols = _symbols_with_optional_eos(payload, eos_id, token_merge_size, token_merge_alphabet)
    total_bits = 0.0
    probability_rows: list[np.ndarray] = []
    targets_for_encoding: list[int] = []

    if overlap_stride is None:
        mode = NON_OVERLAP_MODE
        window_starts = list(range(0, len(symbols), seq_length)) or [0]
    else:
        if overlap_stride <= 0 or overlap_stride >= seq_length:
            raise ValueError("overlap_stride must satisfy 0 < overlap_stride < seq_length")
        mode = OVERLAP_MODE
        window_starts = _window_starts_for_overlap(len(symbols), seq_length, overlap_stride)

    total_batches = max(1, math.ceil(len(window_starts) / batch_size))
    processed_batches = 0

    model.eval()
    compute_started = perf_counter()
    with torch.no_grad():
        for batch_start in range(0, len(window_starts), batch_size):
            starts = window_starts[batch_start : batch_start + batch_size]
            windows = torch.full((len(starts), seq_length), pad_id, dtype=torch.long)
            lengths: list[int] = []
            for row_index, start in enumerate(starts):
                chunk = symbols[start : start + seq_length]
                lengths.append(len(chunk))
                if chunk:
                    windows[row_index, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)

            batch = windows.to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                output = model(batch, return_loss=False)
                log_probs = torch.log_softmax(output.lm_logits, dim=-1)
            rows_cpu = log_probs.float().cpu()

            for row_index, (start, chunk_length) in enumerate(zip(starts, lengths)):
                if chunk_length <= 0:
                    continue

                local_start = 0
                if overlap_stride is not None and start > 0:
                    local_start = min(seq_length - overlap_stride, chunk_length)

                for local_pos in range(local_start, chunk_length):
                    target = symbols[start + local_pos]
                    row_cpu = rows_cpu[row_index, local_pos, :]
                    total_bits += float(-row_cpu[target].item() / math.log(2))
                    probability_rows.append(row_cpu.exp().numpy())
                    targets_for_encoding.append(target)

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    probability_compute_seconds = perf_counter() - compute_started
    encode_started = perf_counter()
    encoder = ArithmeticEncoder()
    for target, probs in zip(targets_for_encoding, probability_rows):
        encoder.update(probabilities_to_cumulative(probs), target)
    encoded = encoder.finish()
    arithmetic_encode_seconds = perf_counter() - encode_started

    mode_details: dict[str, object] = {
        "window_policy": "contiguous_train_style",
        "cache_reuse": False,
    }
    if overlap_stride is None:
        mode_details["window_stride"] = seq_length
    else:
        mode_details.update(
            {
                "window_stride": overlap_stride,
                "cache_note": (
                    "This evaluator recomputes each overlap window exactly. Patch-aligned overlap "
                    "makes cache reuse plausible, but hidden-state reuse is not implemented here."
                ),
            }
        )

    return _finalize_metrics(
        payload=payload,
        symbols=symbols,
        symbol_count_without_eos=len(symbols) - (1 if eos_id is not None else 0),
        token_merge_size=token_merge_size,
        total_bits=total_bits,
        encoded=encoded,
        mode=mode,
        probability_compute_seconds=probability_compute_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        mode_details=mode_details,
    )


def compress_source(
    *,
    model: torch.nn.Module,
    source: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    requested_bytes: int | None,
    mode: CompressionMode,
    overlap_stride: int = 1,
    token_merge_size: int = 1,
    token_merge_alphabet: str = "ACGTN",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    payload = sample_payload(source, requested_bytes)
    if mode == SLIDING_TOKEN_MODE:
        return compress_sequence_sliding_token(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            progress_callback=progress_callback,
        )
    if mode == NON_OVERLAP_MODE:
        return compress_sequence_train_windows(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            overlap_stride=None,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            progress_callback=progress_callback,
        )
    if mode == OVERLAP_MODE:
        return compress_sequence_train_windows(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            overlap_stride=overlap_stride,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unsupported compression mode '{mode}'")


def summarize_per_source(
    per_source: Iterable[dict[str, object]],
) -> dict[str, object]:
    rows = list(per_source)
    total_sample_bytes = sum(int(row["sample_bytes"]) for row in rows)
    total_sample_bases = sum(int(row["sample_bases"]) for row in rows)
    total_theoretical_bits = sum(float(row["theoretical_bits"]) for row in rows)
    total_arithmetic_bytes = sum(int(row["arithmetic_coded_bytes"]) for row in rows)
    total_ascii_bytes = sum(int(row["ascii_bytes"]) for row in rows)
    total_two_bit_pack_bytes = sum(int(row["two_bit_pack_bytes"]) for row in rows)
    total_gzip_bytes = sum(int(row["gzip_bytes"]) for row in rows)
    total_bz2_bytes = sum(int(row["bz2_bytes"]) for row in rows)
    total_lzma_bytes = sum(int(row["lzma_bytes"]) for row in rows)
    total_probability_compute_seconds = sum(float(row.get("probability_compute_seconds", 0.0)) for row in rows)
    total_arithmetic_encode_seconds = sum(float(row.get("arithmetic_encode_seconds", 0.0)) for row in rows)
    total_compression_process_seconds = sum(float(row.get("compression_process_seconds", 0.0)) for row in rows)

    return {
        "source_count": len(rows),
        "total_sample_bytes": total_sample_bytes,
        "total_sample_bases": total_sample_bases,
        "total_theoretical_bits": total_theoretical_bits,
        "total_theoretical_bits_per_base": total_theoretical_bits / max(total_sample_bases, 1),
        "total_arithmetic_coded_bytes": total_arithmetic_bytes,
        "total_arithmetic_bits_per_base": (total_arithmetic_bytes * 8) / max(total_sample_bases, 1),
        "total_ascii_bytes": total_ascii_bytes,
        "total_two_bit_pack_bytes": total_two_bit_pack_bytes,
        "total_gzip_bytes": total_gzip_bytes,
        "total_bz2_bytes": total_bz2_bytes,
        "total_lzma_bytes": total_lzma_bytes,
        "total_probability_compute_seconds": total_probability_compute_seconds,
        "total_arithmetic_encode_seconds": total_arithmetic_encode_seconds,
        "total_compression_process_seconds": total_compression_process_seconds,
        "total_compression_bytes_per_second": total_sample_bytes / max(total_compression_process_seconds, 1e-12),
        "total_compression_bases_per_second": total_sample_bases / max(total_compression_process_seconds, 1e-12),
    }
