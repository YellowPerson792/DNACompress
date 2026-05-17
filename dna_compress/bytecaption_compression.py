from __future__ import annotations

import math
from time import perf_counter
from typing import Callable, Iterable

import torch

from .bytecaption_loader import BYTECAPTION_HIDDEN_STORAGE_DTYPES, BYTECAPTION_LATENT_MODES, ByteCaptionDNACompressor
from .bytecaption_tokenization import ByteCaptionTokenizerSpec, tokenize_bytecaption_source
from .compression import ArithmeticEncoder, baseline_sizes
from .compression_eval import NON_OVERLAP_MODE, sample_payload
from .experiment import autocast_context
from .fixed_token_factorization import FixedTokenArithmeticFactorizer
from .nugget_compression import (
    _encode_fixed_token_units,
    _encode_model_symbol_probabilities,
    _resolve_arithmetic_metadata,
    _tail_side_info_bytes,
)
from .nugget_data import IGNORE_INDEX, build_nugget_window, max_nugget_target_tokens


BYTECAPTION_ARITHMETIC_CODING_MODES = ("model_symbol", "fixed_token_units")
SUPPORTED_BYTECAPTION_COMPRESSION_MODES = (NON_OVERLAP_MODE,)

_TORCH_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def validate_bytecaption_hidden_policy(storage_dtype: str) -> None:
    if storage_dtype not in BYTECAPTION_HIDDEN_STORAGE_DTYPES:
        raise ValueError(
            "model.bytecaption_hidden_storage_dtype must be one of: "
            + ", ".join(BYTECAPTION_HIDDEN_STORAGE_DTYPES)
        )


def _stored_latent_payload(
    *,
    model: ByteCaptionDNACompressor,
    latent_payload: torch.Tensor,
    attention_mask: torch.Tensor,
    storage_dtype_name: str,
) -> dict[str, object]:
    runtime_dtype = latent_payload.dtype
    if storage_dtype_name == "runtime":
        storage_dtype = runtime_dtype
        stored = latent_payload
        storage_name = _dtype_name(runtime_dtype)
        cast_applied = False
    else:
        storage_dtype = _TORCH_DTYPE_BY_NAME[storage_dtype_name]
        stored = latent_payload.to(dtype=storage_dtype)
        storage_name = storage_dtype_name
        cast_applied = storage_dtype != runtime_dtype

    batch_size = int(latent_payload.shape[0])
    valid_count = int(attention_mask.sum().item())
    if model.latent_mode == "dense":
        hidden_bytes = valid_count * int(latent_payload.shape[-1]) * _element_size(storage_dtype)
    elif model.latent_mode == "continuous_bottleneck":
        hidden_bytes = valid_count * int(model.code_dim) * _element_size(storage_dtype)
    elif model.latent_mode == "flatten_bottleneck":
        hidden_bytes = batch_size * int(model.flatten_bottleneck_dim) * _element_size(storage_dtype)
    else:
        raise ValueError(f"Unsupported ByteCaption latent mode: {model.latent_mode}")
    metadata_bytes = batch_size * 4
    return {
        "stored_payload": stored,
        "runtime_dtype": _dtype_name(runtime_dtype),
        "storage_dtype": storage_name,
        "cast_applied": cast_applied,
        "hidden_bytes": hidden_bytes,
        "metadata_bytes": metadata_bytes,
        "valid_count": valid_count,
    }


def compress_bytecaption_source(
    *,
    model: ByteCaptionDNACompressor,
    species: str,
    source: bytes,
    tokenizer_spec: ByteCaptionTokenizerSpec,
    seq_length: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    requested_bytes: int | None,
    mode: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    arithmetic_coding_mode: str,
    arithmetic_merge_size: int,
    hidden_storage_dtype: str,
    species_prefix_map: dict[str, str] | None = None,
    fixed_factorizer: FixedTokenArithmeticFactorizer | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    if mode != NON_OVERLAP_MODE:
        raise ValueError(f"ByteCaption compression currently supports only {NON_OVERLAP_MODE}.")
    if arithmetic_coding_mode not in BYTECAPTION_ARITHMETIC_CODING_MODES:
        raise ValueError(f"Unsupported ByteCaption arithmetic coding mode: {arithmetic_coding_mode}")
    validate_bytecaption_hidden_policy(hidden_storage_dtype)
    payload = sample_payload(source, requested_bytes)
    tokenized = tokenize_bytecaption_source(
        species=species,
        source=payload,
        spec=tokenizer_spec,
        species_prefix_map=species_prefix_map,
        drop_tail_to_full_kmer=False,
    )
    if len(tokenized.dna_token_ids) == 0:
        raise ValueError("ByteCaption compression requires at least one DNA token.")

    arithmetic_metadata = _resolve_arithmetic_metadata(
        tokenizer_spec=tokenizer_spec,
        arithmetic_frequency_total=arithmetic_frequency_total,
        arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
        arithmetic_coding_mode=arithmetic_coding_mode,
        arithmetic_merge_size=arithmetic_merge_size,
        fixed_factorizer=fixed_factorizer,
        prefix_trie=None,
    )

    target_capacity = max_nugget_target_tokens(seq_length, len(tokenized.prefix_ids))
    starts = list(range(0, len(tokenized.dna_token_ids), target_capacity))
    encoder = ArithmeticEncoder()
    decoder_bits = 0.0
    hidden_bytes = 0
    metadata_bytes = 0
    valid_count = 0
    runtime_dtype_name: str | None = None
    storage_dtype_name: str | None = None
    cast_applied = False
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    quantize_seconds_total = 0.0
    aggregate_seconds_total = 0.0
    emitted_count = 0
    total_batches = max(1, math.ceil(len(starts) / batch_size))
    processed_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            windows = [
                build_nugget_window(
                    tokenized,
                    start=start,
                    target_length=min(target_capacity, len(tokenized.dna_token_ids) - start),
                    seq_length=seq_length,
                    pad_id=tokenizer_spec.pad_id,
                )
                for start in batch_starts
            ]
            input_ids = torch.stack([window.input_ids for window in windows], dim=0)
            attention_mask = torch.stack([window.attention_mask for window in windows], dim=0)
            labels = torch.stack([window.labels for window in windows], dim=0)
            decoder_attention_mask = torch.stack([window.decoder_attention_mask for window in windows], dim=0)

            transfer_started = perf_counter()
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            decoder_attention_mask = decoder_attention_mask.to(device, non_blocking=True)
            data_transfer_seconds += perf_counter() - transfer_started

            with autocast_context(device, dtype_name):
                forward_started = perf_counter()
                latent = model.encode_latent(input_ids=input_ids, attention_mask=attention_mask)
                payload_info = _stored_latent_payload(
                    model=model,
                    latent_payload=latent.payload,
                    attention_mask=latent.attention_mask,
                    storage_dtype_name=hidden_storage_dtype,
                )
                stored_payload = payload_info["stored_payload"]
                decoder_dtype = next(model.decoder.parameters()).dtype
                if stored_payload.dtype != decoder_dtype and device.type == "cpu":
                    stored_payload = stored_payload.to(dtype=decoder_dtype)
                output = model.decode_from_latent(
                    stored_payload,
                    latent.attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )
                model_forward_seconds += perf_counter() - forward_started
                softmax_started = perf_counter()
                log_probs = torch.log_softmax(output.logits, dim=-1)
                softmax_seconds += perf_counter() - softmax_started

            hidden_bytes += int(payload_info["hidden_bytes"])
            metadata_bytes += int(payload_info["metadata_bytes"])
            valid_count += int(payload_info["valid_count"])
            runtime_dtype_name = str(payload_info["runtime_dtype"])
            storage_dtype_name = str(payload_info["storage_dtype"])
            cast_applied = cast_applied or bool(payload_info["cast_applied"])

            encode_started = perf_counter()
            valid_mask = labels != IGNORE_INDEX
            flat_log_probs = log_probs[valid_mask]
            flat_targets = labels[valid_mask].to(dtype=torch.long)
            if flat_targets.numel() > 0:
                if arithmetic_coding_mode == "model_symbol":
                    bits, transfer_seconds, count, quantize_seconds = _encode_model_symbol_probabilities(
                        log_prob_rows=flat_log_probs,
                        target_symbols=flat_targets,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                    decoder_bits += bits
                    data_transfer_seconds += transfer_seconds
                    emitted_count += count
                    quantize_seconds_total += quantize_seconds
                else:
                    if fixed_factorizer is None:
                        raise ValueError("fixed_token_units requires fixed_factorizer.")
                    bits, transfer_seconds, aggregate_seconds, count, quantize_seconds = _encode_fixed_token_units(
                        log_prob_rows=flat_log_probs,
                        target_symbols=flat_targets,
                        factorizer=fixed_factorizer,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                    decoder_bits += bits
                    data_transfer_seconds += transfer_seconds
                    aggregate_seconds_total += aggregate_seconds
                    emitted_count += count
                    quantize_seconds_total += quantize_seconds
            arithmetic_encode_seconds += perf_counter() - encode_started
            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    encoded = encoder.finish()
    tail_payload = _tail_side_info_bytes(tokenizer_spec, tokenized.tail_sequence)
    side_info_bytes = hidden_bytes + metadata_bytes + len(tail_payload)
    side_info_bits = side_info_bytes * 8
    total_theoretical_bits = decoder_bits + side_info_bits
    total_coded_bytes = len(encoded) + side_info_bytes
    sample_bases = int(tokenized.total_bases)
    process_seconds = model_forward_seconds + softmax_seconds + data_transfer_seconds + arithmetic_encode_seconds
    return {
        "mode": mode,
        "sample_bytes": len(payload),
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": len(tokenized.dna_token_ids),
        "uses_eos": False,
        "decoder_theoretical_bits": decoder_bits,
        "decoder_theoretical_bits_per_base": decoder_bits / max(sample_bases, 1),
        "latent_side_info_bits_per_base": side_info_bits / max(sample_bases, 1),
        "theoretical_bits": total_theoretical_bits,
        "theoretical_bits_per_base": total_theoretical_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_base": (len(encoded) * 8) / max(sample_bases, 1),
        "bytecaption_latent_mode": model.latent_mode,
        "bytecaption_code_dim": int(model.code_dim),
        "bytecaption_flatten_bottleneck_dim": int(model.flatten_bottleneck_dim),
        "bytecaption_flatten_input_dim": int(model.flatten_input_dim) if model.latent_mode == "flatten_bottleneck" else 0,
        "bytecaption_hidden_bytes": hidden_bytes,
        "bytecaption_metadata_bytes": metadata_bytes,
        "bytecaption_valid_count": valid_count,
        "bytecaption_storage_dtype": storage_dtype_name or hidden_storage_dtype,
        "bytecaption_runtime_dtype": runtime_dtype_name or "unknown",
        "bytecaption_hidden_cast_applied": cast_applied,
        "tail_base_count": len(tokenized.tail_sequence),
        "tail_side_info_bits": len(tail_payload) * 8,
        "side_info_bytes": side_info_bytes,
        "side_info_bits": side_info_bits,
        "total_coded_bytes": total_coded_bytes,
        "total_bits_per_base": (total_coded_bytes * 8) / max(sample_bases, 1),
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": softmax_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "gpu_prefix_aggregate_seconds": aggregate_seconds_total,
        "cpu_small_alphabet_quantize_seconds": quantize_seconds_total,
        "compression_process_seconds": process_seconds,
        "compression_bytes_per_second": len(payload) / max(process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(process_seconds, 1e-12),
        "compression_symbols_per_second": emitted_count / max(process_seconds, 1e-12),
        "emitted_arithmetic_symbol_count": emitted_count,
        "window_stride": target_capacity,
        "window_policy": "bytecaption_encoder_decoder_nonoverlap",
        "cache_reuse": False,
        **arithmetic_metadata,
        **baseline_sizes(payload),
    }


def summarize_bytecaption_per_source(per_source: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(per_source)
    total_sample_bytes = sum(int(row["sample_bytes"]) for row in rows)
    total_sample_bases = sum(int(row["sample_bases"]) for row in rows)
    total_arithmetic_bytes = sum(int(row["arithmetic_coded_bytes"]) for row in rows)
    total_hidden_bytes = sum(int(row["bytecaption_hidden_bytes"]) for row in rows)
    total_metadata_bytes = sum(int(row["bytecaption_metadata_bytes"]) for row in rows)
    total_side_info_bytes = sum(int(row["side_info_bytes"]) for row in rows)
    total_coded_bytes = sum(int(row["total_coded_bytes"]) for row in rows)
    total_decoder_bits = sum(float(row["decoder_theoretical_bits"]) for row in rows)
    total_theoretical_bits = sum(float(row["theoretical_bits"]) for row in rows)
    total_process_seconds = sum(float(row.get("compression_process_seconds", 0.0)) for row in rows)
    summary: dict[str, object] = {
        "source_count": len(rows),
        "total_sample_bytes": total_sample_bytes,
        "total_sample_bases": total_sample_bases,
        "total_decoder_theoretical_bits": total_decoder_bits,
        "total_decoder_theoretical_bits_per_base": total_decoder_bits / max(total_sample_bases, 1),
        "total_theoretical_bits": total_theoretical_bits,
        "total_theoretical_bits_per_base": total_theoretical_bits / max(total_sample_bases, 1),
        "total_arithmetic_coded_bytes": total_arithmetic_bytes,
        "total_arithmetic_bits_per_base": (total_arithmetic_bytes * 8) / max(total_sample_bases, 1),
        "total_bytecaption_hidden_bytes": total_hidden_bytes,
        "total_bytecaption_metadata_bytes": total_metadata_bytes,
        "total_side_info_bytes": total_side_info_bytes,
        "total_coded_bytes": total_coded_bytes,
        "total_bits_per_base": (total_coded_bytes * 8) / max(total_sample_bases, 1),
        "total_compression_process_seconds": total_process_seconds,
        "total_compression_bytes_per_second": total_sample_bytes / max(total_process_seconds, 1e-12),
        "total_compression_bases_per_second": total_sample_bases / max(total_process_seconds, 1e-12),
    }
    for key in (
        "arithmetic_frequency_total",
        "arithmetic_vocab_size",
        "arithmetic_target_uniform_mass",
        "arithmetic_effective_uniform_mass",
        "arithmetic_coding_mode",
        "arithmetic_merge_size",
        "bytecaption_latent_mode",
        "bytecaption_code_dim",
        "bytecaption_flatten_bottleneck_dim",
        "bytecaption_flatten_input_dim",
        "bytecaption_storage_dtype",
    ):
        if rows and all(row.get(key) == rows[0].get(key) for row in rows):
            summary[key] = rows[0].get(key)
    return summary
