from __future__ import annotations

import math
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import torch

from .compression import (
    ArithmeticEncoder,
    baseline_sizes,
    probabilities_to_cumulative_batch,
    resolve_arithmetic_coding_metadata,
)
from .compression_eval import NON_OVERLAP_MODE, sample_payload
from .dnagpt_compression import (
    _encode_base_prefix_probabilities,
    _encode_grouped_prefix_probabilities,
    _tail_side_info_bytes as _dnagpt_tail_side_info_bytes,
)
from .dnagpt_prefix_coding import (
    DNAGPT_PREFIX_ALPHABET_SIZE,
    DNAGPTPrefixTrie,
    grouped_prefix_vocab_size,
)
from .experiment import autocast_context
from .fixed_token_factorization import (
    FixedTokenArithmeticFactorizer,
    factorize_fixed_token_log_probs,
)
from .nugget_data import IGNORE_INDEX, build_nugget_window, max_nugget_target_tokens
from .nugget_loader import NUGGET_HIDDEN_MODES, NUGGET_HIDDEN_STORAGE_DTYPES, NuggetAutoencoder
from .nugget_tokenization import NuggetTokenizerSpec, tokenize_nugget_source


NUGGET_ARITHMETIC_CODING_MODES = (
    "model_symbol",
    "fixed_token_units",
    "base_prefix_exact_gpu_cpu",
)
SUPPORTED_NUGGET_COMPRESSION_MODES = (NON_OVERLAP_MODE,)

_TORCH_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _element_size_for_dtype(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def validate_nugget_hidden_policy(hidden_mode: str, hidden_storage_dtype: str) -> None:
    if hidden_mode not in NUGGET_HIDDEN_MODES:
        raise ValueError(f"model.nugget_hidden_mode must be one of: {', '.join(NUGGET_HIDDEN_MODES)}")
    if hidden_storage_dtype not in NUGGET_HIDDEN_STORAGE_DTYPES:
        raise ValueError(
            "model.nugget_hidden_storage_dtype must be one of: "
            + ", ".join(NUGGET_HIDDEN_STORAGE_DTYPES)
        )
    if hidden_mode == "runtime_hidden" and hidden_storage_dtype != "runtime":
        raise ValueError("runtime_hidden requires nugget_hidden_storage_dtype='runtime'.")
    if hidden_mode == "stored_hidden" and hidden_storage_dtype == "runtime":
        raise ValueError("stored_hidden requires an explicit float32/float16/bfloat16 storage dtype.")


def _resolve_arithmetic_metadata(
    *,
    tokenizer_spec: NuggetTokenizerSpec,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    arithmetic_coding_mode: str,
    arithmetic_merge_size: int,
    fixed_factorizer: FixedTokenArithmeticFactorizer | None,
    prefix_trie: DNAGPTPrefixTrie | None,
) -> dict[str, object]:
    if arithmetic_coding_mode == "model_symbol":
        vocab_size = tokenizer_spec.vocab_size
    elif arithmetic_coding_mode == "fixed_token_units":
        if fixed_factorizer is None:
            raise ValueError("fixed_token_units requires a fixed_kmer tokenizer and factorizer.")
        vocab_size = fixed_factorizer.max_emitted_vocab_size
    elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
        if prefix_trie is None:
            raise ValueError("base_prefix_exact_gpu_cpu requires a dnagpt_kmer tokenizer and prefix trie.")
        vocab_size = (
            DNAGPT_PREFIX_ALPHABET_SIZE
            if arithmetic_merge_size == 1
            else grouped_prefix_vocab_size(
                merge_size=arithmetic_merge_size,
                max_token_length=prefix_trie.max_token_length,
            )
        )
    else:
        raise ValueError(f"Unsupported Nugget arithmetic coding mode '{arithmetic_coding_mode}'.")
    metadata = resolve_arithmetic_coding_metadata(
        vocab_size=vocab_size,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )
    metadata["arithmetic_coding_mode"] = arithmetic_coding_mode
    metadata["arithmetic_merge_size"] = arithmetic_merge_size
    return metadata


def _encode_model_symbol_probabilities(
    *,
    log_prob_rows: torch.Tensor,
    target_symbols: torch.Tensor,
    total: int,
    encoder: ArithmeticEncoder,
) -> tuple[float, float, int, float]:
    target_log_probs = log_prob_rows.gather(1, target_symbols.unsqueeze(1)).squeeze(1)
    bits = float((-target_log_probs / math.log(2)).sum().item())
    transfer_started = perf_counter()
    probabilities = log_prob_rows.float().exp().cpu().numpy()
    targets = target_symbols.cpu().numpy()
    transfer_seconds = perf_counter() - transfer_started
    quantize_started = perf_counter()
    cumulative_batch = probabilities_to_cumulative_batch(probabilities, total=total)
    quantize_seconds = perf_counter() - quantize_started
    for cumulative, target in zip(cumulative_batch, targets):
        encoder.update(cumulative, int(target))
    return bits, transfer_seconds, len(targets), quantize_seconds


def _encode_fixed_token_units(
    *,
    log_prob_rows: torch.Tensor,
    target_symbols: torch.Tensor,
    factorizer: FixedTokenArithmeticFactorizer,
    total: int,
    encoder: ArithmeticEncoder,
) -> tuple[float, float, float, int, float]:
    aggregate_started = perf_counter()
    factorized = factorize_fixed_token_log_probs(log_prob_rows, target_symbols, factorizer)
    aggregate_seconds = perf_counter() - aggregate_started

    transfer_started = perf_counter()
    root_probabilities = factorized.root_probabilities.cpu().numpy()
    root_symbols = factorized.root_symbols.cpu().numpy()
    regular_step_probabilities = tuple(step.cpu().numpy() for step in factorized.regular_step_probabilities)
    regular_step_symbols = tuple(step.cpu().numpy() for step in factorized.regular_step_symbols)
    regular_row_positions = factorized.regular_row_positions.cpu().numpy()
    special_step_probabilities = factorized.special_step_probabilities.cpu().numpy()
    special_step_symbols = factorized.special_step_symbols.cpu().numpy()
    special_row_positions = factorized.special_row_positions.cpu().numpy()
    transfer_seconds = perf_counter() - transfer_started

    quantize_started = perf_counter()
    root_cumulative = probabilities_to_cumulative_batch(root_probabilities, total=total)
    regular_step_cumulative = tuple(
        probabilities_to_cumulative_batch(step, total=total)
        for step in regular_step_probabilities
    )
    if special_step_probabilities.shape[0] > 0:
        special_step_cumulative = probabilities_to_cumulative_batch(special_step_probabilities, total=total)
    else:
        special_step_cumulative = np.zeros((0, 1), dtype=np.int64)
    quantize_seconds = perf_counter() - quantize_started

    for row_index in range(root_symbols.shape[0]):
        encoder.update(root_cumulative[row_index], int(root_symbols[row_index]))
        regular_position = int(regular_row_positions[row_index])
        if regular_position >= 0:
            for cumulative_batch, symbol_batch in zip(regular_step_cumulative, regular_step_symbols):
                encoder.update(cumulative_batch[regular_position], int(symbol_batch[regular_position]))
            continue
        special_position = int(special_row_positions[row_index])
        if special_position >= 0:
            encoder.update(special_step_cumulative[special_position], int(special_step_symbols[special_position]))

    bits = float((-factorized.target_log_probs / math.log(2)).sum().item())
    return bits, transfer_seconds, aggregate_seconds, factorized.emitted_symbol_count, quantize_seconds


def _stored_hidden_payload(
    *,
    encoding: torch.Tensor,
    scores: torch.Tensor | None,
    mask: torch.Tensor,
    hidden_mode: str,
    hidden_storage_dtype: str,
    requires_scores_side_info: bool,
) -> dict[str, object]:
    runtime_dtype = encoding.dtype
    if hidden_mode == "runtime_hidden":
        stored_encoding = encoding
        stored_scores = scores
        storage_dtype = runtime_dtype
        cast_applied = False
        storage_dtype_name = _dtype_name(runtime_dtype)
    else:
        storage_dtype = _TORCH_DTYPE_BY_NAME[hidden_storage_dtype]
        stored_encoding = encoding.to(dtype=storage_dtype)
        stored_scores = scores.to(dtype=storage_dtype) if scores is not None else None
        cast_applied = storage_dtype != runtime_dtype
        storage_dtype_name = hidden_storage_dtype

    valid_count = int(mask.sum().item())
    hidden_dim = int(stored_encoding.shape[-1])
    hidden_bytes = valid_count * hidden_dim * _element_size_for_dtype(storage_dtype)
    score_bytes = 0
    if requires_scores_side_info and stored_scores is not None:
        score_bytes = valid_count * _element_size_for_dtype(storage_dtype)
    metadata_bytes = int(mask.shape[0]) * 4 + score_bytes
    return {
        "stored_encoding": stored_encoding,
        "stored_scores": stored_scores,
        "runtime_dtype": _dtype_name(runtime_dtype),
        "storage_dtype": storage_dtype_name,
        "cast_applied": cast_applied,
        "hidden_bytes": hidden_bytes,
        "score_bytes": score_bytes,
        "metadata_bytes": metadata_bytes,
        "valid_count": valid_count,
    }


def _tail_side_info_bytes(tokenizer_spec: NuggetTokenizerSpec, tail_sequence: str) -> bytes:
    if not tail_sequence:
        return b""
    if tokenizer_spec.name == "dnagpt_kmer":
        return _dnagpt_tail_side_info_bytes(tail_sequence)
    return tail_sequence.encode("ascii")


def compress_nugget_source(
    *,
    model: NuggetAutoencoder,
    species: str,
    source: bytes,
    tokenizer_spec: NuggetTokenizerSpec,
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
    hidden_mode: str,
    hidden_storage_dtype: str,
    species_prefix_map: dict[str, str] | None = None,
    fixed_factorizer: FixedTokenArithmeticFactorizer | None = None,
    prefix_trie: DNAGPTPrefixTrie | None = None,
    requires_scores_side_info: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    if mode != NON_OVERLAP_MODE:
        raise ValueError(f"Nugget compression currently supports only {NON_OVERLAP_MODE}.")
    validate_nugget_hidden_policy(hidden_mode, hidden_storage_dtype)
    payload = sample_payload(source, requested_bytes)
    use_dnagpt_tail_side_info = (
        tokenizer_spec.name == "dnagpt_kmer"
        and arithmetic_coding_mode == "base_prefix_exact_gpu_cpu"
        and arithmetic_merge_size > 1
        and tokenizer_spec.dnagpt_dynamic_kmer
    )
    tokenized = tokenize_nugget_source(
        species=species,
        source=payload,
        spec=tokenizer_spec,
        species_prefix_map=species_prefix_map,
        drop_tail_to_full_kmer=use_dnagpt_tail_side_info,
    )
    if len(tokenized.dna_token_ids) == 0:
        raise ValueError("Nugget compression requires at least one DNA token.")

    arithmetic_metadata = _resolve_arithmetic_metadata(
        tokenizer_spec=tokenizer_spec,
        arithmetic_frequency_total=arithmetic_frequency_total,
        arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
        arithmetic_coding_mode=arithmetic_coding_mode,
        arithmetic_merge_size=arithmetic_merge_size,
        fixed_factorizer=fixed_factorizer,
        prefix_trie=prefix_trie,
    )

    target_capacity = max_nugget_target_tokens(seq_length, len(tokenized.prefix_ids))
    starts = list(range(0, len(tokenized.dna_token_ids), target_capacity))
    encoder = ArithmeticEncoder()
    decoder_theoretical_bits = 0.0
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    gpu_prefix_aggregate_seconds = 0.0
    cpu_small_alphabet_quantize_seconds = 0.0
    emitted_arithmetic_symbol_count = 0
    nugget_hidden_bytes = 0
    nugget_metadata_bytes = 0
    nugget_score_bytes = 0
    nugget_valid_count = 0
    runtime_dtype_name: str | None = None
    storage_dtype_name: str | None = None
    cast_applied = False
    processed_batches = 0
    total_batches = max(1, math.ceil(len(starts) / batch_size))

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
                nuggets = model.encode_nuggets(input_ids=input_ids, attention_mask=attention_mask)
                payload_info = _stored_hidden_payload(
                    encoding=nuggets.encoding,
                    scores=nuggets.scores,
                    mask=nuggets.mask,
                    hidden_mode=hidden_mode,
                    hidden_storage_dtype=hidden_storage_dtype,
                    requires_scores_side_info=requires_scores_side_info,
                )
                decoder_encoding = payload_info["stored_encoding"]
                decoder_scores = payload_info["stored_scores"] if requires_scores_side_info else nuggets.scores
                model_dtype = next(model.decoder.parameters()).dtype
                if decoder_encoding.dtype != model_dtype and device.type == "cpu":
                    decoder_encoding = decoder_encoding.to(dtype=model_dtype)
                    if decoder_scores is not None:
                        decoder_scores = decoder_scores.to(dtype=model_dtype)
                output = model.decode_from_nuggets(
                    nugget_encoding=decoder_encoding,
                    nugget_mask=nuggets.mask,
                    nugget_scores=decoder_scores,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )
                model_forward_seconds += perf_counter() - forward_started

                softmax_started = perf_counter()
                log_probs = torch.log_softmax(output.logits, dim=-1)
                softmax_seconds += perf_counter() - softmax_started

            nugget_hidden_bytes += int(payload_info["hidden_bytes"])
            nugget_metadata_bytes += int(payload_info["metadata_bytes"])
            nugget_score_bytes += int(payload_info["score_bytes"])
            nugget_valid_count += int(payload_info["valid_count"])
            runtime_dtype_name = str(payload_info["runtime_dtype"])
            storage_dtype_name = str(payload_info["storage_dtype"])
            cast_applied = cast_applied or bool(payload_info["cast_applied"])

            encode_started = perf_counter()
            valid_mask = labels != IGNORE_INDEX
            flat_log_probs = log_probs[valid_mask]
            flat_targets = labels[valid_mask].to(dtype=torch.long)
            if flat_targets.numel() > 0:
                if arithmetic_coding_mode == "model_symbol":
                    (
                        bits,
                        transfer_seconds,
                        emitted_count,
                        quantize_seconds,
                    ) = _encode_model_symbol_probabilities(
                        log_prob_rows=flat_log_probs,
                        target_symbols=flat_targets,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                    decoder_theoretical_bits += bits
                    data_transfer_seconds += transfer_seconds
                    emitted_arithmetic_symbol_count += emitted_count
                    cpu_small_alphabet_quantize_seconds += quantize_seconds
                elif arithmetic_coding_mode == "fixed_token_units":
                    if fixed_factorizer is None:
                        raise ValueError("fixed_token_units requires fixed_factorizer.")
                    (
                        bits,
                        transfer_seconds,
                        aggregate_seconds,
                        emitted_count,
                        quantize_seconds,
                    ) = _encode_fixed_token_units(
                        log_prob_rows=flat_log_probs,
                        target_symbols=flat_targets,
                        factorizer=fixed_factorizer,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                    decoder_theoretical_bits += bits
                    data_transfer_seconds += transfer_seconds
                    gpu_prefix_aggregate_seconds += aggregate_seconds
                    emitted_arithmetic_symbol_count += emitted_count
                    cpu_small_alphabet_quantize_seconds += quantize_seconds
                elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
                    if prefix_trie is None:
                        raise ValueError("base_prefix_exact_gpu_cpu requires prefix_trie.")
                    if arithmetic_merge_size == 1:
                        (
                            bits,
                            transfer_seconds,
                            aggregate_seconds,
                            emitted_count,
                            quantize_seconds,
                        ) = _encode_base_prefix_probabilities(
                            log_prob_rows=flat_log_probs,
                            target_token_ids=flat_targets,
                            trie=prefix_trie,
                            total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                            encoder=encoder,
                        )
                    else:
                        (
                            bits,
                            transfer_seconds,
                            aggregate_seconds,
                            emitted_count,
                            quantize_seconds,
                        ) = _encode_grouped_prefix_probabilities(
                            log_prob_rows=flat_log_probs,
                            target_token_ids=flat_targets,
                            trie=prefix_trie,
                            merge_size=arithmetic_merge_size,
                            total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                            encoder=encoder,
                        )
                    decoder_theoretical_bits += bits
                    data_transfer_seconds += transfer_seconds
                    gpu_prefix_aggregate_seconds += aggregate_seconds
                    emitted_arithmetic_symbol_count += emitted_count
                    cpu_small_alphabet_quantize_seconds += quantize_seconds
                else:
                    raise ValueError(f"Unsupported Nugget arithmetic coding mode '{arithmetic_coding_mode}'.")
            arithmetic_encode_seconds += perf_counter() - encode_started

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    encoded = encoder.finish()
    tail_payload = _tail_side_info_bytes(tokenizer_spec, tokenized.tail_sequence)
    tail_side_info_bits = len(tail_payload) * 8
    side_info_bytes = nugget_hidden_bytes + nugget_metadata_bytes + len(tail_payload)
    total_coded_bytes = len(encoded) + side_info_bytes
    side_info_bits = side_info_bytes * 8
    total_theoretical_bits = decoder_theoretical_bits + side_info_bits
    sample_bases = tokenized.total_bases
    process_seconds = model_forward_seconds + softmax_seconds + data_transfer_seconds + arithmetic_encode_seconds

    return {
        "mode": mode,
        "sample_bytes": len(payload),
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": len(tokenized.dna_token_ids),
        "uses_eos": False,
        "prefix_token_count": len(tokenized.prefix_ids),
        "decoder_theoretical_bits": decoder_theoretical_bits,
        "decoder_theoretical_bits_per_base": decoder_theoretical_bits / max(sample_bases, 1),
        "theoretical_bits": total_theoretical_bits,
        "theoretical_bits_per_base": total_theoretical_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_base": (len(encoded) * 8) / max(sample_bases, 1),
        "nugget_hidden_bytes": nugget_hidden_bytes,
        "nugget_metadata_bytes": nugget_metadata_bytes,
        "nugget_score_bytes": nugget_score_bytes,
        "nugget_valid_count": nugget_valid_count,
        "nugget_hidden_runtime_dtype": runtime_dtype_name or "unknown",
        "nugget_hidden_storage_dtype": storage_dtype_name or hidden_storage_dtype,
        "nugget_hidden_mode": hidden_mode,
        "nugget_hidden_cast_applied": cast_applied,
        "tail_base_count": len(tokenized.tail_sequence),
        "tail_side_info_bits": tail_side_info_bits,
        "side_info_bytes": side_info_bytes,
        "side_info_bits": side_info_bits,
        "total_coded_bytes": total_coded_bytes,
        "total_bits_per_base": (total_coded_bytes * 8) / max(sample_bases, 1),
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": softmax_seconds,
        "model_forward_softmax_seconds": model_forward_seconds + softmax_seconds,
        "probability_compute_seconds": model_forward_seconds + softmax_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "gpu_prefix_aggregate_seconds": gpu_prefix_aggregate_seconds,
        "cpu_small_alphabet_quantize_seconds": cpu_small_alphabet_quantize_seconds,
        "compression_process_seconds": process_seconds,
        "compression_bytes_per_second": len(payload) / max(process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(process_seconds, 1e-12),
        "compression_symbols_per_second": emitted_arithmetic_symbol_count / max(process_seconds, 1e-12),
        "emitted_arithmetic_symbol_count": emitted_arithmetic_symbol_count,
        "window_stride": target_capacity,
        "window_policy": "nugget_encoder_decoder_nonoverlap",
        "cache_reuse": False,
        **arithmetic_metadata,
        **baseline_sizes(payload),
    }


def summarize_nugget_per_source(per_source: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(per_source)
    total_sample_bytes = sum(int(row["sample_bytes"]) for row in rows)
    total_sample_bases = sum(int(row["sample_bases"]) for row in rows)
    total_arithmetic_bytes = sum(int(row["arithmetic_coded_bytes"]) for row in rows)
    total_hidden_bytes = sum(int(row["nugget_hidden_bytes"]) for row in rows)
    total_metadata_bytes = sum(int(row["nugget_metadata_bytes"]) for row in rows)
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
        "total_nugget_hidden_bytes": total_hidden_bytes,
        "total_nugget_metadata_bytes": total_metadata_bytes,
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
        "nugget_hidden_mode",
        "nugget_hidden_storage_dtype",
    ):
        if rows and all(row.get(key) == rows[0].get(key) for row in rows):
            summary[key] = rows[0].get(key)
    return summary
