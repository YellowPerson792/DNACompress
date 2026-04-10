from __future__ import annotations

import math
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .compression import (
    ArithmeticEncoder,
    BitOutputStream,
    baseline_sizes,
    probabilities_to_cumulative_batch,
    resolve_arithmetic_coding_metadata,
)
from .compression_eval import NON_OVERLAP_MODE, SLIDING_TOKEN_MODE
from .dnagpt_data import max_target_tokens
from .dnagpt_prefix_coding import (
    DNAGPT_PREFIX_ALPHABET_SIZE,
    DNAGPT_PREFIX_BASE_ORDER,
    DNAGPTPrefixTrie,
    factorize_dnagpt_log_probs_to_base_prefix_stream,
    factorize_dnagpt_log_probs_to_grouped_prefix_stream,
    grouped_prefix_vocab_size,
)
from .dnagpt_tokenization import TokenizedDNASource, tokenize_dna_source
from .experiment import autocast_context


SUPPORTED_DNAGPT_COMPRESSION_MODES = (
    SLIDING_TOKEN_MODE,
    NON_OVERLAP_MODE,
)
DNAGPT_ARITHMETIC_CODING_MODES = (
    "model_symbol",
    "base_prefix_exact_gpu_cpu",
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
    model_forward_seconds: float,
    softmax_seconds: float,
    data_transfer_seconds: float,
    arithmetic_encode_seconds: float,
    gpu_prefix_aggregate_seconds: float,
    cpu_small_alphabet_quantize_seconds: float,
    arithmetic_metadata: dict[str, object],
    arithmetic_coding_mode: str,
    arithmetic_merge_size: int,
    emitted_arithmetic_symbol_count: int,
    core_model_theoretical_bits: float,
    tail_base_count: int,
    tail_side_info_bits: int,
    mode_details: dict[str, object] | None = None,
) -> dict[str, object]:
    sample_bytes = len(payload)
    sample_bases = tokenized_source.total_bases
    model_forward_softmax_seconds = model_forward_seconds + softmax_seconds
    compression_process_seconds = (
        model_forward_seconds + softmax_seconds + data_transfer_seconds + arithmetic_encode_seconds
    )
    return {
        "mode": mode,
        "sample_bytes": sample_bytes,
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": len(tokenized_source.dna_token_ids),
        "uses_eos": False,
        "prefix_token_count": len(tokenized_source.prefix_ids),
        "theoretical_bits": total_bits,
        "theoretical_bits_per_base": total_bits / max(sample_bases, 1),
        "core_model_theoretical_bits": core_model_theoretical_bits,
        "tail_base_count": tail_base_count,
        "tail_side_info_bits": tail_side_info_bits,
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_base": (len(encoded) * 8) / max(sample_bases, 1),
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": softmax_seconds,
        "model_forward_softmax_seconds": model_forward_softmax_seconds,
        "probability_compute_seconds": model_forward_softmax_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "gpu_prefix_aggregate_seconds": gpu_prefix_aggregate_seconds,
        "cpu_small_alphabet_quantize_seconds": cpu_small_alphabet_quantize_seconds,
        "compression_process_seconds": compression_process_seconds,
        "compression_bytes_per_second": sample_bytes / max(compression_process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(compression_process_seconds, 1e-12),
        "compression_symbols_per_second": emitted_arithmetic_symbol_count / max(compression_process_seconds, 1e-12),
        "arithmetic_coding_mode": arithmetic_coding_mode,
        "arithmetic_merge_size": arithmetic_merge_size,
        "emitted_arithmetic_symbol_count": emitted_arithmetic_symbol_count,
        **arithmetic_metadata,
        **baseline_sizes(payload),
        **(mode_details or {}),
    }


def _encode_model_symbol_probabilities(
    *,
    probability_rows: np.ndarray,
    target_symbols: np.ndarray,
    total: int,
    encoder: ArithmeticEncoder,
) -> tuple[float, int]:
    quantize_started = perf_counter()
    cumulative_batch = probabilities_to_cumulative_batch(probability_rows, total=total)
    quantize_seconds = perf_counter() - quantize_started
    for cumulative, target in zip(cumulative_batch, target_symbols):
        encoder.update(cumulative, int(target))
    return quantize_seconds, len(target_symbols)


def _encode_base_prefix_probabilities(
    *,
    log_prob_rows: torch.Tensor,
    target_token_ids: torch.Tensor,
    trie: DNAGPTPrefixTrie,
    total: int,
    encoder: ArithmeticEncoder,
) -> tuple[float, float, float, int, float]:
    aggregate_started = perf_counter()
    factorized = factorize_dnagpt_log_probs_to_base_prefix_stream(
        log_probs=log_prob_rows,
        target_token_ids=target_token_ids,
        trie=trie,
    )
    gpu_prefix_aggregate_seconds = perf_counter() - aggregate_started

    transfer_started = perf_counter()
    flat_probabilities = factorized.emitted_probabilities[factorized.emitted_valid_mask].cpu().numpy()
    flat_symbols = factorized.emitted_symbols[factorized.emitted_valid_mask].cpu().numpy()
    data_transfer_seconds = perf_counter() - transfer_started

    quantize_started = perf_counter()
    cumulative_batch = probabilities_to_cumulative_batch(flat_probabilities, total=total)
    cpu_small_alphabet_quantize_seconds = perf_counter() - quantize_started
    for cumulative, symbol in zip(cumulative_batch, flat_symbols):
        encoder.update(cumulative, int(symbol))
    total_bits = float((-factorized.target_log_probs / math.log(2)).sum().item())
    return (
        total_bits,
        data_transfer_seconds,
        gpu_prefix_aggregate_seconds,
        factorized.emitted_symbol_count,
        cpu_small_alphabet_quantize_seconds,
    )


def _encode_grouped_prefix_probabilities(
    *,
    log_prob_rows: torch.Tensor,
    target_token_ids: torch.Tensor,
    trie: DNAGPTPrefixTrie,
    merge_size: int,
    total: int,
    encoder: ArithmeticEncoder,
) -> tuple[float, float, float, int, float]:
    aggregate_started = perf_counter()
    factorized = factorize_dnagpt_log_probs_to_grouped_prefix_stream(
        log_probs=log_prob_rows,
        target_token_ids=target_token_ids,
        trie=trie,
        merge_size=merge_size,
    )
    gpu_prefix_aggregate_seconds = perf_counter() - aggregate_started

    transfer_started = perf_counter()
    step_probabilities = tuple(step.cpu().numpy() for step in factorized.step_probabilities)
    step_symbols = tuple(step.cpu().numpy() for step in factorized.step_symbols)
    step_row_positions = tuple(step.cpu().numpy() for step in factorized.step_row_positions)
    data_transfer_seconds = perf_counter() - transfer_started

    quantize_started = perf_counter()
    step_cumulative = tuple(
        probabilities_to_cumulative_batch(step, total=total)
        for step in step_probabilities
    )
    cpu_small_alphabet_quantize_seconds = perf_counter() - quantize_started

    for row_index in range(target_token_ids.shape[0]):
        for cumulative_batch, symbol_batch, row_positions in zip(
            step_cumulative,
            step_symbols,
            step_row_positions,
        ):
            position = int(row_positions[row_index])
            if position < 0:
                continue
            encoder.update(cumulative_batch[position], int(symbol_batch[position]))

    total_bits = float((-factorized.target_log_probs / math.log(2)).sum().item())
    return (
        total_bits,
        data_transfer_seconds,
        gpu_prefix_aggregate_seconds,
        factorized.emitted_symbol_count,
        cpu_small_alphabet_quantize_seconds,
    )


def _tail_side_info_bytes(tail_sequence: str) -> bytes:
    if not tail_sequence:
        stream = BitOutputStream()
        for shift in range(2, -1, -1):
            stream.write((0 >> shift) & 1)
        return stream.finish()

    base_to_value = {base: index for index, base in enumerate(DNAGPT_PREFIX_BASE_ORDER)}
    stream = BitOutputStream()
    tail_length = len(tail_sequence)
    for shift in range(2, -1, -1):
        stream.write((tail_length >> shift) & 1)
    for base in tail_sequence:
        try:
            value = base_to_value[base]
        except KeyError as error:
            raise ValueError(f"Unexpected tail base '{base}' outside {DNAGPT_PREFIX_BASE_ORDER}.") from error
        for shift in range(2, -1, -1):
            stream.write((value >> shift) & 1)
    return stream.finish()


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
    arithmetic_metadata: dict[str, object],
    arithmetic_coding_mode: str,
    arithmetic_merge_size: int,
    prefix_trie: DNAGPTPrefixTrie | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    dna_tokens = tokenized_source.dna_token_ids
    if not dna_tokens:
        raise ValueError("DNAGPT compression requires at least one DNA token.")

    encoder = ArithmeticEncoder()
    total_bits = 0.0
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    gpu_prefix_aggregate_seconds = 0.0
    cpu_small_alphabet_quantize_seconds = 0.0
    emitted_arithmetic_symbol_count = 0
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

            transfer_started = perf_counter()
            batch_input = batch_input.to(device, non_blocking=True)
            data_transfer_seconds += perf_counter() - transfer_started

            with autocast_context(device, dtype_name):
                forward_started = perf_counter()
                logits = model(batch_input)
                model_forward_seconds += perf_counter() - forward_started

                softmax_started = perf_counter()
                gather_index = torch.tensor([length - 1 for length in used_lengths], device=device, dtype=torch.long)
                row_index = torch.arange(len(batch_targets), device=device)
                next_token_logits = logits[row_index, gather_index, :]
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                softmax_seconds += perf_counter() - softmax_started

            encode_started = perf_counter()
            if arithmetic_coding_mode == "model_symbol":
                targets_device = torch.tensor(batch_targets, dtype=torch.long, device=device)
                target_log_probs = next_token_log_probs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
                total_bits += float((-target_log_probs / math.log(2)).sum().item())
                transfer_started = perf_counter()
                probs_np = next_token_log_probs.float().exp().cpu().numpy()
                data_transfer_seconds += perf_counter() - transfer_started
                quantize_seconds, emitted_count = _encode_model_symbol_probabilities(
                    probability_rows=probs_np,
                    target_symbols=np.asarray(batch_targets, dtype=np.int64),
                    total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                    encoder=encoder,
                )
                cpu_small_alphabet_quantize_seconds += quantize_seconds
                emitted_arithmetic_symbol_count += emitted_count
            elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
                if prefix_trie is None:
                    raise ValueError("prefix_trie is required for DNAGPT base-prefix arithmetic coding.")
                if arithmetic_merge_size == 1:
                    (
                        batch_bits,
                        batch_transfer_seconds,
                        batch_gpu_aggregate_seconds,
                        emitted_count,
                        batch_quantize_seconds,
                    ) = _encode_base_prefix_probabilities(
                        log_prob_rows=next_token_log_probs,
                        target_token_ids=torch.tensor(batch_targets, dtype=torch.long, device=device),
                        trie=prefix_trie,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                else:
                    (
                        batch_bits,
                        batch_transfer_seconds,
                        batch_gpu_aggregate_seconds,
                        emitted_count,
                        batch_quantize_seconds,
                    ) = _encode_grouped_prefix_probabilities(
                        log_prob_rows=next_token_log_probs,
                        target_token_ids=torch.tensor(batch_targets, dtype=torch.long, device=device),
                        trie=prefix_trie,
                        merge_size=arithmetic_merge_size,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                total_bits += batch_bits
                data_transfer_seconds += batch_transfer_seconds
                gpu_prefix_aggregate_seconds += batch_gpu_aggregate_seconds
                cpu_small_alphabet_quantize_seconds += batch_quantize_seconds
                emitted_arithmetic_symbol_count += emitted_count
            else:
                raise ValueError(f"Unsupported DNAGPT arithmetic coding mode '{arithmetic_coding_mode}'.")
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
        model_forward_seconds=model_forward_seconds,
        softmax_seconds=softmax_seconds,
        data_transfer_seconds=data_transfer_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        gpu_prefix_aggregate_seconds=gpu_prefix_aggregate_seconds,
        cpu_small_alphabet_quantize_seconds=cpu_small_alphabet_quantize_seconds,
        arithmetic_metadata=arithmetic_metadata,
        arithmetic_coding_mode=arithmetic_coding_mode,
        arithmetic_merge_size=arithmetic_merge_size,
        emitted_arithmetic_symbol_count=emitted_arithmetic_symbol_count,
        core_model_theoretical_bits=total_bits,
        tail_base_count=len(tokenized_source.tail_sequence),
        tail_side_info_bits=0,
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
    arithmetic_metadata: dict[str, object],
    arithmetic_coding_mode: str,
    arithmetic_merge_size: int,
    prefix_trie: DNAGPTPrefixTrie | None,
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
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    gpu_prefix_aggregate_seconds = 0.0
    cpu_small_alphabet_quantize_seconds = 0.0
    emitted_arithmetic_symbol_count = 0
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

            transfer_started = perf_counter()
            batch_input = batch_input.to(device, non_blocking=True)
            data_transfer_seconds += perf_counter() - transfer_started

            with autocast_context(device, dtype_name):
                forward_started = perf_counter()
                logits = model(batch_input)
                model_forward_seconds += perf_counter() - forward_started

                softmax_started = perf_counter()
                log_probs = torch.log_softmax(logits, dim=-1)
                softmax_seconds += perf_counter() - softmax_started

            encode_started = perf_counter()
            if arithmetic_coding_mode == "model_symbol":
                transfer_started = perf_counter()
                rows_cpu = log_probs.float().cpu().numpy()
                probs_cpu = log_probs.float().exp().cpu().numpy()
                data_transfer_seconds += perf_counter() - transfer_started

                for row_index, (chunk, chunk_length) in enumerate(zip(chunk_targets, chunk_lengths)):
                    if chunk_length <= 0:
                        continue
                    row_log_probs = rows_cpu[row_index, prefix_length : prefix_length + chunk_length, :]
                    if row_log_probs.shape[0] == 0:
                        continue
                    targets_np = np.asarray(chunk, dtype=np.int64)
                    total_bits += float((-row_log_probs[np.arange(row_log_probs.shape[0]), targets_np] / math.log(2)).sum())
                    probs_np = probs_cpu[row_index, prefix_length : prefix_length + chunk_length, :]
                    quantize_seconds, emitted_count = _encode_model_symbol_probabilities(
                        probability_rows=probs_np,
                        target_symbols=targets_np,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                    cpu_small_alphabet_quantize_seconds += quantize_seconds
                    emitted_arithmetic_symbol_count += emitted_count
            elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
                if prefix_trie is None:
                    raise ValueError("prefix_trie is required for DNAGPT base-prefix arithmetic coding.")
                flat_log_prob_rows: list[torch.Tensor] = []
                flat_target_ids: list[torch.Tensor] = []
                for row_index, (chunk, chunk_length) in enumerate(zip(chunk_targets, chunk_lengths)):
                    if chunk_length <= 0:
                        continue
                    flat_log_prob_rows.append(log_probs[row_index, prefix_length : prefix_length + chunk_length, :])
                    flat_target_ids.append(torch.tensor(chunk, dtype=torch.long, device=device))
                if flat_log_prob_rows:
                    if arithmetic_merge_size == 1:
                        (
                            batch_bits,
                            batch_transfer_seconds,
                            batch_gpu_aggregate_seconds,
                            emitted_count,
                            batch_quantize_seconds,
                        ) = _encode_base_prefix_probabilities(
                            log_prob_rows=torch.cat(flat_log_prob_rows, dim=0),
                            target_token_ids=torch.cat(flat_target_ids, dim=0),
                            trie=prefix_trie,
                            total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                            encoder=encoder,
                        )
                    else:
                        (
                            batch_bits,
                            batch_transfer_seconds,
                            batch_gpu_aggregate_seconds,
                            emitted_count,
                            batch_quantize_seconds,
                        ) = _encode_grouped_prefix_probabilities(
                            log_prob_rows=torch.cat(flat_log_prob_rows, dim=0),
                            target_token_ids=torch.cat(flat_target_ids, dim=0),
                            trie=prefix_trie,
                            merge_size=arithmetic_merge_size,
                            total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                            encoder=encoder,
                        )
                    total_bits += batch_bits
                    data_transfer_seconds += batch_transfer_seconds
                    gpu_prefix_aggregate_seconds += batch_gpu_aggregate_seconds
                    cpu_small_alphabet_quantize_seconds += batch_quantize_seconds
                    emitted_arithmetic_symbol_count += emitted_count
            else:
                raise ValueError(f"Unsupported DNAGPT arithmetic coding mode '{arithmetic_coding_mode}'.")
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
        model_forward_seconds=model_forward_seconds,
        softmax_seconds=softmax_seconds,
        data_transfer_seconds=data_transfer_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        gpu_prefix_aggregate_seconds=gpu_prefix_aggregate_seconds,
        cpu_small_alphabet_quantize_seconds=cpu_small_alphabet_quantize_seconds,
        arithmetic_metadata=arithmetic_metadata,
        arithmetic_coding_mode=arithmetic_coding_mode,
        arithmetic_merge_size=arithmetic_merge_size,
        emitted_arithmetic_symbol_count=emitted_arithmetic_symbol_count,
        core_model_theoretical_bits=total_bits,
        tail_base_count=len(tokenized_source.tail_sequence),
        tail_side_info_bits=0,
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
    dynamic_kmer: bool,
    species_prefix_map: dict[str, str] | None,
    seq_length: int,
    pad_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    requested_bytes: int | None,
    mode: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    arithmetic_coding_mode: str,
    arithmetic_merge_size: int,
    prefix_trie: DNAGPTPrefixTrie | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    payload = sample_payload(source, requested_bytes)
    use_tail_side_info = (
        arithmetic_coding_mode == "base_prefix_exact_gpu_cpu"
        and arithmetic_merge_size > 1
        and dynamic_kmer
    )
    tokenized_source = tokenize_dna_source(
        species=species,
        source=payload,
        tokenizer=tokenizer,
        kmer_size=kmer_size,
        species_prefix_map=species_prefix_map,
        drop_tail_to_full_kmer=use_tail_side_info,
    )
    arithmetic_vocab_size = (
        len(tokenizer)
        if arithmetic_coding_mode == "model_symbol"
        else (
            DNAGPT_PREFIX_ALPHABET_SIZE
            if arithmetic_merge_size == 1
            else grouped_prefix_vocab_size(
                merge_size=arithmetic_merge_size,
                max_token_length=kmer_size,
            )
        )
    )
    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=arithmetic_vocab_size,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
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
            arithmetic_metadata=arithmetic_metadata,
            arithmetic_coding_mode=arithmetic_coding_mode,
            arithmetic_merge_size=arithmetic_merge_size,
            prefix_trie=prefix_trie,
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
            arithmetic_metadata=arithmetic_metadata,
            arithmetic_coding_mode=arithmetic_coding_mode,
            arithmetic_merge_size=arithmetic_merge_size,
            prefix_trie=prefix_trie,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unsupported DNAGPT compression mode '{mode}'.")

    tail_side_info = _tail_side_info_bytes(tokenized_source.tail_sequence) if use_tail_side_info else b""
    tail_side_info_bits = (3 + 3 * len(tokenized_source.tail_sequence)) if use_tail_side_info else 0
    metrics["sample_bytes"] = len(payload)
    metrics["sample_bases"] = len(payload)
    metrics["core_model_theoretical_bits"] = float(metrics.get("core_model_theoretical_bits", metrics["theoretical_bits"]))
    metrics["tail_base_count"] = len(tokenized_source.tail_sequence)
    metrics["tail_side_info_bits"] = tail_side_info_bits
    metrics["theoretical_bits"] = float(metrics["core_model_theoretical_bits"]) + float(tail_side_info_bits)
    metrics["theoretical_bits_per_base"] = float(metrics["theoretical_bits"]) / max(int(metrics["sample_bases"]), 1)
    metrics["arithmetic_coded_bytes"] = int(metrics["arithmetic_coded_bytes"]) + len(tail_side_info)
    metrics["arithmetic_bits_per_base"] = (int(metrics["arithmetic_coded_bytes"]) * 8) / max(int(metrics["sample_bases"]), 1)
    return metrics
