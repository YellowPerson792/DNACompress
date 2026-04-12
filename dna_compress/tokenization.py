from __future__ import annotations

import numpy as np
from typing import Iterable


DEFAULT_DNA_ALPHABET = "ACGTN"


def normalize_alphabet(alphabet: str) -> str:
    cleaned = "".join(ch for ch in alphabet.upper() if not ch.isspace())
    unique_chars: list[str] = []
    seen: set[str] = set()
    for ch in cleaned:
        if ch in seen:
            continue
        seen.add(ch)
        unique_chars.append(ch)
    if len(unique_chars) < 2:
        raise ValueError("token_merge_alphabet must contain at least 2 unique characters")
    return "".join(unique_chars)


def uses_token_merge(token_merge_size: int) -> bool:
    return token_merge_size > 1


def resolve_vocab_and_special_ids(token_merge_size: int, token_merge_alphabet: str) -> tuple[int, int, int] | None:
    if not uses_token_merge(token_merge_size):
        return None
    alphabet = normalize_alphabet(token_merge_alphabet)
    vocab_without_special = len(alphabet) ** token_merge_size
    vocab_size = vocab_without_special + 2
    pad_id = vocab_size - 2
    eos_id = vocab_size - 1
    return vocab_size, pad_id, eos_id


def tokenize_source_bytes(source: bytes, token_merge_size: int, token_merge_alphabet: str) -> list[int]:
    if token_merge_size <= 1:
        return list(source)

    alphabet = normalize_alphabet(token_merge_alphabet)
    alpha_to_digit: dict[int, int] = {}
    for index, ch in enumerate(alphabet):
        alpha_to_digit[ord(ch)] = index
        alpha_to_digit[ord(ch.lower())] = index

    digits: list[int] = []
    for byte_value in source:
        if byte_value in alpha_to_digit:
            digits.append(alpha_to_digit[byte_value])

    if len(digits) < token_merge_size:
        return []

    base = len(alphabet)
    tokens: list[int] = []
    for start in range(0, len(digits) - token_merge_size + 1, token_merge_size):
        token_id = 0
        for digit in digits[start : start + token_merge_size]:
            token_id = token_id * base + digit
        tokens.append(token_id)
    return tokens


def _resolve_compact_token_dtype(max_value: int):
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def tokenize_source_array(source: bytes, token_merge_size: int, token_merge_alphabet: str) -> np.ndarray:
    if token_merge_size <= 1:
        return np.frombuffer(source, dtype=np.uint8).copy()

    alphabet = normalize_alphabet(token_merge_alphabet)
    alpha_to_digit: dict[int, int] = {}
    for index, ch in enumerate(alphabet):
        alpha_to_digit[ord(ch)] = index
        alpha_to_digit[ord(ch.lower())] = index

    max_regular_token_id = len(alphabet) ** token_merge_size - 1
    dtype = _resolve_compact_token_dtype(max_regular_token_id)
    if len(source) < token_merge_size:
        return np.empty((0,), dtype=dtype)

    lookup = np.full(256, -1, dtype=np.int16)
    for byte_value, digit in alpha_to_digit.items():
        lookup[byte_value] = digit

    base = len(alphabet)
    weights = np.array(
        [base ** power for power in range(token_merge_size - 1, -1, -1)],
        dtype=np.uint64,
    )
    raw = np.frombuffer(source, dtype=np.uint8)
    chunk_size = max(1 << 20, token_merge_size * 8192)
    carry = np.empty((0,), dtype=np.int16)
    token_chunks: list[np.ndarray] = []

    for start in range(0, raw.shape[0], chunk_size):
        chunk = raw[start : start + chunk_size]
        digits = lookup[chunk]
        if np.any(digits < 0):
            digits = digits[digits >= 0]
        if carry.size > 0:
            digits = np.concatenate((carry, digits))
        full_digit_count = (digits.shape[0] // token_merge_size) * token_merge_size
        if full_digit_count == 0:
            carry = digits
            continue

        carry = digits[full_digit_count:]
        merged = digits[:full_digit_count].reshape(-1, token_merge_size).astype(np.uint64, copy=False)
        token_chunk = (merged * weights).sum(axis=1, dtype=np.uint64).astype(dtype, copy=False)
        token_chunks.append(token_chunk)

    if not token_chunks:
        return np.empty((0,), dtype=dtype)
    if len(token_chunks) == 1:
        return token_chunks[0]
    return np.concatenate(token_chunks)


def apply_token_merge_to_model_config(model_config, data_config) -> None:
    resolved = resolve_vocab_and_special_ids(data_config.token_merge_size, data_config.token_merge_alphabet)
    if resolved is None:
        return
    vocab_size, pad_id, eos_id = resolved
    model_config.vocab_size = vocab_size
    model_config.pad_id = pad_id
    model_config.eos_id = eos_id
