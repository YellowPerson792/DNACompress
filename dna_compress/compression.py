from __future__ import annotations

import bz2
import gzip
import io
import lzma
import math
from typing import Iterable

import numpy as np


class BitOutputStream:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.current_byte = 0
        self.num_bits_filled = 0

    def write(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | bit
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.num_bits_filled = 0

    def finish(self) -> bytes:
        if self.num_bits_filled > 0:
            self.current_byte <<= 8 - self.num_bits_filled
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.num_bits_filled = 0
        return bytes(self.buffer)


class ArithmeticEncoder:
    STATE_BITS = 32
    FULL_RANGE = 1 << STATE_BITS
    HALF_RANGE = FULL_RANGE >> 1
    QUARTER_RANGE = HALF_RANGE >> 1
    MASK = FULL_RANGE - 1

    def __init__(self) -> None:
        self.low = 0
        self.high = self.MASK
        self.pending_underflow = 0
        self.bit_output = BitOutputStream()

    def update(self, cumulative: np.ndarray, symbol: int) -> None:
        total = int(cumulative[-1])
        symbol_low = int(cumulative[symbol])
        symbol_high = int(cumulative[symbol + 1])
        current_range = self.high - self.low + 1

        self.high = self.low + (current_range * symbol_high // total) - 1
        self.low = self.low + (current_range * symbol_low // total)

        while ((self.low ^ self.high) & self.HALF_RANGE) == 0:
            bit = self.low >> (self.STATE_BITS - 1)
            self._shift(bit)
            self.low = ((self.low << 1) & self.MASK)
            self.high = ((self.high << 1) & self.MASK) | 1

        while (self.low & ~self.high & self.QUARTER_RANGE) != 0:
            self.pending_underflow += 1
            self.low = (self.low << 1) & (self.MASK >> 1)
            self.high = ((self.high ^ self.HALF_RANGE) << 1) | self.HALF_RANGE | 1

    def _shift(self, bit: int) -> None:
        self.bit_output.write(bit)
        for _ in range(self.pending_underflow):
            self.bit_output.write(bit ^ 1)
        self.pending_underflow = 0

    def finish(self) -> bytes:
        self.pending_underflow += 1
        if self.low < self.QUARTER_RANGE:
            self._shift(0)
        else:
            self._shift(1)
        return self.bit_output.finish()


MIN_FREQUENCY_TOTAL = 1 << 15


def max_supported_frequency_total(state_bits: int = ArithmeticEncoder.STATE_BITS) -> int:
    if state_bits < 2:
        raise ValueError("state_bits must be >= 2")
    return (1 << (state_bits - 2)) + 2


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def resolve_frequency_total(
    vocab_size: int,
    requested_total: int | None,
    target_uniform_mass: float,
    state_bits: int = ArithmeticEncoder.STATE_BITS,
) -> int:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if not (0.0 < target_uniform_mass <= 1.0):
        raise ValueError("target_uniform_mass must be in (0.0, 1.0]")

    if requested_total is None:
        minimum_required = max(
            MIN_FREQUENCY_TOTAL,
            vocab_size + 1,
            math.ceil(vocab_size / target_uniform_mass),
        )
        total = _next_power_of_two(minimum_required)
    else:
        total = int(requested_total)

    if total <= vocab_size:
        raise ValueError(
            f"frequency total ({total}) must exceed vocabulary size ({vocab_size})"
        )

    max_total = max_supported_frequency_total(state_bits)
    if total > max_total:
        raise ValueError(
            f"frequency total ({total}) exceeds 32-bit arithmetic coding limit ({max_total})"
        )
    return total


def resolve_arithmetic_coding_metadata(
    vocab_size: int,
    requested_total: int | None,
    target_uniform_mass: float,
    state_bits: int = ArithmeticEncoder.STATE_BITS,
) -> dict[str, float | int]:
    frequency_total = resolve_frequency_total(
        vocab_size=vocab_size,
        requested_total=requested_total,
        target_uniform_mass=target_uniform_mass,
        state_bits=state_bits,
    )
    return {
        "arithmetic_frequency_total": frequency_total,
        "arithmetic_vocab_size": int(vocab_size),
        "arithmetic_target_uniform_mass": float(target_uniform_mass),
        "arithmetic_effective_uniform_mass": float(vocab_size) / float(frequency_total),
    }


def probabilities_to_cumulative(probabilities: np.ndarray, total: int = 1 << 15) -> np.ndarray:
    return probabilities_to_cumulative_batch(probabilities, total=total)


def probabilities_to_cumulative_batch(probabilities: np.ndarray, total: int = 1 << 15) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    is_single = probs.ndim == 1
    if is_single:
        probs = probs[None, :]

    if probs.ndim != 2:
        raise ValueError("probabilities must be a 1D or 2D array")

    n = probs.shape[1]
    if total <= n:
        raise ValueError("frequency total must exceed vocabulary size")

    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    probs = probs / row_sums

    scaled = probs * (total - n)
    floor_scaled = np.floor(scaled)
    freq = floor_scaled.astype(np.int64) + 1
    remainder = total - freq.sum(axis=1)
    fractional = scaled - floor_scaled

    for row_index, row_remainder in enumerate(remainder.tolist()):
        if row_remainder > 0:
            order = np.argsort(-fractional[row_index])
            freq[row_index, order[:row_remainder]] += 1
            continue

        if row_remainder < 0:
            order = np.argsort(-(freq[row_index] - 1))
            debt = -row_remainder
            for index in order:
                removable = int(freq[row_index, index] - 1)
                if removable <= 0:
                    continue
                take = min(removable, debt)
                freq[row_index, index] -= take
                debt -= take
                if debt == 0:
                    break
            if debt != 0:
                raise ValueError("failed to normalize arithmetic coding frequencies")

    cumulative = np.zeros((freq.shape[0], freq.shape[1] + 1), dtype=np.int64)
    cumulative[:, 1:] = np.cumsum(freq, axis=1, dtype=np.int64)
    if is_single:
        return cumulative[0]
    return cumulative


def arithmetic_encode(
    symbols: Iterable[int],
    probability_rows: Iterable[np.ndarray],
    batch_size: int = 512,
    total: int = MIN_FREQUENCY_TOTAL,
) -> bytes:
    encoder = ArithmeticEncoder()

    symbol_chunk: list[int] = []
    probs_chunk: list[np.ndarray] = []

    def flush_chunk() -> None:
        if not probs_chunk:
            return
        cumulative_batch = probabilities_to_cumulative_batch(np.stack(probs_chunk, axis=0), total=total)
        for symbol_value, cumulative in zip(symbol_chunk, cumulative_batch):
            encoder.update(cumulative, int(symbol_value))
        symbol_chunk.clear()
        probs_chunk.clear()

    for symbol, probs in zip(symbols, probability_rows):
        symbol_chunk.append(int(symbol))
        probs_chunk.append(np.asarray(probs, dtype=np.float64))
        if len(probs_chunk) >= batch_size:
            flush_chunk()

    flush_chunk()
    return encoder.finish()


def baseline_sizes(payload: bytes) -> dict[str, int]:
    return {
        "ascii_bytes": len(payload),
        "two_bit_pack_bytes": (len(payload) * 2 + 7) // 8,
        "gzip_bytes": len(gzip.compress(payload)),
        "bz2_bytes": len(bz2.compress(payload)),
        "lzma_bytes": len(lzma.compress(payload)),
    }
