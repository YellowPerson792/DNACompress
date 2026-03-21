from __future__ import annotations

import bz2
import gzip
import io
import lzma
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


def probabilities_to_cumulative(probabilities: np.ndarray, total: int = 1 << 15) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / probs.sum()
    n = probs.shape[0]
    if total <= n:
        raise ValueError("frequency total must exceed vocabulary size")

    scaled = probs * (total - n)
    freq = np.floor(scaled).astype(np.int64) + 1
    remainder = total - int(freq.sum())
    fractional = scaled - np.floor(scaled)

    if remainder > 0:
        order = np.argsort(-fractional)
        freq[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(-(freq - 1))
        debt = -remainder
        for index in order:
            removable = int(freq[index] - 1)
            if removable <= 0:
                continue
            take = min(removable, debt)
            freq[index] -= take
            debt -= take
            if debt == 0:
                break
        if debt != 0:
            raise ValueError("failed to normalize arithmetic coding frequencies")

    cumulative = np.zeros(freq.shape[0] + 1, dtype=np.int64)
    cumulative[1:] = np.cumsum(freq, dtype=np.int64)
    return cumulative


def arithmetic_encode(symbols: Iterable[int], probability_rows: Iterable[np.ndarray]) -> bytes:
    encoder = ArithmeticEncoder()
    for symbol, probs in zip(symbols, probability_rows):
        encoder.update(probabilities_to_cumulative(probs), symbol)
    return encoder.finish()


def baseline_sizes(payload: bytes) -> dict[str, int]:
    return {
        "ascii_bytes": len(payload),
        "two_bit_pack_bytes": (len(payload) * 2 + 7) // 8,
        "gzip_bytes": len(gzip.compress(payload)),
        "bz2_bytes": len(bz2.compress(payload)),
        "lzma_bytes": len(lzma.compress(payload)),
    }
