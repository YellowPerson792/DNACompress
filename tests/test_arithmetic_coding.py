from __future__ import annotations

import unittest

import numpy as np

from dna_compress.compression import (
    MIN_FREQUENCY_TOTAL,
    probabilities_to_cumulative_batch,
    resolve_frequency_total,
)


class ArithmeticCodingTests(unittest.TestCase):
    def test_resolve_frequency_total_for_dnagpt_vocab(self) -> None:
        self.assertEqual(resolve_frequency_total(19564, None, 0.01), 2097152)

    def test_resolve_frequency_total_keeps_small_vocab_floor(self) -> None:
        self.assertEqual(resolve_frequency_total(259, None, 0.01), MIN_FREQUENCY_TOTAL)

    def test_probabilities_to_cumulative_batch_large_vocab(self) -> None:
        vocab_size = 19564
        frequency_total = 2097152
        probabilities = np.full((1, vocab_size), 1.0 / vocab_size, dtype=np.float64)

        cumulative = probabilities_to_cumulative_batch(probabilities, total=frequency_total)

        self.assertEqual(cumulative.shape, (1, vocab_size + 1))
        self.assertEqual(int(cumulative[0, 0]), 0)
        self.assertEqual(int(cumulative[0, -1]), frequency_total)
        self.assertTrue(np.all(np.diff(cumulative[0]) > 0))

    def test_larger_frequency_total_reduces_quantization_penalty(self) -> None:
        vocab_size = 19564
        logits = np.linspace(0.0, -12.0, vocab_size, dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()

        entropy_bits = float(-(probabilities * np.log2(probabilities)).sum())
        low_total = probabilities_to_cumulative_batch(probabilities, total=1 << 15)
        high_total = probabilities_to_cumulative_batch(probabilities, total=1 << 21)

        low_quantized = np.diff(low_total).astype(np.float64)
        low_quantized /= low_quantized.sum()
        high_quantized = np.diff(high_total).astype(np.float64)
        high_quantized /= high_quantized.sum()

        low_penalty = float(-(probabilities * np.log2(low_quantized)).sum()) - entropy_bits
        high_penalty = float(-(probabilities * np.log2(high_quantized)).sum()) - entropy_bits

        self.assertGreater(low_penalty, 0.1)
        self.assertLess(high_penalty, low_penalty * 0.2)


if __name__ == "__main__":
    unittest.main()
