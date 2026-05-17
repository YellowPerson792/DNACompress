from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse
import torch

from dna_compress.gencoder import (
    GenCoderAutoEncoder,
    decode_sequence,
    deserialize_csr,
    encode_sequence,
    fpzip_compress_array,
    fpzip_decompress_array,
    pad_and_chunk,
    quantize_reconstruction,
    serialize_csr,
)


class GenCoderTests(unittest.TestCase):
    def test_acgt_integer_roundtrip(self) -> None:
        sequence = b"ACGTACGT"
        encoded = encode_sequence(sequence)
        self.assertEqual(encoded.tolist(), [1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(decode_sequence(encoded), sequence)

    def test_padding_chunking_and_trimmed_decode(self) -> None:
        encoded = encode_sequence(b"ACGTA")
        chunks, padding = pad_and_chunk(encoded, seq_length=4)
        self.assertEqual(padding, 3)
        self.assertEqual(chunks.shape, (2, 4))
        self.assertEqual(decode_sequence(chunks, original_length=5), b"ACGTA")

    def test_residual_makes_quantized_autoencoder_lossless(self) -> None:
        original = np.asarray([[1, 2, 3, 4]], dtype=np.uint8)
        reconstructed_float = np.asarray([[0.24, 0.51, 0.76, 0.99]], dtype=np.float32)
        reconstructed_int = quantize_reconstruction(reconstructed_float)
        residual = original.astype(np.int16) - reconstructed_int
        recovered = reconstructed_int + residual
        self.assertTrue(np.array_equal(recovered, original))
        self.assertEqual(decode_sequence(recovered), b"ACGT")

    def test_csr_serialization_roundtrip(self) -> None:
        matrix = sparse.csr_matrix(np.asarray([[0, 1, 0], [-2, 0, 3]], dtype=np.int16))
        payload = serialize_csr(matrix)
        restored = deserialize_csr(payload)
        self.assertGreater(len(payload), 0)
        self.assertTrue(np.array_equal(restored.toarray(), matrix.toarray()))

    def test_fpzip_latent_roundtrip(self) -> None:
        latent = np.arange(24, dtype=np.float32).reshape(3, 8) / 7.0
        payload = fpzip_compress_array(latent)
        restored = fpzip_decompress_array(payload, latent.shape)
        self.assertEqual(restored.shape, latent.shape)
        self.assertTrue(np.array_equal(restored, latent))

    def test_model_shapes(self) -> None:
        model = GenCoderAutoEncoder(seq_length=16, bottleneck_dim=4)
        batch = torch.full((2, 16), 0.5)
        latent = model.encode(batch)
        reconstructed = model.decode(latent)
        self.assertEqual(tuple(latent.shape), (2, 4))
        self.assertEqual(tuple(reconstructed.shape), (2, 16))


if __name__ == "__main__":
    unittest.main()
