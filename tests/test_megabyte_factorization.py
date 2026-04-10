from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest

import torch

from dna_compress.compression_eval import NON_OVERLAP_MODE, compress_source
from dna_compress.config import ExperimentConfig
from dna_compress.fixed_token_factorization import (
    build_fixed_token_arithmetic_factorizer,
    factorize_fixed_token_log_probs,
)
from scripts.run_dna_compression import _validate_args


class _FakeMegabyteModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, batch: torch.Tensor, return_loss: bool = False):
        del return_loss
        batch_size, seq_length = batch.shape
        base_logits = torch.linspace(2.0, -2.0, self.vocab_size, device=batch.device, dtype=torch.float32)
        logits = base_logits.view(1, 1, -1).repeat(batch_size, seq_length, 1)
        position_bias = torch.arange(seq_length, device=batch.device, dtype=torch.float32).view(1, seq_length, 1) * 0.05
        token_bias = (batch.to(dtype=torch.float32).unsqueeze(-1) % 7.0) * 0.01
        return SimpleNamespace(lm_logits=logits + position_bias + token_bias)


class MegabyteFactorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.alphabet = "ACGTN"
        self.model_merge_size = 3
        self.regular_vocab_size = len(self.alphabet) ** self.model_merge_size
        self.pad_id = self.regular_vocab_size
        self.eos_id = self.regular_vocab_size + 1

    def test_factorization_matches_target_probability_for_supported_merge_sizes(self) -> None:
        target_ids = torch.tensor([0, 57, 124, self.eos_id], dtype=torch.long)
        logits = torch.randn((target_ids.shape[0], self.regular_vocab_size + 2), dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)

        for arithmetic_merge_size in (1, 2, 3):
            factorizer = build_fixed_token_arithmetic_factorizer(
                vocab_size=self.regular_vocab_size + 2,
                special_token_ids=[self.pad_id, self.eos_id],
                model_merge_size=self.model_merge_size,
                arithmetic_merge_size=arithmetic_merge_size,
                alphabet=self.alphabet,
            )
            factorized = factorize_fixed_token_log_probs(log_probs, target_ids, factorizer)

            for row_index in range(target_ids.shape[0]):
                emitted_log_prob = torch.log(
                    factorized.root_probabilities[row_index, factorized.root_symbols[row_index]]
                )
                regular_position = int(factorized.regular_row_positions[row_index].item())
                if regular_position >= 0:
                    for probabilities, symbols in zip(
                        factorized.regular_step_probabilities,
                        factorized.regular_step_symbols,
                    ):
                        emitted_log_prob = emitted_log_prob + torch.log(probabilities[regular_position, symbols[regular_position]])
                else:
                    special_position = int(factorized.special_row_positions[row_index].item())
                    emitted_log_prob = emitted_log_prob + torch.log(
                        factorized.special_step_probabilities[special_position, factorized.special_step_symbols[special_position]]
                    )

                self.assertAlmostEqual(
                    emitted_log_prob.item(),
                    factorized.target_log_probs[row_index].item(),
                    places=5,
                )

    def test_validate_args_rejects_invalid_factorized_merge_size(self) -> None:
        config = ExperimentConfig()
        config.model.implementation = "megabyte"
        config.model.seq_length = 8
        config.model.patch_size = 4
        config.model.input_causal_conv_kernel_size = 1
        config.data.token_merge_size = 3
        config.data.token_merge_alphabet = self.alphabet
        config.arithmetic.coding_mode = "base_prefix_exact_gpu_cpu"
        config.arithmetic.merge_size = 4

        args = argparse.Namespace(
            overlap_patches=None,
            overlap_stride=None,
            compression_modes=[NON_OVERLAP_MODE],
        )

        with self.assertRaises(ValueError):
            _validate_args(config, args)

    def test_validate_args_allows_model_symbol_to_ignore_arithmetic_merge_size(self) -> None:
        config = ExperimentConfig()
        config.model.implementation = "megabyte"
        config.model.seq_length = 8
        config.model.patch_size = 4
        config.model.input_causal_conv_kernel_size = 1
        config.data.token_merge_size = 3
        config.data.token_merge_alphabet = self.alphabet
        config.arithmetic.coding_mode = "model_symbol"
        config.arithmetic.merge_size = 99

        args = argparse.Namespace(
            overlap_patches=None,
            overlap_stride=None,
            compression_modes=[NON_OVERLAP_MODE],
        )

        _validate_args(config, args)

    def test_compress_source_matches_theoretical_bits_across_coding_modes(self) -> None:
        factorizer = build_fixed_token_arithmetic_factorizer(
            vocab_size=self.regular_vocab_size + 2,
            special_token_ids=[self.pad_id, self.eos_id],
            model_merge_size=self.model_merge_size,
            arithmetic_merge_size=1,
            alphabet=self.alphabet,
        )
        model = _FakeMegabyteModel(self.regular_vocab_size + 2)
        source = b"ACGTNACGTACG"

        model_symbol = compress_source(
            model=model,
            source=source,
            seq_length=4,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode=NON_OVERLAP_MODE,
            token_merge_size=self.model_merge_size,
            token_merge_alphabet=self.alphabet,
            arithmetic_coding_mode="model_symbol",
        )
        factorized = compress_source(
            model=model,
            source=source,
            seq_length=4,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode=NON_OVERLAP_MODE,
            token_merge_size=self.model_merge_size,
            token_merge_alphabet=self.alphabet,
            arithmetic_coding_mode="base_prefix_exact_gpu_cpu",
            arithmetic_merge_size=1,
            factorizer=factorizer,
        )

        self.assertAlmostEqual(
            float(model_symbol["theoretical_bits"]),
            float(factorized["theoretical_bits"]),
            places=5,
        )
        self.assertEqual(factorized["arithmetic_coding_mode"], "base_prefix_exact_gpu_cpu")
        self.assertEqual(int(factorized["arithmetic_merge_size"]), 1)
        self.assertGreater(
            int(factorized["emitted_arithmetic_symbol_count"]),
            int(model_symbol["sample_symbols_with_eos"]),
        )
        self.assertLess(
            int(factorized["arithmetic_vocab_size"]),
            int(model_symbol["arithmetic_vocab_size"]),
        )


if __name__ == "__main__":
    unittest.main()
