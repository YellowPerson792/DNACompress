from __future__ import annotations

import unittest

import torch

from dna_compress.config import ExperimentConfig
from dna_compress.dnagpt_compression import compress_dnagpt_source
from dna_compress.dnagpt_experiment import validate_dnagpt_config
from dna_compress.dnagpt_loader import build_dnagpt_tokenizer, get_variant_spec
from dna_compress.dnagpt_prefix_coding import (
    DNAGPT_PREFIX_ALPHABET_SIZE,
    build_dnagpt_prefix_trie,
    factorize_dnagpt_log_probs_to_base_prefix_stream,
    factorize_dnagpt_log_probs_to_grouped_prefix_stream,
)


class _FakeDNAGPTModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, batch_input: torch.Tensor):
        batch_size, seq_length = batch_input.shape
        base_logits = torch.linspace(1.5, -1.5, self.vocab_size, device=batch_input.device, dtype=torch.float32)
        logits = base_logits.view(1, 1, -1).repeat(batch_size, seq_length, 1)
        position_bias = torch.arange(seq_length, device=batch_input.device, dtype=torch.float32).view(1, seq_length, 1) * 0.03
        token_bias = (batch_input.to(dtype=torch.float32).unsqueeze(-1) % 13.0) * 0.005
        return logits + position_bias + token_bias


class DNAGPTPrefixCodingTests(unittest.TestCase):
    def _factorized_row_log_prob(
        self,
        factorized,
        row_index: int,
    ) -> float:
        emitted_log_prob = 0.0
        for probabilities, symbols, row_positions in zip(
            factorized.step_probabilities,
            factorized.step_symbols,
            factorized.step_row_positions,
        ):
            position = int(row_positions[row_index].item())
            if position < 0:
                continue
            emitted_log_prob += float(torch.log(probabilities[position, symbols[position]]).item())
        return emitted_log_prob

    def test_trie_metadata_matches_dynamic_and_fixed_layouts(self) -> None:
        for variant in ("dna_gpt0.1b_m", "dna_gpt0.1b_h"):
            tokenizer = build_dnagpt_tokenizer(variant)
            trie = build_dnagpt_prefix_trie(tokenizer)
            self.assertEqual(trie.max_token_length, 6)
            self.assertEqual(int(trie.reserved_token_ids.numel()), 34)
            self.assertEqual(int(trie.node_depths.shape[0]), 19531)
            self.assertEqual(tuple(trie.root_child_indices.shape), (5,))
            self.assertTrue(torch.all(trie.root_child_indices > 0))

    def test_base_factorization_matches_target_token_probability(self) -> None:
        tokenizer = build_dnagpt_tokenizer("dna_gpt0.1b_m")
        trie = build_dnagpt_prefix_trie(tokenizer)
        target_pieces = ["A", "NAG", "CTAGCT"]
        target_ids = torch.tensor([tokenizer.piece_to_id(piece) for piece in target_pieces], dtype=torch.long)
        logits = torch.randn((len(target_pieces), len(tokenizer)), dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)

        factorized = factorize_dnagpt_log_probs_to_base_prefix_stream(log_probs, target_ids, trie)

        self.assertEqual(tuple(factorized.emitted_probabilities.shape), (3, 6, DNAGPT_PREFIX_ALPHABET_SIZE))
        for row_index, piece in enumerate(target_pieces):
            valid_mask = factorized.emitted_valid_mask[row_index]
            emitted_probabilities = factorized.emitted_probabilities[row_index][valid_mask]
            emitted_symbols = factorized.emitted_symbols[row_index][valid_mask]
            emitted_log_prob = torch.log(
                emitted_probabilities.gather(1, emitted_symbols.unsqueeze(1)).squeeze(1)
            ).sum()
            self.assertAlmostEqual(
                emitted_log_prob.item(),
                factorized.target_log_probs[row_index].item(),
                places=5,
            )
            expected_steps = 6 if len(piece) == 6 else len(piece) + 1
            self.assertEqual(int(valid_mask.sum().item()), expected_steps)

    def test_grouped_factorization_matches_target_probability_for_all_supported_variants(self) -> None:
        for variant, merge_sizes in {
            "dna_gpt0.1b_m": (1, 2, 3, 6),
            "dna_gpt0.1b_h": (2, 3, 6),
        }.items():
            tokenizer = build_dnagpt_tokenizer(variant)
            trie = build_dnagpt_prefix_trie(tokenizer)
            target_pieces = ["A", "NAGCTA", "CTAGCT"] if variant.endswith("_m") else ["AAAAAA", "NAGCTA", "CTAGCT"]
            target_ids = torch.tensor([tokenizer.piece_to_id(piece) for piece in target_pieces], dtype=torch.long)
            logits = torch.randn((len(target_pieces), len(tokenizer)), dtype=torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)

            for merge_size in merge_sizes:
                factorized = factorize_dnagpt_log_probs_to_grouped_prefix_stream(
                    log_probs=log_probs,
                    target_token_ids=target_ids,
                    trie=trie,
                    merge_size=merge_size,
                )
                for row_index in range(target_ids.shape[0]):
                    self.assertAlmostEqual(
                        self._factorized_row_log_prob(factorized, row_index),
                        factorized.target_log_probs[row_index].item(),
                        places=5,
                    )

    def test_validate_dnagpt_config_rejects_invalid_merge_size_and_mode_combination(self) -> None:
        config = ExperimentConfig()
        config.model.implementation = "dnagpt"
        config.model.variant = "dna_gpt0.1b_m"
        config.arithmetic.coding_mode = "model_symbol"
        config.arithmetic.merge_size = 2
        with self.assertRaises(ValueError):
            validate_dnagpt_config(config)

        config.arithmetic.coding_mode = "base_prefix_exact_gpu_cpu"
        config.arithmetic.merge_size = 7
        with self.assertRaises(ValueError):
            validate_dnagpt_config(config)

        config.arithmetic.merge_size = 2
        validate_dnagpt_config(config)

    def test_grouped_compression_adds_tail_side_info_and_preserves_core_bits(self) -> None:
        variant = "dna_gpt0.1b_m"
        tokenizer = build_dnagpt_tokenizer(variant)
        trie = build_dnagpt_prefix_trie(tokenizer)
        spec = get_variant_spec(variant)
        model = _FakeDNAGPTModel(len(tokenizer))
        source = b"ACGTNNAGCTAAG"
        truncated_core = source[:12]

        baseline = compress_dnagpt_source(
            model=model,
            species="HoSa",
            source=truncated_core,
            tokenizer=tokenizer,
            kmer_size=spec.kmer_size,
            dynamic_kmer=spec.dynamic_kmer,
            species_prefix_map=None,
            seq_length=32,
            pad_id=tokenizer.pad_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode="windows_nonoverlap",
            arithmetic_frequency_total=None,
            arithmetic_target_uniform_mass=0.01,
            arithmetic_coding_mode="model_symbol",
            arithmetic_merge_size=1,
            prefix_trie=trie,
        )
        grouped = compress_dnagpt_source(
            model=model,
            species="HoSa",
            source=source,
            tokenizer=tokenizer,
            kmer_size=spec.kmer_size,
            dynamic_kmer=spec.dynamic_kmer,
            species_prefix_map=None,
            seq_length=32,
            pad_id=tokenizer.pad_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode="windows_nonoverlap",
            arithmetic_frequency_total=None,
            arithmetic_target_uniform_mass=0.01,
            arithmetic_coding_mode="base_prefix_exact_gpu_cpu",
            arithmetic_merge_size=2,
            prefix_trie=trie,
        )

        self.assertAlmostEqual(
            float(grouped["core_model_theoretical_bits"]),
            float(baseline["theoretical_bits"]),
            places=5,
        )
        self.assertEqual(int(grouped["tail_base_count"]), 1)
        self.assertEqual(int(grouped["tail_side_info_bits"]), 6)
        self.assertAlmostEqual(
            float(grouped["theoretical_bits"]),
            float(grouped["core_model_theoretical_bits"]) + 6.0,
            places=5,
        )
        self.assertEqual(grouped["arithmetic_coding_mode"], "base_prefix_exact_gpu_cpu")
        self.assertEqual(int(grouped["arithmetic_merge_size"]), 2)
        self.assertGreater(int(grouped["emitted_arithmetic_symbol_count"]), 0)
        self.assertGreater(int(grouped["arithmetic_vocab_size"]), DNAGPT_PREFIX_ALPHABET_SIZE)


if __name__ == "__main__":
    unittest.main()
