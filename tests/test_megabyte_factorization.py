from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from dna_compress.compression_eval import NON_OVERLAP_MODE, OVERLAP_MODE, SLIDING_TOKEN_MODE, compress_source
from dna_compress.config import ExperimentConfig
from dna_compress.fixed_token_factorization import (
    build_fixed_token_arithmetic_factorizer,
    factorize_fixed_token_log_probs,
)
from dna_compress.tokenization import tokenize_source_bytes
from scripts.run_dna_compression import _export_position_bits_profile_artifacts, _validate_args


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

    def _build_single_base_model(self) -> tuple[_FakeMegabyteModel, int, int]:
        pad_id = 256
        eos_id = 257
        return _FakeMegabyteModel(eos_id + 1), pad_id, eos_id

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

    def test_position_bits_profile_tracks_nonoverlap_base_positions(self) -> None:
        model, pad_id, eos_id = self._build_single_base_model()
        source = b"ACGTN"
        metrics = compress_source(
            model=model,
            source=source,
            seq_length=4,
            pad_id=pad_id,
            eos_id=eos_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode=NON_OVERLAP_MODE,
            token_merge_size=1,
            token_merge_alphabet=self.alphabet,
            arithmetic_coding_mode="model_symbol",
            collect_position_bits_profile=True,
        )

        profile = metrics["position_bits_profile"]
        alphabet = list(str(profile["alphabet"]))
        counts = profile["counts"]
        sum_bits = profile["sum_bits_per_base"]
        base_to_index = {base: index for index, base in enumerate(alphabet)}

        self.assertEqual(int(metrics["sample_bases"]), 5)
        self.assertEqual(sum(sum(int(value) for value in row) for row in counts), 5)
        self.assertEqual(int(counts[base_to_index["A"]][0]), 1)
        self.assertEqual(int(counts[base_to_index["C"]][1]), 1)
        self.assertEqual(int(counts[base_to_index["G"]][2]), 1)
        self.assertEqual(int(counts[base_to_index["T"]][3]), 1)
        self.assertEqual(int(counts[base_to_index["N"]][0]), 1)

        windows = torch.tensor(
            [
                [ord("A"), ord("C"), ord("G"), ord("T")],
                [ord("N"), eos_id, pad_id, pad_id],
            ],
            dtype=torch.long,
        )
        with torch.no_grad():
            log_probs = torch.log_softmax(model(windows, return_loss=False).lm_logits, dim=-1)

        expected_pairs = {
            ("A", 0): float((-log_probs[0, 0, ord("A")] / torch.log(torch.tensor(2.0))).item()),
            ("C", 1): float((-log_probs[0, 1, ord("C")] / torch.log(torch.tensor(2.0))).item()),
            ("G", 2): float((-log_probs[0, 2, ord("G")] / torch.log(torch.tensor(2.0))).item()),
            ("T", 3): float((-log_probs[0, 3, ord("T")] / torch.log(torch.tensor(2.0))).item()),
            ("N", 0): float((-log_probs[1, 0, ord("N")] / torch.log(torch.tensor(2.0))).item()),
        }
        for (base, position), expected_bits in expected_pairs.items():
            self.assertAlmostEqual(float(sum_bits[base_to_index[base]][position]), expected_bits, places=5)

        expected_total_bits = sum(expected_pairs.values())
        self.assertAlmostEqual(float(profile["total_bits"]), expected_total_bits, places=5)
        self.assertAlmostEqual(float(metrics["position_bits_profile_total_bits"]), expected_total_bits, places=5)
        self.assertTrue(bool(metrics["position_bits_profile_excludes_eos"]))

    def test_position_bits_profile_evenly_splits_merged_token_bits(self) -> None:
        model = _FakeMegabyteModel(self.regular_vocab_size + 2)
        source = b"ACGTNACGTACG"
        metrics = compress_source(
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
            collect_position_bits_profile=True,
        )

        profile = metrics["position_bits_profile"]
        token_symbols = tokenize_source_bytes(source, self.model_merge_size, self.alphabet)
        symbols = token_symbols + [self.eos_id]
        window_starts = list(range(0, len(symbols), 4)) or [0]
        windows = torch.full((len(window_starts), 4), self.pad_id, dtype=torch.long)
        lengths: list[int] = []
        for row_index, start in enumerate(window_starts):
            chunk = symbols[start : start + 4]
            lengths.append(len(chunk))
            if chunk:
                windows[row_index, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)

        with torch.no_grad():
            log_probs = torch.log_softmax(model(windows, return_loss=False).lm_logits, dim=-1)

        expected_non_eos_bits = 0.0
        expected_eos_bits = 0.0
        token_count_without_eos = len(token_symbols)
        for row_index, start in enumerate(window_starts):
            chunk_length = lengths[row_index]
            for local_index in range(chunk_length):
                target_symbol = symbols[start + local_index]
                bits = float((-log_probs[row_index, local_index, target_symbol] / torch.log(torch.tensor(2.0))).item())
                if start + local_index < token_count_without_eos:
                    expected_non_eos_bits += bits
                else:
                    expected_eos_bits += bits

        total_profile_bits = sum(sum(float(value) for value in row) for row in profile["sum_bits_per_base"])
        total_profile_count = sum(sum(int(value) for value in row) for row in profile["counts"])
        self.assertEqual(total_profile_count, int(metrics["sample_bases"]))
        self.assertAlmostEqual(total_profile_bits, expected_non_eos_bits, places=5)
        self.assertAlmostEqual(float(metrics["position_bits_profile_total_bits"]), expected_non_eos_bits, places=5)
        self.assertAlmostEqual(float(metrics["theoretical_bits"]), expected_non_eos_bits + expected_eos_bits, places=5)

    def test_position_bits_profile_ignores_trailing_partial_merged_token(self) -> None:
        model = _FakeMegabyteModel(self.regular_vocab_size + 2)
        source = b"ACGTNACGTACGAA"
        metrics = compress_source(
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
            collect_position_bits_profile=True,
        )

        token_symbols = tokenize_source_bytes(source, self.model_merge_size, self.alphabet)
        expected_base_count = len(token_symbols) * self.model_merge_size
        profile = metrics["position_bits_profile"]
        total_profile_count = sum(sum(int(value) for value in row) for row in profile["counts"])

        self.assertEqual(expected_base_count, 12)
        self.assertEqual(int(metrics["sample_bases"]), expected_base_count)
        self.assertEqual(total_profile_count, expected_base_count)

    def test_position_bits_profile_tracks_overlap_suffix_positions(self) -> None:
        model, pad_id, eos_id = self._build_single_base_model()
        source = b"ACGTNA"
        metrics = compress_source(
            model=model,
            source=source,
            seq_length=4,
            pad_id=pad_id,
            eos_id=eos_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode=OVERLAP_MODE,
            overlap_stride=2,
            token_merge_size=1,
            token_merge_alphabet=self.alphabet,
            arithmetic_coding_mode="model_symbol",
            collect_position_bits_profile=True,
        )

        profile = metrics["position_bits_profile"]
        alphabet = list(str(profile["alphabet"]))
        counts = profile["counts"]
        base_to_index = {base: index for index, base in enumerate(alphabet)}

        expected_counts = {
            ("A", 0): 1,
            ("C", 1): 1,
            ("G", 2): 1,
            ("T", 3): 1,
            ("N", 2): 1,
            ("A", 3): 1,
        }
        self.assertEqual(sum(sum(int(value) for value in row) for row in counts), 6)
        for (base, position), expected_count in expected_counts.items():
            self.assertEqual(int(counts[base_to_index[base]][position]), expected_count)

    def test_sliding_mode_ignores_position_bits_profile_collection(self) -> None:
        model, pad_id, eos_id = self._build_single_base_model()
        metrics = compress_source(
            model=model,
            source=b"ACGTN",
            seq_length=4,
            pad_id=pad_id,
            eos_id=eos_id,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size=2,
            requested_bytes=None,
            mode=SLIDING_TOKEN_MODE,
            token_merge_size=1,
            token_merge_alphabet=self.alphabet,
            arithmetic_coding_mode="model_symbol",
            collect_position_bits_profile=True,
        )

        self.assertNotIn("position_bits_profile", metrics)
        self.assertNotIn("position_bits_profile_total_bits", metrics)

    def test_export_position_bits_profile_artifacts_writes_csv_and_png(self) -> None:
        profile = {
            "alphabet": "ACGTN",
            "window_base_length": 2,
            "counts": [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            "sum_bits_per_base": [
                [1.25, 0.0],
                [0.0, 0.75],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "total_bits": 2.0,
            "excludes_eos": True,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = _export_position_bits_profile_artifacts(
                out_dir=Path(temp_dir),
                split_name="test",
                mode_name="windows_nonoverlap",
                species_name="AeCa",
                source_name="AeCa:chr 1",
                profile=profile,
            )

            csv_path = Path(str(metadata["position_bits_curve_csv"]))
            png_path = Path(str(metadata["position_bits_curve_png"]))
            self.assertTrue(bool(metadata["position_bits_profile_enabled"]))
            self.assertTrue(csv_path.exists())
            self.assertTrue(png_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), int(profile["window_base_length"]))
            self.assertEqual(sum(int(row["count"]) for row in rows), 2)
            self.assertAlmostEqual(
                sum(float(row["sum_bits_per_base"]) for row in rows),
                float(metadata["position_bits_profile_total_bits"]),
                places=5,
            )
            self.assertEqual(rows[0]["window_position_zero_based"], "0")
            self.assertEqual(rows[1]["window_position_zero_based"], "1")
            self.assertNotIn("base", rows[0])
            self.assertTrue(bool(metadata["position_bits_profile_excludes_eos"]))


if __name__ == "__main__":
    unittest.main()
