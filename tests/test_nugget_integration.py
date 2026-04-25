from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from dna_compress.config import ExperimentConfig
from dna_compress.data import load_splits
from dna_compress.nugget_compression import _stored_hidden_payload, validate_nugget_hidden_policy
from dna_compress.nugget_experiment import _build_nugget_cache_source_descriptors, validate_nugget_config
from dna_compress.nugget_loader import _resolve_backbone_spec
from dna_compress.nugget_tokenization import (
    NuggetCacheSourceDescriptor,
    build_nugget_tokenizer_spec,
    tokenize_nugget_source,
    tokenize_nugget_sources_with_cache,
)


class NuggetIntegrationTests(unittest.TestCase):
    def test_hidden_policy_accepts_only_consistent_pairs(self) -> None:
        validate_nugget_hidden_policy("runtime_hidden", "runtime")
        validate_nugget_hidden_policy("stored_hidden", "float16")
        with self.assertRaises(ValueError):
            validate_nugget_hidden_policy("runtime_hidden", "float16")
        with self.assertRaises(ValueError):
            validate_nugget_hidden_policy("stored_hidden", "runtime")

    def test_stored_hidden_payload_counts_cast_dtype_bytes(self) -> None:
        encoding = torch.zeros((2, 4, 8), dtype=torch.float32)
        scores = torch.zeros((2, 4), dtype=torch.float32)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

        runtime = _stored_hidden_payload(
            encoding=encoding,
            scores=scores,
            mask=mask,
            hidden_mode="runtime_hidden",
            hidden_storage_dtype="runtime",
            requires_scores_side_info=False,
        )
        self.assertEqual(runtime["hidden_bytes"], 3 * 8 * 4)
        self.assertEqual(runtime["metadata_bytes"], 2 * 4)
        self.assertFalse(runtime["cast_applied"])

        stored = _stored_hidden_payload(
            encoding=encoding,
            scores=scores,
            mask=mask,
            hidden_mode="stored_hidden",
            hidden_storage_dtype="float16",
            requires_scores_side_info=True,
        )
        self.assertEqual(stored["hidden_bytes"], 3 * 8 * 2)
        self.assertEqual(stored["score_bytes"], 3 * 2)
        self.assertEqual(stored["metadata_bytes"], 2 * 4 + 3 * 2)
        self.assertTrue(stored["cast_applied"])
        self.assertEqual(stored["stored_encoding"].dtype, torch.float16)

    def test_fixed_kmer_tokenizer_preserves_tail_for_side_info(self) -> None:
        config = ExperimentConfig()
        config.model.implementation = "nugget"
        config.data.nugget_tokenizer = "fixed_kmer"
        config.data.token_merge_size = 3
        spec = build_nugget_tokenizer_spec(config.data, config.model)
        tokenized = tokenize_nugget_source(species="HoSa", source_name="HoSa", source=b"ACGTN", spec=spec)
        self.assertIsInstance(tokenized.dna_token_ids, np.ndarray)
        self.assertIsInstance(tokenized.dna_token_base_lengths, np.ndarray)
        self.assertEqual(len(tokenized.dna_token_ids), 1)
        self.assertEqual(tokenized.dna_token_base_lengths.tolist(), [3])
        self.assertEqual(tokenized.tail_sequence, "TN")
        self.assertEqual(tokenized.total_bases, 5)

    def test_nugget_cache_reuses_megabyte_clean_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "human" / "dna"
            dna_dir.mkdir(parents=True)
            fasta_path = dna_dir / "Human.dna.chromosome.1.fa"
            fasta_path.write_text(">chr1\nACGTNACGTNAC\n", encoding="utf-8")

            config = ExperimentConfig()
            config.model.implementation = "nugget"
            config.data.dataset_dir = tmpdir
            config.data.species = ["human"]
            config.data.nugget_tokenizer = "fixed_kmer"
            config.data.token_merge_size = 3
            config.data.multi_sequence_mode = "separate"
            config.data.train_ratio = 0.6
            config.data.val_ratio = 0.2
            config.data.test_ratio = 0.2
            config.data.clean_cache_enabled = True

            spec = build_nugget_tokenizer_spec(config.data, config.model)

            first_splits = load_splits(config.data, seq_length=4)
            first_descriptors = _build_nugget_cache_source_descriptors(first_splits.train_sources, first_splits, split_scope="train")
            first_tokenized, first_stats = tokenize_nugget_sources_with_cache(
                source_descriptors=first_descriptors,
                spec=spec,
                dataset_dir=Path(config.data.dataset_dir),
                cache_enabled=True,
                cache_dir=config.data.clean_cache_dir,
                species_prefix_map=config.data.species_prefix_map,
                split_scope="train",
            )
            self.assertEqual(first_stats.created, 1)
            self.assertEqual(first_stats.hits, 0)
            self.assertIsNotNone(first_descriptors[0].clean_cache_path)
            self.assertIsInstance(first_tokenized[0].dna_token_ids, np.ndarray)
            cache_files = list((Path(config.data.dataset_dir) / ".dna_cache" / "nugget_tokens").glob("**/*.npz"))
            self.assertEqual(len(cache_files), 1)

            second_splits = load_splits(config.data, seq_length=8)
            second_descriptors = _build_nugget_cache_source_descriptors(second_splits.train_sources, second_splits, split_scope="train")
            second_tokenized, second_stats = tokenize_nugget_sources_with_cache(
                source_descriptors=second_descriptors,
                spec=spec,
                dataset_dir=Path(config.data.dataset_dir),
                cache_enabled=True,
                cache_dir=config.data.clean_cache_dir,
                species_prefix_map=config.data.species_prefix_map,
                split_scope="train",
            )
            self.assertEqual(second_stats.hits, 1)
            self.assertEqual(second_stats.created, 0)
            self.assertEqual(second_tokenized[0].dna_token_ids.tolist(), first_tokenized[0].dna_token_ids.tolist())
            self.assertEqual(first_descriptors[0].clean_cache_path, second_descriptors[0].clean_cache_path)

            fasta_path.write_text(">chr1\nTTTTTTTTTTTT\n", encoding="utf-8")
            rebuilt_splits = load_splits(config.data, seq_length=4)
            rebuilt_descriptors = _build_nugget_cache_source_descriptors(rebuilt_splits.train_sources, rebuilt_splits, split_scope="train")
            _, rebuilt_stats = tokenize_nugget_sources_with_cache(
                source_descriptors=rebuilt_descriptors,
                spec=spec,
                dataset_dir=Path(config.data.dataset_dir),
                cache_enabled=True,
                cache_dir=config.data.clean_cache_dir,
                species_prefix_map=config.data.species_prefix_map,
                split_scope="train",
            )
            self.assertEqual(rebuilt_stats.hits, 0)
            self.assertEqual(rebuilt_stats.created, 1)

    def test_nugget_cache_key_distinguishes_split_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig()
            config.model.implementation = "nugget"
            config.data.dataset_dir = tmpdir
            config.data.nugget_tokenizer = "byte"
            spec = build_nugget_tokenizer_spec(config.data, config.model)
            descriptor_kwargs = {
                "species": "HoSa",
                "source_name": "HoSa",
                "payload": b"ACGTACGT",
                "source_path": None,
                "split_start": 0,
                "split_length": 8,
            }
            train_descriptor = [NuggetCacheSourceDescriptor(**descriptor_kwargs)]
            _, train_stats = tokenize_nugget_sources_with_cache(
                source_descriptors=train_descriptor,
                spec=spec,
                dataset_dir=Path(tmpdir),
                cache_enabled=True,
                cache_dir=None,
                split_scope="train",
            )
            _, val_stats = tokenize_nugget_sources_with_cache(
                source_descriptors=train_descriptor,
                spec=spec,
                dataset_dir=Path(tmpdir),
                cache_enabled=True,
                cache_dir=None,
                split_scope="val",
            )
            self.assertEqual(train_stats.created, 1)
            self.assertEqual(val_stats.created, 1)
            self.assertEqual(len(list((Path(tmpdir) / ".dna_cache" / "nugget_tokens").glob("**/*.npz"))), 2)

    def test_validate_arithmetic_mode_matches_tokenizer(self) -> None:
        config = ExperimentConfig()
        config.model.implementation = "nugget"
        config.arithmetic.coding_mode = "fixed_token_units"
        config.data.nugget_tokenizer = "byte"
        with self.assertRaises(ValueError):
            validate_nugget_config(config)

        config.data.nugget_tokenizer = "fixed_kmer"
        config.data.token_merge_size = 3
        validate_nugget_config(config)

        config.arithmetic.coding_mode = "base_prefix_exact_gpu_cpu"
        with self.assertRaises(ValueError):
            validate_nugget_config(config)

    def test_bart_decoder_start_tracks_tokenizer_eos_when_available(self) -> None:
        config = ExperimentConfig()
        config.model.implementation = "nugget"
        config.model.nugget_backbone = "bart"

        config.data.nugget_tokenizer = "byte"
        byte_spec = build_nugget_tokenizer_spec(config.data, config.model)
        byte_backbone = _resolve_backbone_spec(config.model, byte_spec)
        self.assertEqual(byte_backbone.decoder_start_token_id, byte_spec.eos_id)
        self.assertEqual(byte_backbone.decoder_start_source, "eos_id")
        self.assertIn("facebook_bart_base_config.json", byte_backbone.config_source)

        config.data.nugget_tokenizer = "dnagpt_kmer"
        dnagpt_spec = build_nugget_tokenizer_spec(config.data, config.model)
        dnagpt_backbone = _resolve_backbone_spec(config.model, dnagpt_spec)
        self.assertIsNone(dnagpt_spec.eos_id)
        self.assertEqual(dnagpt_backbone.decoder_start_token_id, dnagpt_spec.pad_id)
        self.assertEqual(dnagpt_backbone.decoder_start_source, "pad_id_no_eos")


if __name__ == "__main__":
    unittest.main()
