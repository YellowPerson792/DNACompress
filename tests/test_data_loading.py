from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from dna_compress.config import DataConfig, ExperimentConfig
from dna_compress.data import RandomWindowDataset, SequentialWindowDataset, load_splits
from scripts.run_dna_compression import _apply_overrides as apply_compression_overrides
from scripts.run_dna_compression import _build_parser as build_compression_parser
from scripts.run_dna_experiment import _apply_overrides as apply_experiment_overrides
from scripts.run_dna_experiment import _build_parser as build_experiment_parser


class DataLoadingTests(unittest.TestCase):
    def test_flat_file_mode_keeps_legacy_split_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            species_path = Path(tmpdir) / "Legacy"
            species_path.write_bytes(b"ACGTNACGTN")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["Legacy"],
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
            )
            splits = load_splits(config, seq_length=4)

            self.assertEqual(splits.train_sources, [b"ACGTNA"])
            self.assertEqual(splits.val_sources, [b"CG"])
            self.assertEqual(splits.test_sources, [b"TN"])
            self.assertEqual(splits.summary["species"][0]["source_mode"], "flat_file")
            self.assertEqual(splits.summary["species"][0]["source_name"], "Legacy")

    def test_fasta_separate_mode_sanitizes_and_labels_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "yeast" / "dna"
            dna_dir.mkdir(parents=True)
            (dna_dir / "Yeast.dna.chromosome.I.fa").write_text(">chrI\nacgtrxnn\n", encoding="utf-8")
            (dna_dir / "Yeast.dna.chromosome.II.fa").write_text(">chrII\nTTGGTTGGTTGG\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["yeast"],
                multi_sequence_mode="separate",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
            )
            splits = load_splits(config, seq_length=4)

            self.assertEqual(len(splits.train_sources), 2)
            self.assertEqual(splits.summary["species"][0]["source_name"], "yeast:I")
            self.assertEqual(splits.summary["species"][1]["source_name"], "yeast:II")
            self.assertEqual(splits.summary["species"][0]["total_size"], 8)
            self.assertEqual(splits.summary["species"][1]["total_size"], 12)
            self.assertEqual(splits.train_sources[0], b"ACGT")
            self.assertEqual(splits.val_sources[0], b"NN")
            self.assertEqual(splits.test_sources[0], b"NN")

    def test_sequence_include_map_filters_selected_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "human" / "dna"
            dna_dir.mkdir(parents=True)
            (dna_dir / "Human.dna.chromosome.1.fa").write_text(">chr1\nAAAAAAAAAAAA\n", encoding="utf-8")
            (dna_dir / "Human.dna.chromosome.2.fa").write_text(">chr2\nCCCCCCCCCCCC\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["human"],
                multi_sequence_mode="separate",
                sequence_include_map={"human": ["2"]},
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
            )
            splits = load_splits(config, seq_length=4)

            self.assertEqual(len(splits.train_sources), 1)
            self.assertEqual(splits.summary["species"][0]["source_name"], "human:2")

    def test_sequence_include_map_errors_for_unknown_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "worm" / "dna"
            dna_dir.mkdir(parents=True)
            (dna_dir / "Worm.dna.chromosome.I.fa").write_text(">chrI\nAAAAAA\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["worm"],
                sequence_include_map={"worm": ["X"]},
            )

            with self.assertRaises(ValueError):
                load_splits(config, seq_length=4)

    def test_concat_mode_inserts_boundary_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "plant" / "dna"
            dna_dir.mkdir(parents=True)
            (dna_dir / "Plant.dna.chromosome.1.fa").write_text(">chr1\nAAAA\n", encoding="utf-8")
            (dna_dir / "Plant.dna.chromosome.2.fa").write_text(">chr2\nCCCC\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["plant"],
                multi_sequence_mode="concat",
                token_merge_size=2,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
            )
            splits = load_splits(config, seq_length=3)

            summary = splits.summary["species"][0]
            self.assertEqual(summary["source_name"], "plant")
            self.assertEqual(summary["selected_sequence_count"], 2)
            self.assertEqual(summary["total_size"], 14)

    def test_clean_cache_creates_then_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "yeast" / "dna"
            dna_dir.mkdir(parents=True)
            fasta_path = dna_dir / "Yeast.dna.chromosome.I.fa"
            fasta_path.write_text(">chrI\nACGTNACGTN\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["yeast"],
                multi_sequence_mode="separate",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                clean_cache_enabled=True,
            )
            first = load_splits(config, seq_length=4)
            second = load_splits(config, seq_length=4)

            self.assertEqual(first.train_sources, second.train_sources)
            self.assertEqual(first.summary["clean_cache"]["created"], 1)
            self.assertEqual(first.summary["clean_cache"]["hits"], 0)
            self.assertEqual(second.summary["clean_cache"]["hits"], 1)
            self.assertEqual(second.summary["clean_cache"]["created"], 0)
            cache_root = Path(tmpdir) / ".dna_cache" / "clean" / "yeast" / "dna"
            self.assertTrue((cache_root / "Yeast.dna.chromosome.I.fa.clean.bin").exists())
            self.assertTrue((cache_root / "Yeast.dna.chromosome.I.fa.clean.json").exists())

    def test_clean_cache_rebuilds_after_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "worm" / "dna"
            dna_dir.mkdir(parents=True)
            fasta_path = dna_dir / "Worm.dna.chromosome.I.fa"
            fasta_path.write_text(">chrI\nAAAAAAAAAAAA\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["worm"],
                multi_sequence_mode="separate",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                clean_cache_enabled=True,
            )
            first = load_splits(config, seq_length=4)
            fasta_path.write_text(">chrI\nCCCCCCCC\n", encoding="utf-8")
            second = load_splits(config, seq_length=4)

            self.assertEqual(first.summary["clean_cache"]["created"], 1)
            self.assertEqual(second.summary["clean_cache"]["rebuilt"], 1)
            self.assertEqual(second.train_sources[0], b"CCCC")

    def test_clean_cache_only_builds_selected_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "human" / "dna"
            dna_dir.mkdir(parents=True)
            (dna_dir / "Human.dna.chromosome.1.fa").write_text(">chr1\nAAAAAAAA\n", encoding="utf-8")
            (dna_dir / "Human.dna.chromosome.2.fa").write_text(">chr2\nCCCCCCCC\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["human"],
                multi_sequence_mode="separate",
                sequence_include_map={"human": ["2"]},
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                clean_cache_enabled=True,
            )
            splits = load_splits(config, seq_length=4)

            self.assertEqual(len(splits.train_sources), 1)
            cache_root = Path(tmpdir) / ".dna_cache" / "clean" / "human" / "dna"
            self.assertFalse((cache_root / "Human.dna.chromosome.1.fa.clean.bin").exists())
            self.assertTrue((cache_root / "Human.dna.chromosome.2.fa.clean.bin").exists())

    def test_clean_cache_reuses_across_seq_length_and_token_merge_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_dir = Path(tmpdir) / "plant" / "dna"
            dna_dir.mkdir(parents=True)
            (dna_dir / "Plant.dna.chromosome.1.fa").write_text(">chr1\nACGTNACGTNACGTNACGTN\n", encoding="utf-8")

            config = DataConfig(
                dataset_dir=tmpdir,
                species=["plant"],
                multi_sequence_mode="separate",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                clean_cache_enabled=True,
            )
            first = load_splits(config, seq_length=4)
            config.token_merge_size = 6
            second = load_splits(config, seq_length=8)

            self.assertEqual(first.summary["clean_cache"]["created"], 1)
            self.assertEqual(second.summary["clean_cache"]["hits"], 1)
            self.assertEqual(second.summary["clean_cache"]["created"], 0)

    def test_megabyte_datasets_use_compact_token_storage_for_plain_bases(self) -> None:
        train_dataset = RandomWindowDataset(
            sources=[b"ACGTNACGTN"],
            seq_length=4,
            samples_per_epoch=3,
            seed=123,
            token_merge_size=1,
            token_merge_alphabet="ACGTN",
        )
        self.assertEqual(train_dataset.sources[0].dtype.name, "uint8")
        first_item = train_dataset[0]["input_ids"]
        self.assertEqual(first_item.dtype, torch.int64)
        self.assertEqual(first_item.shape[0], 4)

        eval_dataset = SequentialWindowDataset(
            sources=[b"ACGTNACG"],
            seq_length=4,
            pad_id=257,
            token_merge_size=1,
            token_merge_alphabet="ACGTN",
        )
        self.assertEqual(eval_dataset.sources[0].dtype.name, "uint8")
        self.assertEqual(eval_dataset[0]["input_ids"].tolist(), [65, 67, 71, 84])

    def test_megabyte_datasets_use_compact_token_storage_for_merged_tokens(self) -> None:
        train_dataset = RandomWindowDataset(
            sources=[b"ACGTNACGTNAC"],
            seq_length=2,
            samples_per_epoch=2,
            seed=7,
            token_merge_size=2,
            token_merge_alphabet="ACGTN",
        )
        self.assertEqual(train_dataset.sources[0].dtype.name, "uint8")

        eval_dataset = SequentialWindowDataset(
            sources=[b"ACGTNACGTNAC"],
            seq_length=2,
            pad_id=257,
            token_merge_size=2,
            token_merge_alphabet="ACGTN",
        )
        first_chunk = eval_dataset[0]["input_ids"].tolist()
        self.assertEqual(first_chunk, [1, 13])


class MegabyteCliOverrideTests(unittest.TestCase):
    def test_experiment_cli_parses_new_sequence_options(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(
            [
                "--config",
                "dummy.json",
                "--sequence-source-mode",
                "fasta_dir",
                "--multi-sequence-mode",
                "separate",
                "--clean-cache-enabled",
                "--clean-cache-dir",
                "tmp_cache",
                "--sequence-include",
                "human=1,2,X",
                "--sequence-include",
                "mouse=1,MT",
            ]
        )
        config = ExperimentConfig()
        apply_experiment_overrides(config, args)

        self.assertEqual(config.data.sequence_source_mode, "fasta_dir")
        self.assertEqual(config.data.multi_sequence_mode, "separate")
        self.assertTrue(config.data.clean_cache_enabled)
        self.assertEqual(config.data.clean_cache_dir, "tmp_cache")
        self.assertEqual(config.data.sequence_include_map["human"], ["1", "2", "X"])
        self.assertEqual(config.data.sequence_include_map["mouse"], ["1", "MT"])

    def test_compression_cli_parses_new_sequence_options(self) -> None:
        parser = build_compression_parser()
        args = parser.parse_args(
            [
                "--sequence-source-mode",
                "fasta_dir",
                "--multi-sequence-mode",
                "concat",
                "--no-clean-cache",
                "--sequence-include",
                "yeast=I,II",
            ]
        )
        config = ExperimentConfig()
        apply_compression_overrides(config, args)

        self.assertEqual(config.data.sequence_source_mode, "fasta_dir")
        self.assertEqual(config.data.multi_sequence_mode, "concat")
        self.assertFalse(config.data.clean_cache_enabled)
        self.assertEqual(config.data.sequence_include_map["yeast"], ["I", "II"])


if __name__ == "__main__":
    unittest.main()
