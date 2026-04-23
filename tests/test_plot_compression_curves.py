from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.plot_compression_curves import _build_split_mode_rows, generate_artifacts_for_compression_compare


class _FakeAxis:
    def __init__(self) -> None:
        self.transAxes = object()

    def plot(self, *args, **kwargs) -> None:
        del args, kwargs

    def text(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_ylabel(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_title(self, *args, **kwargs) -> None:
        del args, kwargs

    def grid(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_xlabel(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_xticks(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_xticklabels(self, *args, **kwargs) -> None:
        del args, kwargs

    def tick_params(self, *args, **kwargs) -> None:
        del args, kwargs

    def legend(self, *args, **kwargs) -> None:
        del args, kwargs


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: str | Path, dpi: int = 160) -> None:
        del dpi
        Path(path).write_bytes(b"fake-png")


class _FakePyplot:
    def subplots(self, rows: int, cols: int, figsize: tuple[float, float], sharex: bool = False):
        del cols, figsize, sharex
        return _FakeFigure(), [_FakeAxis() for _ in range(rows)]

    def close(self, figure: _FakeFigure) -> None:
        del figure


class PlotCompressionCurvesTests(unittest.TestCase):
    def test_build_split_mode_rows_derives_expected_metrics(self) -> None:
        compression_compare = {
            "dataset": {
                "species": [
                    {"species": "BuEb", "source_name": "BuEb"},
                    {"species": "HoSa", "source_name": "HoSa"},
                ]
            },
            "results": {
                "train": {
                    "windows_nonoverlap": {
                        "per_source": [
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "arithmetic_bits_per_base": 1.5,
                                "theoretical_bits_per_base": 1.4,
                                "compression_bases_per_second": 2_000_000.0,
                                "compression_bytes_per_second": 500_000.0,
                            },
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "arithmetic_bits_per_base": 1.25,
                                "theoretical_bits_per_base": 1.2,
                                "compression_bases_per_second": 3_000_000.0,
                                "compression_bytes_per_second": 750_000.0,
                            },
                        ]
                    }
                }
            },
        }

        rows = _build_split_mode_rows(
            compression_compare=compression_compare,
            split_name="train",
            mode_name="windows_nonoverlap",
        )

        self.assertEqual([row["source_name"] for row in rows], ["BuEb", "HoSa"])
        self.assertAlmostEqual(float(rows[0]["vs_2bit_percent"]), 62.5)
        self.assertAlmostEqual(float(rows[0]["paper_baseline_percent"]), 99.0)
        self.assertAlmostEqual(float(rows[0]["paper_baseline_bpb"]), 1.98)
        self.assertAlmostEqual(float(rows[0]["compression_mbases_per_second"]), 3.0)
        self.assertAlmostEqual(float(rows[1]["compression_mbytes_per_second"]), 0.5)

    def test_generate_artifacts_for_compression_compare_writes_csv_and_png(self) -> None:
        compression_compare = {
            "dataset": {
                "species": [
                    {"species": "BuEb", "source_name": "BuEb"},
                    {"species": "HoSa", "source_name": "HoSa"},
                ]
            },
            "results": {
                "train": {
                    "windows_nonoverlap": {
                        "per_source": [
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.25,
                                "theoretical_bits_per_base": 1.2,
                                "compression_bases_per_second": 3_000_000.0,
                                "compression_bytes_per_second": 750_000.0,
                            },
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.5,
                                "theoretical_bits_per_base": 1.4,
                                "compression_bases_per_second": 2_000_000.0,
                                "compression_bytes_per_second": 500_000.0,
                            },
                        ]
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_dir = Path(tmpdir)
            compression_compare_path = stats_dir / "compression_compare.json"
            compression_compare_path.write_text(__import__("json").dumps(compression_compare), encoding="utf-8")

            with patch("scripts.plot_compression_curves._load_matplotlib_pyplot", return_value=_FakePyplot()):
                generated_paths = generate_artifacts_for_compression_compare(compression_compare_path)

            self.assertEqual(len(generated_paths), 2)
            csv_path = stats_dir / "compression_curves" / "windows_nonoverlap_compression_curve_data.csv"
            png_path = stats_dir / "compression_curves" / "windows_nonoverlap_compression_curves.png"
            self.assertTrue(csv_path.exists())
            self.assertTrue(png_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["source_name"], "BuEb")
            self.assertEqual(rows[0]["vs_2bit_percent"], "62.5")
            self.assertEqual(rows[0]["paper_baseline_percent"], "99.0")
            self.assertEqual(rows[0]["paper_baseline_bpb"], "1.98")


if __name__ == "__main__":
    unittest.main()
