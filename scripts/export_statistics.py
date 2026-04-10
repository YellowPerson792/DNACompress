from __future__ import annotations

"""Export the same payload used for W&B upload into local files.

Outputs:
- run_metadata.json
- resolved_config.json (copied when available)
- summary_metrics.csv
- dataset_splits.csv
- compression_per_source_legacy.csv
- compression_aggregate_by_split_mode.csv
- compression_per_source_by_split_mode.csv

Example:
    python scripts/export_statistics.py \
      --run-dir outputs/dna_megabyte_large_all_finished
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_dict(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_dict(next_prefix, nested, out)
        return
    if isinstance(value, list):
        return
    out[prefix] = value


def _dataset_summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dataset = metrics.get("dataset")
    if not isinstance(dataset, dict):
        return result

    species_rows = dataset.get("species")
    if isinstance(species_rows, list):
        result["dataset.species_count"] = len(species_rows)
        total_size = 0
        total_train = 0
        total_val = 0
        total_test = 0
        for row in species_rows:
            if not isinstance(row, dict):
                continue
            total_size += int(row.get("total_size", 0) or 0)
            total_train += int(row.get("train_bytes", 0) or 0)
            total_val += int(row.get("val_bytes", 0) or 0)
            total_test += int(row.get("test_bytes", 0) or 0)
        result["dataset.total_size_bytes"] = total_size
        result["dataset.total_train_bytes"] = total_train
        result["dataset.total_val_bytes"] = total_val
        result["dataset.total_test_bytes"] = total_test

    alphabet_bytes = dataset.get("alphabet_bytes")
    if isinstance(alphabet_bytes, list):
        result["dataset.alphabet_size"] = len(alphabet_bytes)

    return result


def _collect_summary_metrics(metrics: dict[str, Any], compression_compare: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for key in ["device", "model_parameters", "best_val_bits_per_base"]:
        if key in metrics:
            summary[key] = metrics[key]

    validation = metrics.get("validation")
    if isinstance(validation, dict):
        for k, v in validation.items():
            summary[f"validation.{k}"] = v

    test = metrics.get("test")
    if isinstance(test, dict):
        for k, v in test.items():
            summary[f"test.{k}"] = v

    compression = metrics.get("compression")
    if isinstance(compression, dict):
        aggregate = compression.get("aggregate")
        if isinstance(aggregate, dict):
            for k, v in aggregate.items():
                summary[f"compression.aggregate.{k}"] = v

    summary.update(_dataset_summary_from_metrics(metrics))

    if isinstance(compression_compare, dict):
        for key in ["checkpoint_step", "best_val_bpb", "overlap_stride_tokens", "overlap_stride_patches"]:
            if key in compression_compare:
                summary[f"compression_compare.{key}"] = compression_compare[key]
        arithmetic = compression_compare.get("arithmetic")
        if isinstance(arithmetic, dict):
            for key, value in arithmetic.items():
                summary[f"compression_compare.arithmetic.{key}"] = value

    return summary


def _build_dataset_table_rows(dataset: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(dataset, dict):
        return rows

    species_rows = dataset.get("species")
    if not isinstance(species_rows, list):
        return rows

    for row in species_rows:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "species": row.get("species"),
                "total_size": row.get("total_size"),
                "train_bytes": row.get("train_bytes"),
                "val_bytes": row.get("val_bytes"),
                "test_bytes": row.get("test_bytes"),
            }
        )
    return rows


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _build_compression_tables(compression_compare: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []

    if not isinstance(compression_compare, dict):
        return aggregate_rows, per_source_rows

    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return aggregate_rows, per_source_rows

    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for mode_name, mode_payload in split_payload.items():
            if not isinstance(mode_payload, dict):
                continue

            aggregate = mode_payload.get("aggregate")
            if isinstance(aggregate, dict):
                aggregate_row = {"split": split_name, "mode": mode_name}
                aggregate_row.update(aggregate)
                aggregate_rows.append(aggregate_row)

            per_source = mode_payload.get("per_source")
            if not isinstance(per_source, list):
                continue

            for source_row in per_source:
                if not isinstance(source_row, dict):
                    continue

                row = {
                    "split": split_name,
                    "mode": mode_name,
                    "species": source_row.get("species"),
                    "sample_bytes": source_row.get("sample_bytes"),
                    "sample_bases": source_row.get("sample_bases"),
                    "theoretical_bits_per_base": source_row.get("theoretical_bits_per_base"),
                    "arithmetic_bits_per_base": source_row.get("arithmetic_bits_per_base"),
                    "arithmetic_coding_mode": source_row.get("arithmetic_coding_mode"),
                    "arithmetic_merge_size": source_row.get("arithmetic_merge_size"),
                    "arithmetic_frequency_total": source_row.get("arithmetic_frequency_total"),
                    "arithmetic_vocab_size": source_row.get("arithmetic_vocab_size"),
                    "arithmetic_target_uniform_mass": source_row.get("arithmetic_target_uniform_mass"),
                    "arithmetic_effective_uniform_mass": source_row.get("arithmetic_effective_uniform_mass"),
                    "emitted_arithmetic_symbol_count": source_row.get("emitted_arithmetic_symbol_count"),
                    "core_model_theoretical_bits": source_row.get("core_model_theoretical_bits"),
                    "tail_base_count": source_row.get("tail_base_count"),
                    "tail_side_info_bits": source_row.get("tail_side_info_bits"),
                    "gpu_prefix_aggregate_seconds": source_row.get("gpu_prefix_aggregate_seconds"),
                    "cpu_small_alphabet_quantize_seconds": source_row.get("cpu_small_alphabet_quantize_seconds"),
                    "data_transfer_seconds": source_row.get("data_transfer_seconds"),
                    "arithmetic_encode_seconds": source_row.get("arithmetic_encode_seconds"),
                    "compression_process_seconds": source_row.get("compression_process_seconds"),
                    "ascii_bytes": source_row.get("ascii_bytes"),
                    "two_bit_pack_bytes": source_row.get("two_bit_pack_bytes"),
                    "gzip_bytes": source_row.get("gzip_bytes"),
                    "bz2_bytes": source_row.get("bz2_bytes"),
                    "lzma_bytes": source_row.get("lzma_bytes"),
                }

                arithmetic_bpb = source_row.get("arithmetic_bits_per_base")
                if isinstance(arithmetic_bpb, (int, float)):
                    row["arithmetic_vs_2bit_ratio"] = _safe_div(float(arithmetic_bpb), 2.0)

                sample_bytes = source_row.get("sample_bytes")
                arithmetic_bytes = source_row.get("arithmetic_coded_bytes")
                gzip_bytes = source_row.get("gzip_bytes")
                if isinstance(sample_bytes, (int, float)) and isinstance(arithmetic_bytes, (int, float)):
                    row["arithmetic_bytes_ratio_vs_ascii"] = _safe_div(float(arithmetic_bytes), float(sample_bytes))
                if isinstance(gzip_bytes, (int, float)) and isinstance(arithmetic_bytes, (int, float)):
                    row["arithmetic_vs_gzip_ratio"] = _safe_div(float(arithmetic_bytes), float(gzip_bytes))

                per_source_rows.append(row)

    return aggregate_rows, per_source_rows


def _build_legacy_compression_rows(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(metrics, dict):
        return rows

    compression = metrics.get("compression")
    if not isinstance(compression, dict):
        return rows

    per_source = compression.get("per_source")
    if not isinstance(per_source, list):
        return rows

    for row in per_source:
        if not isinstance(row, dict):
            continue
        rows.append({"split": "test", "mode": "legacy", **row})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = [{"metric": key, "value": value} for key, value in sorted(summary.items())]
    _write_csv(path, rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export W&B upload payload into local JSON/CSV files.")
    parser.add_argument("--run-dir", required=True, help="Experiment output directory containing JSON outputs.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for exported files. Defaults to <run-dir>/statistics.",
    )
    parser.add_argument(
        "--resolved-config",
        default="resolved_config.json",
        help="Resolved config file name under run-dir.",
    )
    parser.add_argument(
        "--metrics-json",
        default="metrics.json",
        help="Metrics JSON file name under run-dir.",
    )
    parser.add_argument(
        "--compression-json",
        default="compression_compare.json",
        help="Compression compare JSON file name under run-dir.",
    )
    parser.add_argument("--project", default="", help="Optional project metadata to store in run_metadata.json.")
    parser.add_argument("--entity", default="", help="Optional entity metadata to store in run_metadata.json.")
    parser.add_argument("--name", default=None, help="Optional run name metadata. Defaults to run-dir folder name.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found or not a directory: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "statistics")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / args.resolved_config
    metrics_path = run_dir / args.metrics_json
    compression_compare_path = run_dir / args.compression_json

    resolved_config = _read_json_if_exists(resolved_config_path)
    metrics = _read_json_if_exists(metrics_path)
    compression_compare = _read_json_if_exists(compression_compare_path)

    if metrics is None and compression_compare is None:
        raise ValueError("Neither metrics.json nor compression_compare.json found. Nothing to export.")

    run_name = args.name if args.name else run_dir.name

    run_metadata = {
        "project": args.project,
        "entity": args.entity,
        "name": run_name,
        "run_dir": str(run_dir),
        "has_resolved_config": resolved_config is not None,
        "has_metrics_json": metrics is not None,
        "has_compression_compare_json": compression_compare is not None,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if isinstance(resolved_config, dict):
        (out_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        model_config = resolved_config.get("model")
        if isinstance(model_config, dict):
            (out_dir / "model_config.json").write_text(
                json.dumps(model_config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    summary_source = metrics if isinstance(metrics, dict) else {}
    summary_metrics = _collect_summary_metrics(summary_source, compression_compare)
    if isinstance(resolved_config, dict):
        flat_config: dict[str, Any] = {}
        _flatten_dict("config", resolved_config, flat_config)
        summary_metrics.update(flat_config)
    _write_summary_csv(out_dir / "summary_metrics.csv", summary_metrics)

    dataset_rows = _build_dataset_table_rows(metrics.get("dataset") if isinstance(metrics, dict) else None)
    _write_csv(out_dir / "dataset_splits.csv", dataset_rows)

    legacy_rows = _build_legacy_compression_rows(metrics if isinstance(metrics, dict) else None)
    _write_csv(out_dir / "compression_per_source_legacy.csv", legacy_rows)

    aggregate_rows, per_source_rows = _build_compression_tables(compression_compare)
    _write_csv(out_dir / "compression_aggregate_by_split_mode.csv", aggregate_rows)
    _write_csv(out_dir / "compression_per_source_by_split_mode.csv", per_source_rows)

    print(f"Exported payload files to: {out_dir}")


if __name__ == "__main__":
    main()
