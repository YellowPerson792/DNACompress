from __future__ import annotations

"""Upload DNA experiment outputs to Weights & Biases (W&B).

Example commands:

1) Basic upload (single experiment directory)
     python scripts/upload_to_wandb.py \
         --run-dir outputs/dna_megabyte_1024_4 \
         --project dna-compress

2) Upload with explicit entity/team, run name, group, and tags
     python scripts/upload_to_wandb.py \
         --run-dir outputs/dna_megabyte_1024_4 \
         --project dna-compress \
         --entity my-team \
         --name dna_megabyte_1024_patch4 \
         --group seq1024_patch4 \
         --tags megabyte dna compression seq1024 patch4

3) Offline upload (generate local W&B files; sync later)
     python scripts/upload_to_wandb.py \
         --run-dir outputs/dna_megabyte_1024_4 \
         --project dna-compress \
         --offline

4) Include checkpoints in artifact (best.pt / last.pt if present)
     python scripts/upload_to_wandb.py \
         --run-dir outputs/dna_megabyte_1024_4 \
         --project dna-compress \
         --include-checkpoints \
         --artifact-name dna-megabyte-1024-outputs

5) Upload when output filenames are custom
     python scripts/upload_to_wandb.py \
         --run-dir outputs/dna_megabyte_custom \
         --project dna-compress \
         --resolved-config resolved_config.json \
         --metrics-json metrics.json \
         --compression-json compression_compare.json

6) Recommended command for your current run layout
     python scripts/upload_to_wandb.py \
         --run-dir outputs/dna_megabyte \
         --project dna-compress \
         --entity my-team \
         --group baseline \
         --tags megabyte dna bfloat16 tokenmerge4

Notes:
- At least one of metrics.json or compression_compare.json must exist.
- resolved_config.json is used as W&B config when available.
- The script logs split/mode/species-level compression tables for dataset comparison.
"""

import argparse
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


def _rows_to_wandb_table(rows: list[dict[str, Any]], wandb_module: Any) -> Any | None:
    if not rows:
        return None

    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    data: list[list[Any]] = []
    for row in rows:
        data.append([row.get(column) for column in columns])

    return wandb_module.Table(columns=columns, data=data)


def _collect_artifact_files(
    run_dir: Path,
    resolved_config_path: Path,
    metrics_path: Path,
    compression_compare_path: Path,
    include_checkpoints: bool,
) -> list[Path]:
    files: list[Path] = []
    for path in [resolved_config_path, metrics_path, compression_compare_path]:
        if path.exists() and path.is_file():
            files.append(path)

    if include_checkpoints:
        for checkpoint_name in ["best.pt", "last.pt"]:
            checkpoint_path = run_dir / checkpoint_name
            if checkpoint_path.exists() and checkpoint_path.is_file():
                files.append(checkpoint_path)

    return files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload DNA experiment outputs to Weights & Biases.")
    parser.add_argument("--run-dir", required=True, help="Experiment output directory containing JSON outputs.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", help="W&B entity/team name.")
    parser.add_argument("--name", help="W&B run name. Defaults to run-dir folder name.")
    parser.add_argument("--group", help="Optional W&B group for runs.")
    parser.add_argument("--job-type", default="analysis-upload", help="W&B job_type.")
    parser.add_argument("--tags", nargs="*", default=None, help="Optional W&B tags.")
    parser.add_argument("--notes", help="Optional W&B notes.")
    parser.add_argument("--offline", action="store_true", help="Use offline mode (WANDB_MODE=offline).")
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include best.pt/last.pt in uploaded artifact when present.",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Artifact name. Defaults to '<run_name>-outputs'.",
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found or not a directory: {run_dir}")

    resolved_config_path = run_dir / args.resolved_config
    metrics_path = run_dir / args.metrics_json
    compression_compare_path = run_dir / args.compression_json

    resolved_config = _read_json_if_exists(resolved_config_path)
    metrics = _read_json_if_exists(metrics_path)
    compression_compare = _read_json_if_exists(compression_compare_path)

    if metrics is None and compression_compare is None:
        raise ValueError("Neither metrics.json nor compression_compare.json found. Nothing to upload.")

    if args.offline:
        import os

        os.environ["WANDB_MODE"] = "offline"

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install with: pip install wandb"
        ) from exc

    config_payload: dict[str, Any] = {}
    if isinstance(resolved_config, dict):
        config_payload = resolved_config

    run_name = args.name if args.name else run_dir.name

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        group=args.group,
        job_type=args.job_type,
        tags=args.tags,
        notes=args.notes,
        config=config_payload,
    )

    try:
        summary_source = metrics if isinstance(metrics, dict) else {}
        summary_metrics = _collect_summary_metrics(summary_source, compression_compare)
        if isinstance(resolved_config, dict):
            flat_config: dict[str, Any] = {}
            _flatten_dict("config", resolved_config, flat_config)
            summary_metrics.update(flat_config)
        for key, value in summary_metrics.items():
            run.summary[key] = value

        table_payload: dict[str, Any] = {}

        if isinstance(metrics, dict):
            dataset_rows = _build_dataset_table_rows(metrics.get("dataset"))
            dataset_table = _rows_to_wandb_table(dataset_rows, wandb)
            if dataset_table is not None:
                table_payload["tables/dataset_splits"] = dataset_table

            compression = metrics.get("compression")
            if isinstance(compression, dict):
                per_source_rows = compression.get("per_source")
                if isinstance(per_source_rows, list):
                    normalized_rows: list[dict[str, Any]] = []
                    for row in per_source_rows:
                        if isinstance(row, dict):
                            normalized = {"split": "test", "mode": "legacy", **row}
                            normalized_rows.append(normalized)
                    legacy_table = _rows_to_wandb_table(normalized_rows, wandb)
                    if legacy_table is not None:
                        table_payload["tables/compression_per_source_legacy"] = legacy_table

        aggregate_rows, per_source_rows = _build_compression_tables(compression_compare)
        aggregate_table = _rows_to_wandb_table(aggregate_rows, wandb)
        if aggregate_table is not None:
            table_payload["tables/compression_aggregate_by_split_mode"] = aggregate_table
        per_source_table = _rows_to_wandb_table(per_source_rows, wandb)
        if per_source_table is not None:
            table_payload["tables/compression_per_source_by_split_mode"] = per_source_table

        if table_payload:
            wandb.log(table_payload)

        artifact_name = args.artifact_name or f"{run_name}-outputs"
        artifact = wandb.Artifact(name=artifact_name, type="dna-compression-run")
        artifact_files = _collect_artifact_files(
            run_dir=run_dir,
            resolved_config_path=resolved_config_path,
            metrics_path=metrics_path,
            compression_compare_path=compression_compare_path,
            include_checkpoints=args.include_checkpoints,
        )
        for path in artifact_files:
            artifact.add_file(str(path), name=path.name)

        if artifact_files:
            run.log_artifact(artifact)

        run.summary["upload.run_dir"] = str(run_dir)
        run.summary["upload.artifact_file_count"] = len(artifact_files)
        run.summary["upload.has_resolved_config"] = resolved_config is not None
        run.summary["upload.has_metrics_json"] = metrics is not None
        run.summary["upload.has_compression_compare_json"] = compression_compare is not None
    finally:
        run.finish()


if __name__ == "__main__":
    main()
