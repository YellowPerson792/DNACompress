from __future__ import annotations

"""Generate per-source compression comparison plots from exported compression results.

This script scans an output directory recursively for ``compression_compare.json`` files.
For each statistics directory that contains one, it generates per split/mode artifacts:

- ``<split>_<mode>_compression_curves.png``: a 3-panel plot showing
  arithmetic bits-per-base, percentage of 2-bit encoding, and compression speed.
- ``<split>_<mode>_compression_curve_data.csv``: the tabular values used in the plot.

Example:

    python scripts/plot_compression_curves.py \
      --root-dir outputs/dna_dnagpt_0p1bm_all_finetune
"""

import argparse
import csv
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any


PAPER_BASELINE_PERCENT_BY_SOURCE: dict[str, float] = {
    "OrSa": 80.0,
    "HoSa": 82.0,
    "GaGa": 91.0,
    "DaRe": 74.0,
    "DrMe": 93.0,
    "EnIn": 79.0,
    "ScPo": 95.0,
    "PlFa": 87.0,
    "EsCo": 95.0,
    "HePy": 91.0,
    "AeCa": 97.0,
    "HaHi": 93.0,
    "YeMi": 94.0,
    "BuEb": 99.0,
    "AgPh": 99.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matplotlib_pyplot():
    try:
        matplotlib = importlib.import_module("matplotlib")
    except ImportError as error:
        raise RuntimeError(
            "Compression curve export requires matplotlib. Install it or rerun without this script."
        ) from error
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return sanitized or "artifact"


def _artifact_stem(split_name: str, mode_name: str) -> str:
    sanitized_split = _sanitize_filename(split_name)
    sanitized_mode = _sanitize_filename(mode_name)
    if sanitized_split == "train" and sanitized_mode.startswith("windows_"):
        return sanitized_mode
    return f"{sanitized_split}_{sanitized_mode}"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _source_order_map(compression_compare: dict[str, Any]) -> dict[str, int]:
    dataset = compression_compare.get("dataset")
    if not isinstance(dataset, dict):
        return {}

    species_rows = dataset.get("species")
    if not isinstance(species_rows, list):
        return {}

    order: dict[str, int] = {}
    for index, row in enumerate(species_rows):
        if not isinstance(row, dict):
            continue
        source_name = row.get("source_name")
        species = row.get("species")
        if isinstance(source_name, str) and source_name not in order:
            order[source_name] = index
        if isinstance(species, str) and species not in order:
            order[species] = index
    return order


def _paper_baseline_percent(*, source_name: str, species_name: str) -> float | None:
    for key in (source_name, species_name):
        if key in PAPER_BASELINE_PERCENT_BY_SOURCE:
            return PAPER_BASELINE_PERCENT_BY_SOURCE[key]
    return None


def _resolve_run_name(stats_dir: Path, compression_compare: dict[str, Any]) -> str:
    run_metadata_path = stats_dir / "run_metadata.json"
    if run_metadata_path.exists():
        run_metadata = _read_json(run_metadata_path)
        if isinstance(run_metadata.get("name"), str) and run_metadata["name"]:
            return str(run_metadata["name"])

    resolved_config = compression_compare.get("resolved_config")
    if isinstance(resolved_config, dict):
        output_cfg = resolved_config.get("output")
        if isinstance(output_cfg, dict):
            wandb_name = output_cfg.get("wandb_name")
            run_name = output_cfg.get("run_name")
            if isinstance(wandb_name, str) and wandb_name:
                return wandb_name
            if isinstance(run_name, str) and run_name:
                return run_name

    return stats_dir.name


def _build_split_mode_rows(
    *,
    compression_compare: dict[str, Any],
    split_name: str,
    mode_name: str,
) -> list[dict[str, Any]]:
    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return []

    split_payload = results.get(split_name)
    if not isinstance(split_payload, dict):
        return []

    mode_payload = split_payload.get(mode_name)
    if not isinstance(mode_payload, dict):
        return []

    per_source = mode_payload.get("per_source")
    if not isinstance(per_source, list):
        return []

    order_map = _source_order_map(compression_compare)
    rows: list[dict[str, Any]] = []
    for item in per_source:
        if not isinstance(item, dict):
            continue

        source_name = str(item.get("source_name") or item.get("species") or "unknown")
        species_name = str(item.get("species") or source_name)
        arithmetic_bpb = _safe_float(item.get("arithmetic_bits_per_base"))
        theoretical_bpb = _safe_float(item.get("theoretical_bits_per_base"))
        compression_bases_per_second = _safe_float(item.get("compression_bases_per_second"))
        compression_bytes_per_second = _safe_float(item.get("compression_bytes_per_second"))
        paper_baseline_percent = _paper_baseline_percent(source_name=source_name, species_name=species_name)

        row = {
            "split": split_name,
            "mode": mode_name,
            "species": species_name,
            "source_name": source_name,
            "sample_bytes": item.get("sample_bytes"),
            "sample_bases": item.get("sample_bases"),
            "arithmetic_bits_per_base": arithmetic_bpb,
            "theoretical_bits_per_base": theoretical_bpb,
            "vs_2bit_percent": (arithmetic_bpb / 2.0 * 100.0) if arithmetic_bpb is not None else None,
            "paper_baseline_percent": paper_baseline_percent,
            "paper_baseline_bpb": (paper_baseline_percent / 100.0 * 2.0) if paper_baseline_percent is not None else None,
            "compression_bases_per_second": compression_bases_per_second,
            "compression_mbases_per_second": (
                compression_bases_per_second / 1_000_000.0 if compression_bases_per_second is not None else None
            ),
            "compression_bytes_per_second": compression_bytes_per_second,
            "compression_mbytes_per_second": (
                compression_bytes_per_second / 1_000_000.0 if compression_bytes_per_second is not None else None
            ),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            order_map.get(str(row["source_name"]), order_map.get(str(row["species"]), 10**9)),
            str(row["source_name"]),
        )
    )
    return rows


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "mode",
        "species",
        "source_name",
        "sample_bytes",
        "sample_bases",
        "arithmetic_bits_per_base",
        "theoretical_bits_per_base",
        "vs_2bit_percent",
        "paper_baseline_percent",
        "paper_baseline_bpb",
        "compression_bases_per_second",
        "compression_mbases_per_second",
        "compression_bytes_per_second",
        "compression_mbytes_per_second",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _plot_series(
    axis: Any,
    x_values: list[int],
    y_values: list[float],
    ylabel: str,
    title: str,
    color: str,
    *,
    label: str,
) -> None:
    finite_values = [value for value in y_values if not math.isnan(value)]
    if finite_values:
        axis.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=4.5, color=color, label=label)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25, linewidth=0.6)


def _plot_baseline(axis: Any, x_values: list[int], y_values: list[float], *, label: str, color: str) -> bool:
    finite_values = [value for value in y_values if not math.isnan(value)]
    if not finite_values:
        return False
    axis.plot(
        x_values,
        y_values,
        linestyle="--",
        linewidth=1.6,
        marker="s",
        markersize=4.0,
        color=color,
        alpha=0.9,
        label=label,
    )
    return True


def _write_plot_png(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    run_name: str,
    split_name: str,
    mode_name: str,
) -> None:
    plt = _load_matplotlib_pyplot()

    labels = [str(row["source_name"]) for row in rows]
    x_values = list(range(len(rows)))
    arithmetic_bpb = [
        float(value) if isinstance(value := row.get("arithmetic_bits_per_base"), (int, float)) else float("nan")
        for row in rows
    ]
    vs_2bit_percent = [
        float(value) if isinstance(value := row.get("vs_2bit_percent"), (int, float)) else float("nan")
        for row in rows
    ]
    paper_baseline_percent = [
        float(value) if isinstance(value := row.get("paper_baseline_percent"), (int, float)) else float("nan")
        for row in rows
    ]
    paper_baseline_bpb = [
        float(value) if isinstance(value := row.get("paper_baseline_bpb"), (int, float)) else float("nan")
        for row in rows
    ]
    speed_mbases = [
        float(value) if isinstance(value := row.get("compression_mbases_per_second"), (int, float)) else float("nan")
        for row in rows
    ]

    figure_width = max(12.0, min(28.0, len(rows) * 0.8))
    figure, axes = plt.subplots(3, 1, figsize=(figure_width, 13.0), sharex=True)
    _plot_series(
        axes[0],
        x_values,
        arithmetic_bpb,
        ylabel="Arithmetic BPB",
        title=f"{run_name} | {split_name} | {mode_name} | Compression Ratio (BPB)",
        color="#1f77b4",
        label="Model",
    )
    arithmetic_has_baseline = _plot_baseline(
        axes[0],
        x_values,
        paper_baseline_bpb,
        label="Paper baseline",
        color="#d62728",
    )
    _plot_series(
        axes[1],
        x_values,
        vs_2bit_percent,
        ylabel="% of 2-bit",
        title="Compression Ratio Relative to 2-bit Encoding",
        color="#ff7f0e",
        label="Model",
    )
    percent_has_baseline = _plot_baseline(
        axes[1],
        x_values,
        paper_baseline_percent,
        label="Paper baseline",
        color="#d62728",
    )
    _plot_series(
        axes[2],
        x_values,
        speed_mbases,
        ylabel="Speed (Mbases/s)",
        title="Compression Speed",
        color="#2ca02c",
        label="Model",
    )
    if arithmetic_has_baseline:
        axes[0].legend(loc="best")
    if percent_has_baseline:
        axes[1].legend(loc="best")

    axes[2].set_xlabel("DNA Source")
    axes[2].set_xticks(x_values)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=9)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_artifacts_for_compression_compare(
    compression_compare_path: Path,
    *,
    out_dir_name: str = "compression_curves",
) -> list[Path]:
    compression_compare = _read_json(compression_compare_path)
    stats_dir = compression_compare_path.parent
    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return []

    run_name = _resolve_run_name(stats_dir, compression_compare)
    output_dir = stats_dir / out_dir_name
    generated_paths: list[Path] = []

    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for mode_name in split_payload.keys():
            rows = _build_split_mode_rows(
                compression_compare=compression_compare,
                split_name=str(split_name),
                mode_name=str(mode_name),
            )
            if not rows:
                continue

            artifact_stem = _artifact_stem(str(split_name), str(mode_name))
            csv_path = output_dir / f"{artifact_stem}_compression_curve_data.csv"
            png_path = output_dir / f"{artifact_stem}_compression_curves.png"
            _write_rows_csv(csv_path, rows)
            _write_plot_png(
                path=png_path,
                rows=rows,
                run_name=run_name,
                split_name=str(split_name),
                mode_name=str(mode_name),
            )
            generated_paths.extend([csv_path, png_path])

    return generated_paths


def generate_curves_for_root(root_dir: Path, *, out_dir_name: str = "compression_curves") -> list[Path]:
    generated_paths: list[Path] = []
    for compression_compare_path in sorted(root_dir.rglob("compression_compare.json")):
        generated_paths.extend(
            generate_artifacts_for_compression_compare(
                compression_compare_path,
                out_dir_name=out_dir_name,
            )
        )
    return generated_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively generate per-source compression comparison plots from compression_compare.json files."
    )
    parser.add_argument("--root-dir", required=True, help="Root output directory to scan recursively.")
    parser.add_argument(
        "--out-dir-name",
        default="compression_curves",
        help="Artifact subdirectory name created inside each statistics directory.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    generated_paths = generate_curves_for_root(root_dir, out_dir_name=args.out_dir_name)
    if not generated_paths:
        print(f"[done] no compression_compare.json files found under {root_dir}")
        return

    print(f"[done] generated {len(generated_paths)} artifacts under {root_dir}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
