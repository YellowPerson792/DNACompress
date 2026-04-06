from __future__ import annotations

"""Parse training_metrics.jsonl and upload metrics to Weights & Biases.

Example:
    python scripts/upload_training_log_to_wandb.py \
      --run-dir outputs/dnagpt_0p1bm_all_species_nonoverlap \
      --project dna-compress \
      --name dnagpt-0p1bm-all-species
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists() or not log_path.is_file():
        raise FileNotFoundError(f"training log not found: {log_path}")

    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {log_path}:{line_number}") from error
            if not isinstance(event, dict):
                continue
            events.append(event)
    return events


def _event_to_wandb_row(event: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    event_type = event.get("event")
    step = event.get("step")
    epoch = event.get("epoch")

    row: dict[str, Any] = {}
    if isinstance(step, int):
        row["step"] = step
    if isinstance(epoch, int):
        row["epoch"] = epoch

    if event_type == "train":
        if "loss_nats_per_token" in event:
            row["train/loss"] = event["loss_nats_per_token"]
        if "bits_per_base" in event:
            row["train/bpb"] = event["bits_per_base"]
        if "learning_rate" in event:
            row["train/lr"] = event["learning_rate"]
        if "tokens_per_second" in event:
            row["train/tokens_per_second"] = event["tokens_per_second"]
    elif event_type == "eval":
        split = str(event.get("split", "eval"))
        if "loss_nats_per_token" in event:
            row[f"{split}/loss"] = event["loss_nats_per_token"]
            if split == "val":
                row["eval/loss"] = event["loss_nats_per_token"]
        if "bits_per_base" in event:
            row[f"{split}/bpb"] = event["bits_per_base"]
            if split == "val":
                row["eval/bpb"] = event["bits_per_base"]

    return (step if isinstance(step, int) else None), row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload training JSONL metrics to Weights & Biases.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing training_metrics.jsonl.")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Explicit path to JSONL log file. Defaults to <run-dir>/training_metrics.jsonl.",
    )
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", default=None, help="W&B entity/team.")
    parser.add_argument("--name", default=None, help="W&B run name. Defaults to run-dir folder name.")
    parser.add_argument("--mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--resolved-config",
        default=None,
        help="Optional resolved config path. Defaults to <run-dir>/resolved_config.json when run-dir is set.",
    )
    parser.add_argument(
        "--job-type",
        default="train",
        help="W&B job type for this upload run.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.log_file:
        log_path = Path(args.log_file)
    elif run_dir is not None:
        log_path = run_dir / "training_metrics.jsonl"
    else:
        raise ValueError("Either --log-file or --run-dir must be provided.")

    resolved_config_path: Path | None = None
    if args.resolved_config:
        resolved_config_path = Path(args.resolved_config)
    elif run_dir is not None:
        resolved_config_path = run_dir / "resolved_config.json"

    events = _load_events(log_path)
    if not events:
        raise ValueError(f"No events found in {log_path}")

    try:
        import wandb
    except ImportError as error:
        raise ImportError("wandb is required. Install with: pip install wandb") from error

    run_name = args.name
    if run_name is None and run_dir is not None:
        run_name = run_dir.name

    run_config = _read_json(resolved_config_path) if resolved_config_path is not None else None

    wandb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        mode=args.mode,
        job_type=args.job_type,
        config=run_config,
        reinit=True,
    )

    best_val_bpb: float | None = None
    for event in events:
        step, row = _event_to_wandb_row(event)
        if not row:
            continue
        if "val/bpb" in row and isinstance(row["val/bpb"], (int, float)):
            value = float(row["val/bpb"])
            best_val_bpb = value if best_val_bpb is None else min(best_val_bpb, value)

        if step is not None:
            wandb.log(row, step=step)
        else:
            wandb.log(row)

    if best_val_bpb is not None:
        wandb_run.summary["best_val/bpb"] = best_val_bpb
    wandb_run.summary["uploaded_events"] = len(events)
    wandb_run.summary["training_log_file"] = str(log_path)

    wandb.finish()
    print(f"Uploaded {len(events)} events to W&B from {log_path}")


if __name__ == "__main__":
    main()
