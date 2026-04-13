from __future__ import annotations

"""Parse training_metrics.jsonl and upload metrics to Weights & Biases.

Example:
    python scripts/upload_training_log_to_wandb.py \
      --run-dir outputs/dna_megabyte_large_b128_ensembl_all \
      --project dna-compress 
      
      --name dna_dnagpt_0p1bm_all_scratch
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


def _extract_run_id_from_wandb_file(path: Path) -> str | None:
    stem = path.stem
    if not stem.startswith("run-"):
        return None
    run_id = stem[len("run-") :].strip()
    return run_id or None


def _extract_run_id_from_run_dir(path: Path) -> str | None:
    wandb_files = sorted(path.glob("run-*.wandb"))
    for wandb_file in wandb_files:
        run_id = _extract_run_id_from_wandb_file(wandb_file)
        if run_id is not None:
            return run_id
    return None


def _resolve_local_wandb_run_dir(wandb_dir: Path) -> Path | None:
    latest_run = wandb_dir / "latest-run"
    if latest_run.exists():
        try:
            latest_candidate = latest_run.resolve()
        except OSError:
            latest_candidate = None
        if latest_candidate is not None and latest_candidate.is_dir() and _extract_run_id_from_run_dir(latest_candidate):
            return latest_candidate

    candidates: list[Path] = []
    for candidate in wandb_dir.glob("run-*"):
        if candidate.is_dir() and _extract_run_id_from_run_dir(candidate):
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def _discover_local_wandb_run(run_dir: Path | None) -> tuple[Path, str] | None:
    if run_dir is None:
        return None
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists() or not wandb_dir.is_dir():
        return None

    local_run_dir = _resolve_local_wandb_run_dir(wandb_dir)
    if local_run_dir is None:
        return None

    run_id = _extract_run_id_from_run_dir(local_run_dir)
    if run_id is None:
        return None
    return local_run_dir, run_id


def _fetch_remote_uploaded_event_count(wandb_module: Any, entity: str, project: str, run_id: str) -> int | None:
    try:
        api = wandb_module.Api()
        remote_run = api.run(f"{entity}/{project}/{run_id}")
    except Exception:
        return None

    summary = getattr(remote_run, "summary", None)
    if not isinstance(summary, dict):
        return None

    uploaded_events = summary.get("uploaded_events")
    if not isinstance(uploaded_events, int):
        return None
    if uploaded_events < 0:
        return None
    return uploaded_events


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


def upload_training_log(args: argparse.Namespace, wandb_module: Any) -> str:
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

    run_name = args.name
    if run_name is None and run_dir is not None:
        run_name = run_dir.name

    run_config = _read_json(resolved_config_path) if resolved_config_path is not None else None
    local_wandb_run = _discover_local_wandb_run(run_dir)

    init_kwargs: dict[str, Any] = {
        "project": args.project,
        "entity": args.entity,
        "name": run_name,
        "mode": args.mode,
        "job_type": args.job_type,
        "config": run_config,
        "reinit": True,
    }
    resumed_run_id: str | None = None
    if local_wandb_run is not None:
        _, resumed_run_id = local_wandb_run
        init_kwargs["id"] = resumed_run_id
        init_kwargs["resume"] = "allow"

    wandb_run = wandb_module.init(**init_kwargs)

    remote_uploaded_events: int | None = None
    upload_start_index = 0
    if resumed_run_id is not None:
        resolved_entity = getattr(wandb_run, "entity", None) or args.entity
        resolved_project = getattr(wandb_run, "project", None) or args.project
        if isinstance(resolved_entity, str) and resolved_entity:
            remote_uploaded_events = _fetch_remote_uploaded_event_count(
                wandb_module,
                entity=resolved_entity,
                project=resolved_project,
                run_id=resumed_run_id,
            )
            if remote_uploaded_events is not None:
                upload_start_index = min(remote_uploaded_events, len(events))

    best_val_bpb: float | None = None
    for event in events[upload_start_index:]:
        step, row = _event_to_wandb_row(event)
        if not row:
            continue
        if "val/bpb" in row and isinstance(row["val/bpb"], (int, float)):
            value = float(row["val/bpb"])
            best_val_bpb = value if best_val_bpb is None else min(best_val_bpb, value)

        if step is not None:
            wandb_module.log(row, step=step)
        else:
            wandb_module.log(row)

    if best_val_bpb is None:
        for event in events[:upload_start_index]:
            _, row = _event_to_wandb_row(event)
            if "val/bpb" in row and isinstance(row["val/bpb"], (int, float)):
                value = float(row["val/bpb"])
                best_val_bpb = value if best_val_bpb is None else min(best_val_bpb, value)

    if best_val_bpb is not None:
        wandb_run.summary["best_val/bpb"] = best_val_bpb
    wandb_run.summary["uploaded_events"] = len(events)
    wandb_run.summary["training_log_file"] = str(log_path)
    if resumed_run_id is not None:
        wandb_run.summary["upload_resumed_run_id"] = resumed_run_id
        wandb_run.summary["upload_resumed_from_event"] = upload_start_index
    if remote_uploaded_events is not None:
        wandb_run.summary["upload_remote_uploaded_events_before_resume"] = remote_uploaded_events

    wandb_module.finish()
    return (
        f"Uploaded {len(events) - upload_start_index} events to W&B from {log_path}"
        f" (local events={len(events)}, resumed_from_event={upload_start_index})"
    )


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

    try:
        import wandb
    except ImportError as error:
        raise ImportError("wandb is required. Install with: pip install wandb") from error
    print(upload_training_log(args, wandb))


if __name__ == "__main__":
    main()
