from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .compression import arithmetic_encode, baseline_sizes
from .config import ExperimentConfig, save_experiment_config
from .data import (
    RandomWindowDataset,
    SequentialWindowDataset,
    load_splits,
)
from .megabyte_loader import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: AdamW, step: int, best_val_bpb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_bpb": best_val_bpb,
        },
        path,
    )


def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    pad_id: int,
) -> dict[str, float]:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            valid_tokens = int((ids != pad_id).sum().item())
            with autocast_context(device, dtype_name):
                output = model(ids, return_loss=True)
            total_nats += float(output.loss.item()) * valid_tokens
            total_tokens += valid_tokens

    average_nats = total_nats / total_tokens
    return {
        "loss_nats": average_nats,
        "bits_per_byte": average_nats / math.log(2),
        "tokens": total_tokens,
        "elapsed_seconds": time.time() - start_time,
    }


def evaluate_compression(
    model: torch.nn.Module,
    payload: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
) -> dict[str, float | int]:
    symbols = list(payload) + [eos_id]
    windows = torch.full((len(symbols), seq_length), pad_id, dtype=torch.long)
    for index in range(len(symbols)):
        start = max(0, index - seq_length + 1)
        history = symbols[start : index + 1]
        windows[index, -len(history) :] = torch.tensor(history, dtype=torch.long)

    probability_rows: list[np.ndarray] = []
    total_bits = 0.0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(symbols), batch_size):
            batch = windows[start : start + batch_size].to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                output = model(batch, return_loss=False)
                log_probs = torch.log_softmax(output.lm_logits[:, -1, :], dim=-1).float().cpu()
            targets = symbols[start : start + log_probs.shape[0]]
            for row, target in zip(log_probs, targets):
                total_bits += float(-row[target].item() / math.log(2))
                probability_rows.append(row.exp().numpy())

    compressed = arithmetic_encode(symbols, probability_rows)
    baselines = baseline_sizes(payload)
    return {
        "sample_bytes": len(payload),
        "sample_symbols_with_eos": len(symbols),
        "theoretical_bits": total_bits,
        "theoretical_bits_per_byte": total_bits / len(payload),
        "arithmetic_coded_bytes": len(compressed),
        "arithmetic_bits_per_byte": (len(compressed) * 8) / len(payload),
        **baselines,
    }


def evaluate_compression_per_source(
    model: torch.nn.Module,
    test_sources: list[bytes],
    species_names: list[str],
    requested_bytes: int,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
) -> dict[str, object]:
    per_source: list[dict[str, object]] = []
    total_sample_bytes = 0
    total_theoretical_bits = 0.0
    total_arithmetic_bytes = 0

    for source, species_name in zip(test_sources, species_names):
        payload = source[:requested_bytes] if len(source) >= requested_bytes else source
        metrics = evaluate_compression(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
        )
        per_source.append({"species": species_name, **metrics})
        total_sample_bytes += int(metrics["sample_bytes"])
        total_theoretical_bits += float(metrics["theoretical_bits"])
        total_arithmetic_bytes += int(metrics["arithmetic_coded_bytes"])

    aggregate = {
        "source_count": len(per_source),
        "total_sample_bytes": total_sample_bytes,
        "total_theoretical_bits": total_theoretical_bits,
        "total_theoretical_bits_per_byte": total_theoretical_bits / max(total_sample_bytes, 1),
        "total_arithmetic_coded_bytes": total_arithmetic_bytes,
        "total_arithmetic_bits_per_byte": (total_arithmetic_bytes * 8) / max(total_sample_bytes, 1),
    }
    return {
        "aggregate": aggregate,
        "per_source": per_source,
    }


def run_experiment(config: ExperimentConfig, mode: str = "all") -> dict[str, object]:
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "resolved_config.json")

    splits = load_splits(config.data)
    train_dataset = RandomWindowDataset(
        sources=splits.train_sources,
        seq_length=config.model.seq_length,
        samples_per_epoch=config.data.train_samples_per_epoch,
        seed=config.train.seed,
    )
    val_dataset = SequentialWindowDataset(
        sources=splits.val_sources,
        seq_length=config.model.seq_length,
        pad_id=config.model.pad_id,
    )
    test_dataset = SequentialWindowDataset(
        sources=splits.test_sources,
        seq_length=config.model.seq_length,
        pad_id=config.model.pad_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and config.train.dtype == "float16")

    run_summary: dict[str, object] = {
        "device": str(device),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "dataset": splits.summary,
    }

    best_val_bpb = float("inf")
    global_step = 0

    if mode in {"train", "all"}:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for epoch in range(config.train.epochs):
            for batch in train_loader:
                global_step += 1
                ids = batch["input_ids"].to(device, non_blocking=True)
                with autocast_context(device, config.train.dtype):
                    output = model(ids, return_loss=True)
                    loss = output.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if global_step % config.train.log_interval == 0:
                    tokens_per_second = (
                        config.train.batch_size * config.model.seq_length * config.train.log_interval
                    ) / max(time.time() - started, 1e-6)
                    print(
                        f"[train] epoch={epoch + 1} step={global_step} "
                        f"loss={loss.item():.4f} bpb={loss.item() / math.log(2):.4f} "
                        f"tokens/s={tokens_per_second:.1f}"
                    )
                    started = time.time()

                if global_step % config.train.eval_interval == 0:
                    val_metrics = evaluate_loss(
                        model,
                        val_loader,
                        device,
                        config.train.dtype,
                        config.model.pad_id,
                    )
                    print(
                        f"[eval] step={global_step} val_loss={val_metrics['loss_nats']:.4f} "
                        f"val_bpb={val_metrics['bits_per_byte']:.4f}"
                    )
                    if val_metrics["bits_per_byte"] < best_val_bpb:
                        best_val_bpb = float(val_metrics["bits_per_byte"])
                        save_checkpoint(output_dir / "best.pt", model, optimizer, global_step, best_val_bpb)
                    model.train()

        if best_val_bpb == float("inf"):
            val_metrics = evaluate_loss(
                model,
                val_loader,
                device,
                config.train.dtype,
                config.model.pad_id,
            )
            best_val_bpb = float(val_metrics["bits_per_byte"])
            save_checkpoint(output_dir / "best.pt", model, optimizer, global_step, best_val_bpb)

        save_checkpoint(output_dir / "last.pt", model, optimizer, global_step, best_val_bpb)
        run_summary["best_val_bits_per_byte"] = best_val_bpb

    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    if mode in {"eval", "all", "compress"}:
        val_metrics = evaluate_loss(
            model,
            val_loader,
            device,
            config.train.dtype,
            config.model.pad_id,
        )
        test_metrics = evaluate_loss(
            model,
            test_loader,
            device,
            config.train.dtype,
            config.model.pad_id,
        )
        species_names = [str(item["species"]) for item in splits.summary["species"]]
        compression_metrics = evaluate_compression_per_source(
            model=model,
            test_sources=splits.test_sources,
            species_names=species_names,
            requested_bytes=config.data.compression_sample_bytes,
            seq_length=config.model.seq_length,
            pad_id=config.model.pad_id,
            eos_id=config.model.eos_id,
            device=device,
            dtype_name=config.train.dtype,
            batch_size=config.train.eval_batch_size,
        )

        run_summary["validation"] = val_metrics
        run_summary["test"] = test_metrics
        run_summary["compression"] = compression_metrics

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_summary
