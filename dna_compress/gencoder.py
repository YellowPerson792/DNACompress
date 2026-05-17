from __future__ import annotations

import csv
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_TO_INT = {
    ord("A"): 1,
    ord("C"): 2,
    ord("G"): 3,
    ord("T"): 4,
}
INT_TO_BASE = np.asarray([ord("N"), ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)


@dataclass
class GenCoderRunConfig:
    name: str
    species: list[str]
    seq_length: int
    bottleneck_dim: int | str
    epochs: int = 50
    batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 1e-3
    validation_ratio: float = 0.0
    max_bytes_per_source: int | None = None
    save_artifacts: bool = True


@dataclass
class GenCoderConfig:
    dataset_dir: str = "datasets/DNACorpus"
    output_dir: str = "outputs/dna_gencoder_dnacorpus"
    device: str = "auto"
    seed: int = 42
    runs: list[GenCoderRunConfig] = field(default_factory=list)


class GenCoderAutoEncoder(nn.Module):
    def __init__(self, seq_length: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.seq_length = int(seq_length)
        self.bottleneck_dim = int(bottleneck_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        flattened_dim = 64 * self.seq_length
        self.encoder_dense = nn.Linear(flattened_dim, self.bottleneck_dim)
        self.decoder_dense = nn.Linear(self.bottleneck_dim, flattened_dim)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding="same"),
            nn.Sigmoid(),
        )

    def encode(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.ndim != 2:
            raise ValueError(f"expected input with shape (batch, L), got {tuple(normalized.shape)}")
        hidden = self.encoder_conv(normalized.unsqueeze(1))
        return self.encoder_dense(hidden.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_dense(latent)
        hidden = hidden.view(latent.shape[0], 64, self.seq_length)
        return self.decoder_conv(hidden).squeeze(1)

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(normalized))


class GenCoderChunkDataset(Dataset):
    def __init__(self, chunks: np.ndarray) -> None:
        if chunks.ndim != 2:
            raise ValueError("chunks must be a 2D uint8 array")
        self.chunks = chunks.astype(np.uint8, copy=False)

    def __len__(self) -> int:
        return int(self.chunks.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.as_tensor(self.chunks[index].astype(np.float32) / 4.0)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_bottleneck_dim(value: int | str, seq_length: int) -> int:
    if isinstance(value, int):
        return value
    token = value.strip().upper().replace(" ", "")
    if token.startswith("L/"):
        divisor = int(token[2:])
        if divisor <= 0:
            raise ValueError("bottleneck divisor must be > 0")
        return max(1, seq_length // divisor)
    return int(value)


def load_gencoder_config(path: str | Path) -> GenCoderConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [GenCoderRunConfig(**item) for item in raw.get("runs", [])]
    return GenCoderConfig(
        dataset_dir=raw.get("dataset_dir", "datasets/DNACorpus"),
        output_dir=raw.get("output_dir", "outputs/dna_gencoder_dnacorpus"),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 42)),
        runs=runs,
    )


def load_sequence_payload(path: Path, max_bytes: int | None = None) -> tuple[bytes, dict[str, int]]:
    raw = path.read_bytes()
    if max_bytes is not None and max_bytes > 0:
        raw = raw[:max_bytes]

    cleaned = bytearray()
    removed_non_acgt = 0
    skipped_header_lines = 0
    for raw_line in raw.splitlines():
        line = raw_line.strip().upper()
        if not line:
            continue
        if line.startswith(b">"):
            skipped_header_lines += 1
            continue
        for byte_value in line:
            mapped = BASE_TO_INT.get(byte_value)
            if mapped is None:
                removed_non_acgt += 1
                continue
            cleaned.append(byte_value)

    return bytes(cleaned), {
        "raw_bytes": len(raw),
        "sequence_bytes": len(cleaned),
        "removed_non_acgt": removed_non_acgt,
        "skipped_header_lines": skipped_header_lines,
    }


def encode_sequence(sequence: bytes) -> np.ndarray:
    encoded = np.frombuffer(sequence, dtype=np.uint8)
    values = np.zeros(encoded.shape[0], dtype=np.uint8)
    for base_byte, integer_value in BASE_TO_INT.items():
        values[encoded == base_byte] = integer_value
    if np.any(values == 0):
        raise ValueError("sequence contains non-ACGT bytes; call load_sequence_payload first")
    return values


def decode_sequence(values: np.ndarray, original_length: int | None = None) -> bytes:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if original_length is not None:
        flat = flat[:original_length]
    if np.any((flat < 1) | (flat > 4)):
        raise ValueError("decoded integer sequence contains values outside 1..4")
    return INT_TO_BASE[flat].astype(np.uint8).tobytes()


def pad_and_chunk(encoded: np.ndarray, seq_length: int, pad_value: int = 1) -> tuple[np.ndarray, int]:
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    encoded = np.asarray(encoded, dtype=np.uint8)
    padding = (-encoded.shape[0]) % seq_length
    if padding:
        encoded = np.concatenate([encoded, np.full(padding, pad_value, dtype=np.uint8)])
    return encoded.reshape(-1, seq_length), int(padding)


def chunks_to_normalized_tensor(chunks: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(chunks.astype(np.float32) / 4.0, device=device)


def quantize_reconstruction(normalized: np.ndarray) -> np.ndarray:
    values = np.rint(np.asarray(normalized, dtype=np.float32) * 4.0)
    return np.clip(values, 1, 4).astype(np.int16)


def serialize_csr(matrix: sparse.csr_matrix) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )
    return buffer.getvalue()


def deserialize_csr(payload: bytes) -> sparse.csr_matrix:
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        shape = tuple(int(item) for item in loaded["shape"].tolist())
        return sparse.csr_matrix((loaded["data"], loaded["indices"], loaded["indptr"]), shape=shape)


def fpzip_compress_array(array: np.ndarray) -> bytes:
    try:
        import fpzip
    except ImportError as error:
        raise ImportError(
            "GenCoder latent compression requires fpzip. Install with: "
            "D:\\MLLMs\\.venv\\Scripts\\python.exe -m pip install fpzip==1.2.5"
        ) from error
    return fpzip.compress(np.asarray(array, dtype=np.float32), precision=0)


def fpzip_decompress_array(payload: bytes, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        import fpzip
    except ImportError as error:
        raise ImportError("GenCoder latent decompression requires fpzip.") from error
    decoded = np.asarray(fpzip.decompress(payload), dtype=np.float32)
    return decoded.reshape(-1)[: int(np.prod(expected_shape))].reshape(expected_shape)


def load_run_sources(
    dataset_dir: Path,
    species: list[str],
    max_bytes_per_source: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    entries: list[dict[str, Any]] = []
    encoded_parts: list[np.ndarray] = []
    for species_name in species:
        path = dataset_dir / species_name
        if not path.exists():
            raise FileNotFoundError(f"DNACorpus source not found: {path}")
        payload, stats = load_sequence_payload(path, max_bytes=max_bytes_per_source)
        if not payload:
            raise ValueError(f"{species_name} has no ACGT bases after preprocessing")
        encoded = encode_sequence(payload)
        entries.append(
            {
                "species": species_name,
                "path": str(path),
                **stats,
                "side_info_bytes": 0,
                "header_huffman_bytes": 0,
            }
        )
        encoded_parts.append(encoded)
    return entries, np.concatenate(encoded_parts)


def train_model(
    model: GenCoderAutoEncoder,
    chunks: np.ndarray,
    run_config: GenCoderRunConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    total_chunks = int(chunks.shape[0])
    val_count = int(total_chunks * run_config.validation_ratio)
    train_count = max(1, total_chunks - val_count)
    train_chunks = chunks[:train_count]
    val_chunks = chunks[train_count:] if val_count > 0 else np.empty((0, chunks.shape[1]), dtype=np.uint8)

    train_loader = DataLoader(
        GenCoderChunkDataset(train_chunks),
        batch_size=run_config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        GenCoderChunkDataset(val_chunks),
        batch_size=run_config.eval_batch_size,
        shuffle=False,
        num_workers=0,
    ) if val_count > 0 else None

    optimizer = torch.optim.NAdam(model.parameters(), lr=run_config.learning_rate)
    criterion = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, run_config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            item_count = int(batch.shape[0])
            train_loss_sum += float(loss.item()) * item_count
            train_items += item_count

        event: dict[str, float | int] = {
            "epoch": epoch,
            "train_mse": train_loss_sum / max(train_items, 1),
        }
        if val_loader is not None:
            event["val_mse"] = evaluate_mse(model, val_loader, device)
        history.append(event)
        print(json.dumps({"event": "gencoder_epoch", **event}, ensure_ascii=False), flush=True)

    return {
        "epochs": run_config.epochs,
        "train_chunks": int(train_chunks.shape[0]),
        "val_chunks": int(val_chunks.shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def evaluate_mse(model: GenCoderAutoEncoder, loader: DataLoader, device: torch.device) -> float:
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    total_values = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            total_loss += float(criterion(reconstructed, batch).item())
            total_values += int(batch.numel())
    return total_loss / max(total_values, 1)


def _forward_latent_and_reconstruction(
    model: GenCoderAutoEncoder,
    chunks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    latents: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    loader = DataLoader(GenCoderChunkDataset(chunks), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            latent = model.encode(batch)
            reconstructed = model.decode(latent)
            latents.append(latent.detach().cpu().numpy().astype(np.float32))
            reconstructions.append(reconstructed.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0), np.concatenate(reconstructions, axis=0)


def _decode_latents_to_int_chunks(
    model: GenCoderAutoEncoder,
    latents: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    decoded_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], batch_size):
            latent = torch.as_tensor(latents[start : start + batch_size], device=device)
            reconstructed = model.decode(latent).detach().cpu().numpy()
            decoded_chunks.append(quantize_reconstruction(reconstructed))
    return np.concatenate(decoded_chunks, axis=0)


def compress_source(
    model: GenCoderAutoEncoder,
    source_path: Path,
    run_config: GenCoderRunConfig,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload, stats = load_sequence_payload(source_path, max_bytes=run_config.max_bytes_per_source)
    encoded = encode_sequence(payload)
    chunks, padding = pad_and_chunk(encoded, run_config.seq_length)
    latents, reconstructed = _forward_latent_and_reconstruction(
        model=model,
        chunks=chunks,
        batch_size=run_config.eval_batch_size,
        device=device,
    )
    latent_blob = fpzip_compress_array(latents)
    # Compute the residual against the exact latent stream that will be stored.
    # This avoids one-off quantization drift between "compress" and "decompress"
    # decoder passes, especially on CUDA with different batch shapes.
    stored_latents = fpzip_decompress_array(latent_blob, latents.shape)
    reconstructed_int = _decode_latents_to_int_chunks(
        model=model,
        latents=stored_latents,
        batch_size=run_config.eval_batch_size,
        device=device,
    )
    residual = chunks.astype(np.int16) - reconstructed_int
    residual_csr = sparse.csr_matrix(residual)
    residual_blob = serialize_csr(residual_csr)
    compressed_bytes = len(residual_blob) + len(latent_blob)
    compression_seconds = time.perf_counter() - started

    decoded, decompression_seconds = decompress_payload(
        model=model,
        latent_blob=latent_blob,
        residual_blob=residual_blob,
        latent_shape=latents.shape,
        original_length=len(encoded),
        device=device,
        batch_size=run_config.eval_batch_size,
    )
    exact = decoded == payload
    artifact_dir = None
    if output_dir is not None and run_config.save_artifacts:
        artifact_dir = output_dir / "artifacts" / run_config.name / source_path.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "latent.fpzip").write_bytes(latent_blob)
        (artifact_dir / "residual_csr.npz").write_bytes(residual_blob)
        (artifact_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "latent_shape": list(latents.shape),
                    "residual_shape": list(residual.shape),
                    "original_length": len(encoded),
                    "padding_bases": padding,
                    "seq_length": run_config.seq_length,
                    "bottleneck_dim": model.bottleneck_dim,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    sample_bases = len(encoded)
    return {
        "species": source_path.name,
        **stats,
        "sample_bases": sample_bases,
        "chunk_count": int(chunks.shape[0]),
        "padding_bases": padding,
        "latent_shape": list(latents.shape),
        "residual_nonzeros": int(residual_csr.nnz),
        "residual_density": float(residual_csr.nnz / max(residual.size, 1)),
        "csr_residual_bytes": len(residual_blob),
        "fpzip_latent_bytes": len(latent_blob),
        "paper_compressed_bytes": compressed_bytes,
        "compression_ratio": compressed_bytes / max(sample_bases, 1),
        "bits_per_base": (compressed_bytes * 8) / max(sample_bases, 1),
        "side_info_bytes": 0,
        "header_huffman_bytes": 0,
        "compression_seconds": compression_seconds,
        "decompression_seconds": decompression_seconds,
        "exact_reconstruction": bool(exact),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
    }


def decompress_payload(
    model: GenCoderAutoEncoder,
    latent_blob: bytes,
    residual_blob: bytes,
    latent_shape: tuple[int, ...],
    original_length: int,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[bytes, float]:
    started = time.perf_counter()
    latents = fpzip_decompress_array(latent_blob, latent_shape)
    residual = deserialize_csr(residual_blob).toarray().astype(np.int16)
    reconstructed_all = _decode_latents_to_int_chunks(
        model=model,
        latents=latents,
        batch_size=batch_size,
        device=device,
    )
    recovered = reconstructed_all + residual
    sequence = decode_sequence(recovered, original_length=original_length)
    return sequence, time.perf_counter() - started


def load_checkpoint(path: Path, device: torch.device) -> tuple[GenCoderAutoEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    meta = dict(payload["metadata"])
    model = GenCoderAutoEncoder(
        seq_length=int(meta["seq_length"]),
        bottleneck_dim=int(meta["bottleneck_dim"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, meta


def save_checkpoint(path: Path, model: GenCoderAutoEncoder, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def summarize_per_source(per_source: list[dict[str, Any]], checkpoint_bytes: int) -> dict[str, Any]:
    total_bases = sum(int(row["sample_bases"]) for row in per_source)
    paper_bytes = sum(int(row["paper_compressed_bytes"]) for row in per_source)
    summary = {
        "source_count": len(per_source),
        "total_sample_bases": total_bases,
        "total_paper_compressed_bytes": paper_bytes,
        "total_compression_ratio": paper_bytes / max(total_bases, 1),
        "total_bits_per_base": (paper_bytes * 8) / max(total_bases, 1),
        "checkpoint_bytes": checkpoint_bytes,
        "size_including_model_bytes": paper_bytes + checkpoint_bytes,
        "size_including_model_ratio": (paper_bytes + checkpoint_bytes) / max(total_bases, 1),
        "all_exact_reconstruction": all(bool(row["exact_reconstruction"]) for row in per_source),
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_gencoder_config(config: GenCoderConfig, mode: str, run_name: str | None = None) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_runs = [run for run in config.runs if run_name is None or run.name == run_name]
    if not selected_runs:
        raise ValueError(f"No GenCoder run matched {run_name!r}")

    results: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "runs": {},
    }
    for run_config in selected_runs:
        bottleneck_dim = resolve_bottleneck_dim(run_config.bottleneck_dim, run_config.seq_length)
        run_dir = output_dir / run_config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best.pt"
        metrics_path = run_dir / "metrics.json"
        model: GenCoderAutoEncoder
        train_metrics: dict[str, Any] | None = None

        if mode in {"train", "all"}:
            print(f"[gencoder] loading training sources for {run_config.name}", flush=True)
            source_entries, encoded = load_run_sources(dataset_dir, run_config.species, run_config.max_bytes_per_source)
            chunks, padding = pad_and_chunk(encoded, run_config.seq_length)
            model = GenCoderAutoEncoder(run_config.seq_length, bottleneck_dim)
            train_metrics = train_model(model, chunks, run_config, device)
            save_checkpoint(
                checkpoint_path,
                model,
                {
                    "run_name": run_config.name,
                    "seq_length": run_config.seq_length,
                    "bottleneck_dim": bottleneck_dim,
                    "species": run_config.species,
                    "padding_bases": padding,
                    "source_entries": source_entries,
                },
            )
        else:
            model, _ = load_checkpoint(checkpoint_path, device)

        if metrics_path.exists() and mode == "decompress-check":
            run_result = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            run_result = {}
        run_result.update({
            "seq_length": run_config.seq_length,
            "bottleneck_dim": bottleneck_dim,
            "checkpoint_path": str(checkpoint_path),
        })
        if train_metrics is not None:
            run_result["training"] = train_metrics

        if mode in {"compress", "all", "decompress-check"}:
            if mode == "decompress-check":
                run_result["decompress_check"] = run_saved_artifact_checks(model, run_config, run_dir, device)
            else:
                per_source = []
                for species_name in run_config.species:
                    print(f"[gencoder] compressing {run_config.name}/{species_name}", flush=True)
                    row = compress_source(
                        model=model,
                        source_path=dataset_dir / species_name,
                        run_config=run_config,
                        device=device,
                        output_dir=run_dir,
                    )
                    per_source.append(row)
                checkpoint_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
                aggregate = summarize_per_source(per_source, checkpoint_bytes=checkpoint_bytes)
                run_result["compression"] = {"aggregate": aggregate, "per_source": per_source}
                write_csv(run_dir / "compression_per_source.csv", per_source)

        metrics_path.write_text(json.dumps(run_result, indent=2, ensure_ascii=False), encoding="utf-8")
        results["runs"][run_config.name] = run_result

    (output_dir / "gencoder_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def run_saved_artifact_checks(
    model: GenCoderAutoEncoder,
    run_config: GenCoderRunConfig,
    run_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    artifact_root = run_dir / "artifacts" / run_config.name
    if not artifact_root.exists():
        raise FileNotFoundError(f"No saved GenCoder artifacts found at {artifact_root}")
    for source_dir in sorted(item for item in artifact_root.iterdir() if item.is_dir()):
        metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        decoded, seconds = decompress_payload(
            model=model,
            latent_blob=(source_dir / "latent.fpzip").read_bytes(),
            residual_blob=(source_dir / "residual_csr.npz").read_bytes(),
            latent_shape=tuple(int(item) for item in metadata["latent_shape"]),
            original_length=int(metadata["original_length"]),
            device=device,
            batch_size=run_config.eval_batch_size,
        )
        original_payload, _ = load_sequence_payload(
            Path(metadata["source"]),
            max_bytes=run_config.max_bytes_per_source,
        )
        checks.append(
            {
                "source": source_dir.name,
                "exact_reconstruction": decoded == original_payload,
                "decompression_seconds": seconds,
            }
        )
    return checks
