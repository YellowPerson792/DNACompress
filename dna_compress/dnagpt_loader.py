from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch

from .config import ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "DNAGPT"

DNAGPT_RESERVED_TOKENS = (
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + ["+", "-", "*", "/", "=", "&", "|", "!"]
    + ["M", "B"]
    + ["P"]
    + ["R", "I", "K", "L", "O", "Q", "S", "U", "V"]
    + ["W", "Y", "X", "Z"]
)


@dataclass(frozen=True)
class DNAGPTVariantSpec:
    variant: str
    kmer_size: int
    dynamic_kmer: bool
    max_len: int


DNAGPT_VARIANTS: dict[str, DNAGPTVariantSpec] = {
    "dna_gpt0.1b_h": DNAGPTVariantSpec(
        variant="dna_gpt0.1b_h",
        kmer_size=6,
        dynamic_kmer=False,
        max_len=4096,
    ),
    "dna_gpt0.1b_m": DNAGPTVariantSpec(
        variant="dna_gpt0.1b_m",
        kmer_size=6,
        dynamic_kmer=True,
        max_len=512,
    ),
    "dna_gpt3b_m": DNAGPTVariantSpec(
        variant="dna_gpt3b_m",
        kmer_size=6,
        dynamic_kmer=True,
        max_len=512,
    ),
}


def ensure_repo_on_path() -> None:
    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def get_variant_spec(variant: str) -> DNAGPTVariantSpec:
    try:
        return DNAGPT_VARIANTS[variant]
    except KeyError as error:
        supported = ", ".join(sorted(DNAGPT_VARIANTS))
        raise ValueError(f"Unsupported DNAGPT variant '{variant}'. Expected one of: {supported}.") from error


def default_pretrained_weight_path(variant: str) -> Path:
    return REPO_ROOT / "checkpoints" / f"{variant}.pth"


def build_dnagpt_tokenizer(variant: str):
    ensure_repo_on_path()
    from dna_gpt.tokenizer import KmerTokenizer

    spec = get_variant_spec(variant)
    return KmerTokenizer(
        spec.kmer_size,
        DNAGPT_RESERVED_TOKENS,
        dynamic_kmer=spec.dynamic_kmer,
    )


def build_dnagpt_model(variant: str):
    ensure_repo_on_path()
    from dna_gpt.model import DNAGPT

    tokenizer = build_dnagpt_tokenizer(variant)
    model = DNAGPT.from_name(variant, len(tokenizer))
    return model, tokenizer, get_variant_spec(variant)


def build_dnagpt_components(model_config: ModelConfig):
    return build_dnagpt_model(model_config.variant)


def _extract_model_state(state: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(state, dict):
        raise ValueError("DNAGPT checkpoint must deserialize to a dict-like object.")

    metadata: dict[str, Any] = {}
    if "model_state" in state:
        metadata = {
            key: value
            for key, value in state.items()
            if key not in {"model_state", "optimizer_state"}
        }
        return state["model_state"], metadata
    if "model" in state:
        metadata = {key: value for key, value in state.items() if key != "model"}
        return state["model"], metadata

    tensor_values = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if tensor_values and len(tensor_values) == len(state):
        return state, metadata

    raise ValueError("Unable to locate model weights in DNAGPT checkpoint.")


def load_dnagpt_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"DNAGPT checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location)
    model_state, metadata = _extract_model_state(state)
    model.load_state_dict(model_state, strict=strict)
    return metadata


def load_dnagpt_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"DNAGPT checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location)
    model_state, metadata = _extract_model_state(state)
    if not isinstance(state, dict):
        raise ValueError("DNAGPT checkpoint must deserialize to a dict-like object.")
    return model_state, metadata, state
