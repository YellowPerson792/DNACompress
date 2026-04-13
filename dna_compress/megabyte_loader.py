from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

from .config import ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "shjwudp_megabyte"


def ensure_repo_on_path() -> None:
    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def build_model(model_config: ModelConfig):
    ensure_repo_on_path()
    use_input_causal_conv = model_config.implementation == "megabyte_in_action_causal_conv"
    if model_config.implementation == "megabyte_in_action":
        from model.megabyte_in_action import Megabyte, MegabyteConfig
    elif use_input_causal_conv:
        from dna_compress.megabyte_in_action_causal_conv import Megabyte, MegabyteConfig
    elif model_config.implementation == "megabyte_relative":
        from model.megabyte_relative import Megabyte, MegabyteConfig
    else:
        from model.megabyte import Megabyte, MegabyteConfig

    config_kwargs = dict(
        V=model_config.vocab_size,
        P=model_config.patch_size,
        D_G=model_config.global_dim,
        D_L=model_config.local_dim,
        T_MAX=model_config.seq_length,
        g_nheads=model_config.global_heads,
        g_nlayers=model_config.global_layers,
        l_nheads=model_config.local_heads,
        l_nlayers=model_config.local_layers,
        attn_dropout=model_config.attn_dropout,
        ff_dropout=model_config.ff_dropout,
        initializer_range=model_config.initializer_range,
        pad_id=model_config.pad_id,
        eos_id=model_config.eos_id,
    )
    if use_input_causal_conv:
        config_kwargs["input_causal_conv_kernel_size"] = model_config.input_causal_conv_kernel_size
    native_config = MegabyteConfig(**config_kwargs)
    return Megabyte(native_config)


def _extract_model_state(state: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    if not isinstance(state, dict):
        raise ValueError("Megabyte checkpoint must deserialize to a dict-like object.")

    metadata: dict[str, Any] = {}
    raw_checkpoint: dict[str, Any] | None = None
    if "model_state" in state:
        metadata = {
            key: value
            for key, value in state.items()
            if key not in {"model_state", "optimizer_state"}
        }
        raw_checkpoint = state
        return state["model_state"], metadata, raw_checkpoint

    tensor_values = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if tensor_values and len(tensor_values) == len(state):
        return state, metadata, None

    raise ValueError("Unable to locate model weights in Megabyte checkpoint.")


def load_megabyte_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Megabyte checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location)
    return _extract_model_state(state)
