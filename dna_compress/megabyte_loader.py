from __future__ import annotations

from pathlib import Path
import sys

from .config import ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "shjwudp_megabyte"


def ensure_repo_on_path() -> None:
    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def build_model(model_config: ModelConfig):
    ensure_repo_on_path()
    if model_config.implementation == "megabyte_in_action":
        from model.megabyte_in_action import Megabyte, MegabyteConfig
    else:
        from model.megabyte import Megabyte, MegabyteConfig

    native_config = MegabyteConfig(
        V=model_config.vocab_size,
        P=model_config.patch_size,
        D_G=model_config.global_dim,
        D_L=model_config.local_dim,
        T_MAX=model_config.seq_length,
        g_nheads=model_config.global_heads,
        g_nlayers=model_config.global_layers,
        l_nheads=model_config.local_heads,
        l_nlayers=model_config.local_layers,
        initializer_range=model_config.initializer_range,
        pad_id=model_config.pad_id,
        eos_id=model_config.eos_id,
    )
    return Megabyte(native_config)
