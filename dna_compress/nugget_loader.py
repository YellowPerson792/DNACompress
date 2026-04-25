from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import torch

from .config import ModelConfig
from .nugget_tokenization import NuggetTokenizerSpec


NUGGET_REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "nugget"
REPO_ROOT = Path(__file__).resolve().parents[1]
NUGGET_BACKBONES = ("bart", "mbart", "t5")
NUGGET_HIDDEN_MODES = ("runtime_hidden", "stored_hidden")
NUGGET_HIDDEN_STORAGE_DTYPES = ("runtime", "float32", "float16", "bfloat16")


@dataclass(frozen=True)
class NuggetBackboneSpec:
    backbone: str
    d_model: int
    encoder_layers: int
    decoder_layers: int
    encoder_attention_heads: int
    decoder_attention_heads: int
    encoder_ffn_dim: int
    decoder_ffn_dim: int
    max_position_embeddings: int | None
    dropout: float | None
    attention_dropout: float | None
    activation_dropout: float | None
    t5_d_kv: int | None
    t5_dropout_rate: float | None
    vocab_size: int
    pad_id: int
    eos_id: int | None
    bos_id: int | None
    decoder_start_token_id: int
    forced_bos_token_id: int | None
    forced_eos_token_id: int | None
    config_source: str
    decoder_start_source: str


def ensure_nugget_repo_on_path() -> None:
    path = str(NUGGET_REPO_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def validate_transformers_for_nugget() -> None:
    import transformers

    version = _version_tuple(transformers.__version__)
    if not ((4, 41, 0) <= version < (4, 42, 0)):
        raise RuntimeError(
            "Nugget integration requires transformers>=4.41,<4.42 as requested by "
            f"third_party/nugget README. Current transformers version is {transformers.__version__}."
        )


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _load_json_config(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Nugget backbone config not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _override_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _bart_decoder_start(tokenizer_spec: NuggetTokenizerSpec) -> tuple[int, str]:
    if tokenizer_spec.eos_id is not None:
        return tokenizer_spec.eos_id, "eos_id"
    return tokenizer_spec.pad_id, "pad_id_no_eos"


def _resolve_backbone_spec(model_config: ModelConfig, tokenizer_spec: NuggetTokenizerSpec) -> NuggetBackboneSpec:
    backbone = model_config.nugget_backbone
    if backbone not in NUGGET_BACKBONES:
        raise ValueError(f"model.nugget_backbone must be one of: {', '.join(NUGGET_BACKBONES)}")

    if backbone in {"bart", "mbart"}:
        raw_config = _load_json_config(model_config.nugget_bart_config_path)
        d_model = int(_override_or_default(model_config.nugget_bart_d_model, raw_config["d_model"]))
        encoder_heads = int(_override_or_default(model_config.nugget_bart_encoder_attention_heads, raw_config["encoder_attention_heads"]))
        decoder_heads = int(_override_or_default(model_config.nugget_bart_decoder_attention_heads, raw_config["decoder_attention_heads"]))
        if d_model <= 0:
            raise ValueError("model.nugget_bart_d_model must be > 0 for Nugget.")
        if encoder_heads <= 0 or decoder_heads <= 0:
            raise ValueError("BART attention head counts must be > 0 for Nugget.")
        if d_model % encoder_heads != 0:
            raise ValueError("model.nugget_bart_d_model must be divisible by model.nugget_bart_encoder_attention_heads.")
        if d_model % decoder_heads != 0:
            raise ValueError("model.nugget_bart_d_model must be divisible by model.nugget_bart_decoder_attention_heads.")
        decoder_start_token_id, decoder_start_source = _bart_decoder_start(tokenizer_spec)
        eos_id = tokenizer_spec.eos_id
        forced_eos_id = eos_id if eos_id is not None else None
        return NuggetBackboneSpec(
            backbone=backbone,
            d_model=d_model,
            encoder_layers=max(1, int(_override_or_default(model_config.nugget_bart_encoder_layers, raw_config["encoder_layers"]))),
            decoder_layers=max(1, int(_override_or_default(model_config.nugget_bart_decoder_layers, raw_config["decoder_layers"]))),
            encoder_attention_heads=encoder_heads,
            decoder_attention_heads=decoder_heads,
            encoder_ffn_dim=max(1, int(_override_or_default(model_config.nugget_bart_encoder_ffn_dim, raw_config["encoder_ffn_dim"]))),
            decoder_ffn_dim=max(1, int(_override_or_default(model_config.nugget_bart_decoder_ffn_dim, raw_config["decoder_ffn_dim"]))),
            max_position_embeddings=max(
                int(model_config.seq_length),
                int(_override_or_default(model_config.nugget_bart_max_position_embeddings, raw_config["max_position_embeddings"])),
            ),
            dropout=float(_override_or_default(model_config.nugget_bart_dropout, raw_config.get("dropout", 0.1))),
            attention_dropout=float(_override_or_default(model_config.nugget_bart_attention_dropout, raw_config.get("attention_dropout", 0.0))),
            activation_dropout=float(_override_or_default(model_config.nugget_bart_activation_dropout, raw_config.get("activation_dropout", 0.0))),
            t5_d_kv=None,
            t5_dropout_rate=None,
            vocab_size=tokenizer_spec.vocab_size,
            pad_id=tokenizer_spec.pad_id,
            eos_id=eos_id,
            bos_id=decoder_start_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=forced_eos_id,
            config_source=str(_resolve_repo_path(model_config.nugget_bart_config_path)),
            decoder_start_source=decoder_start_source,
        )

    d_model = int(model_config.nugget_t5_d_model)
    num_heads = int(model_config.nugget_t5_num_heads)
    if d_model <= 0:
        raise ValueError("model.nugget_t5_d_model must be > 0 for Nugget.")
    if num_heads <= 0:
        raise ValueError("model.nugget_t5_num_heads must be > 0 for Nugget.")
    if d_model % num_heads != 0:
        raise ValueError("model.nugget_t5_d_model must be divisible by model.nugget_t5_num_heads.")
    return NuggetBackboneSpec(
        backbone=backbone,
        d_model=d_model,
        encoder_layers=max(1, int(model_config.nugget_t5_num_layers)),
        decoder_layers=max(1, int(model_config.nugget_t5_num_decoder_layers)),
        encoder_attention_heads=num_heads,
        decoder_attention_heads=num_heads,
        encoder_ffn_dim=max(1, int(model_config.nugget_t5_d_ff)),
        decoder_ffn_dim=max(1, int(model_config.nugget_t5_d_ff)),
        max_position_embeddings=None,
        dropout=None,
        attention_dropout=None,
        activation_dropout=None,
        t5_d_kv=int(_override_or_default(model_config.nugget_t5_d_kv, d_model // num_heads)),
        t5_dropout_rate=float(model_config.nugget_t5_dropout_rate),
        vocab_size=tokenizer_spec.vocab_size,
        pad_id=tokenizer_spec.pad_id,
        eos_id=tokenizer_spec.eos_id,
        bos_id=None,
        decoder_start_token_id=tokenizer_spec.decoder_start_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        config_source="t5_config_fields",
        decoder_start_source="tokenizer_pad_id",
    )


def _build_hf_seq2seq_model(spec: NuggetBackboneSpec, seq_length: int):
    if spec.backbone == "t5":
        from transformers import T5Config, T5ForConditionalGeneration

        config = T5Config(
            vocab_size=spec.vocab_size,
            d_model=spec.d_model,
            d_ff=spec.encoder_ffn_dim,
            num_layers=spec.encoder_layers,
            num_decoder_layers=spec.decoder_layers,
            num_heads=spec.encoder_attention_heads,
            d_kv=spec.t5_d_kv,
            dropout_rate=spec.t5_dropout_rate,
            pad_token_id=spec.pad_id,
            eos_token_id=spec.eos_id if spec.eos_id is not None else spec.pad_id,
            decoder_start_token_id=spec.decoder_start_token_id,
        )
        return T5ForConditionalGeneration(config)

    if spec.backbone == "mbart":
        from transformers import MBartConfig, MBartForConditionalGeneration

        config = MBartConfig(
            vocab_size=spec.vocab_size,
            d_model=spec.d_model,
            encoder_layers=spec.encoder_layers,
            decoder_layers=spec.decoder_layers,
            encoder_attention_heads=spec.encoder_attention_heads,
            decoder_attention_heads=spec.decoder_attention_heads,
            encoder_ffn_dim=spec.encoder_ffn_dim,
            decoder_ffn_dim=spec.decoder_ffn_dim,
            max_position_embeddings=spec.max_position_embeddings,
            dropout=spec.dropout,
            attention_dropout=spec.attention_dropout,
            activation_dropout=spec.activation_dropout,
            pad_token_id=spec.pad_id,
            bos_token_id=spec.bos_id,
            eos_token_id=spec.eos_id if spec.eos_id is not None else spec.pad_id,
            decoder_start_token_id=spec.decoder_start_token_id,
            forced_bos_token_id=spec.forced_bos_token_id,
            forced_eos_token_id=spec.forced_eos_token_id,
        )
        return MBartForConditionalGeneration(config)

    from transformers import BartConfig, BartForConditionalGeneration

    config = BartConfig(
        vocab_size=spec.vocab_size,
        d_model=spec.d_model,
        encoder_layers=spec.encoder_layers,
        decoder_layers=spec.decoder_layers,
        encoder_attention_heads=spec.encoder_attention_heads,
        decoder_attention_heads=spec.decoder_attention_heads,
        encoder_ffn_dim=spec.encoder_ffn_dim,
        decoder_ffn_dim=spec.decoder_ffn_dim,
        max_position_embeddings=spec.max_position_embeddings,
        dropout=spec.dropout,
        attention_dropout=spec.attention_dropout,
        activation_dropout=spec.activation_dropout,
        pad_token_id=spec.pad_id,
        bos_token_id=spec.bos_id,
        eos_token_id=spec.eos_id if spec.eos_id is not None else spec.pad_id,
        decoder_start_token_id=spec.decoder_start_token_id,
        forced_bos_token_id=spec.forced_bos_token_id,
        forced_eos_token_id=spec.forced_eos_token_id,
    )
    return BartForConditionalGeneration(config)


class NuggetAutoencoder(torch.nn.Module):
    def __init__(
        self,
        *,
        scorer: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        vocab_size: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.scorer = scorer
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)

    def encode_nuggets(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.scorer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_states=encoder_out.last_hidden_state,
        )

    def decode_from_nuggets(
        self,
        *,
        nugget_encoding: torch.Tensor,
        nugget_mask: torch.Tensor,
        nugget_scores: torch.Tensor | None = None,
        labels: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        score_payload = SimpleNamespace(scores=nugget_scores)
        with self.scorer.score_context(score_payload):
            return self.decoder(
                encoder_outputs=[nugget_encoding],
                attention_mask=nugget_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        nuggets = self.encode_nuggets(input_ids=input_ids, attention_mask=attention_mask)
        with self.scorer.score_context(nuggets):
            return self.decoder(
                encoder_outputs=[nuggets.encoding],
                attention_mask=nuggets.mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )


def build_nugget_model(model_config: ModelConfig, tokenizer_spec: NuggetTokenizerSpec) -> tuple[NuggetAutoencoder, NuggetBackboneSpec]:
    validate_transformers_for_nugget()
    ensure_nugget_repo_on_path()
    from nugget import nuggify

    backbone_spec = _resolve_backbone_spec(model_config, tokenizer_spec)
    base_model = _build_hf_seq2seq_model(backbone_spec, model_config.seq_length)
    scorer, encoder, decoder, _ = nuggify(
        base_model,
        scorer_layer=model_config.nugget_scorer_layer,
        residual_start=model_config.nugget_residual_start,
        residual_end=model_config.nugget_residual_end,
        value_ffn=model_config.nugget_value_ffn,
        straight_through=model_config.nugget_straight_through,
        ratio=model_config.nugget_ratio,
    )
    return (
        NuggetAutoencoder(
            scorer=scorer,
            encoder=encoder,
            decoder=decoder,
            vocab_size=tokenizer_spec.vocab_size,
            pad_id=tokenizer_spec.pad_id,
        ),
        backbone_spec,
    )


def load_nugget_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Nugget checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict) or "model_state" not in state:
        raise ValueError(f"Nugget checkpoint '{path}' is missing 'model_state'.")
    metadata = {key: value for key, value in state.items() if key not in {"model_state", "optimizer_state"}}
    return state["model_state"], metadata, state
