from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .nugget_data import IGNORE_INDEX


BYTECAPTION_LATENT_MODES = ("dense", "continuous_bottleneck", "flatten_bottleneck")
BYTECAPTION_HIDDEN_STORAGE_DTYPES = ("runtime", "float32", "float16", "bfloat16")


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "bytecaption"


def _ensure_bytecaption_vendor_importable() -> Path:
    root = _vendor_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def build_byteformer_encoder(config: ModelConfig) -> nn.Module:
    vendor = _ensure_bytecaption_vendor_importable()
    from corenet.options.opts import get_training_arguments
    from byteformer_hf_migration.utils.hf_adapter_utils import (
        CorenetToHFPretrainedConfig,
        CorenetToHFPretrainedModel,
    )
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import BaseModelOutput

    class ByteFormerWrapper(PreTrainedModel):
        config_class = CorenetToHFPretrainedConfig
        main_input_name = "input_ids"

        def __init__(self, byteformer_model: nn.Module, hf_config: CorenetToHFPretrainedConfig) -> None:
            super().__init__(hf_config)
            self.byteformer = byteformer_model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
            x, key_padding_mask = self.byteformer.get_backbone_inputs(input_ids)
            x, updated_mask = self.byteformer.backbone_forward(x, key_padding_mask)
            output_mask = (updated_mask != float("-inf")).to(dtype=torch.float32)
            output = BaseModelOutput(last_hidden_state=x)
            output.encoder_attention_mask = output_mask
            return output

    config_path = (
        Path(config.bytecaption_byteformer_config_path)
        if config.bytecaption_byteformer_config_path
        else vendor / "byteformer_hf_migration" / "configs" / "conv_kernel_size=4,window_sizes=[128].yaml"
    )
    weight_path = Path(config.bytecaption_byteformer_weight_path) if config.bytecaption_byteformer_weight_path else None
    args = [
        "--common.config-file",
        str(config_path),
        "--model.classification.n-classes",
        "1000",
        "--model.classification.byteformer.vocab-size",
        str(config.vocab_size),
        "--model.classification.byteformer.max-num-tokens",
        str(max(config.seq_length, 1)),
        "--dataset.root-train",
        str(vendor),
        "--dataset.root-val",
        str(vendor),
    ]
    opts = get_training_arguments(args=args)
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer = CorenetToHFPretrainedModel(hf_config, int(config.vocab_size)).model
    if weight_path is not None and weight_path.exists():
        weights = torch.load(weight_path, map_location="cpu")
        model_state = byteformer.state_dict()
        compatible = {
            key: value
            for key, value in weights.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        byteformer.load_state_dict(compatible, strict=False)
    if hasattr(byteformer, "classifier"):
        delattr(byteformer, "classifier")
    return ByteFormerWrapper(byteformer, hf_config)


def byteformer_output_token_count(byteformer: nn.Module, input_token_count: int) -> int:
    """Infer output token count from the actual ByteFormer reduction modules."""
    core = byteformer.byteformer if hasattr(byteformer, "byteformer") else byteformer
    count = int(input_token_count)
    token_reduction = getattr(core, "token_reduction_net", None)
    if token_reduction is not None:
        kernel_value = getattr(token_reduction, "kernel_size", (getattr(core, "conv_kernel_size", 1),))
        stride_value = getattr(token_reduction, "stride", (max(1, int(getattr(core, "conv_kernel_size", 2)) // 2),))
        kernel = int(kernel_value[0] if isinstance(kernel_value, tuple) else kernel_value)
        stride = int(stride_value[0] if isinstance(stride_value, tuple) else stride_value)
        if count < kernel:
            raise ValueError(f"ByteFormer seq_length={count} is smaller than token reduction kernel size {kernel}.")
        count = ((count - kernel) // stride) + 1
    get_downsampler = getattr(core, "get_downsampler", None)
    for layer_index, _layer in enumerate(getattr(core, "transformer", [])):
        downsampler = get_downsampler(layer_index) if get_downsampler is not None else None
        if downsampler is not None:
            count = math.ceil(count / int(getattr(downsampler, "window", 2)))
    return max(1, count)


def _subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((1, size, size), device=device), diagonal=1) == 0


def _sinusoid_encoding_table(max_len: int, dim: int, device: torch.device | None = None) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim))
    table = torch.zeros((max_len, dim), dtype=torch.float32, device=device)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div[: table[:, 1::2].shape[1]])
    if max_len > 0:
        table[0] = 0
    return table


class PureTDecoderAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("bytecaption_decoder_dim must be divisible by bytecaption_decoder_heads.")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch, q_len, _ = q.shape
        q = self.q(q).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(k).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(v).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, q_len, self.dim)
        return self.o(out)


class PureTDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, ff_dropout: float, use_gx: bool = True) -> None:
        super().__init__()
        self.use_gx = use_gx
        self.word_attn = PureTDecoderAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = PureTDecoderAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ff_dropout),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if use_gx:
            self.gx_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout))
            self.gx_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        gx: torch.Tensor,
        seq_mask: torch.Tensor,
        att_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_gx:
            x = self.gx_norm(x + self.gx_fuse(torch.cat([x, gx.unsqueeze(1).expand_as(x)], dim=-1)))
        shortcut = x
        x = self.word_attn(x, x, x, seq_mask)
        x = self.norm1(shortcut + self.dropout(x))

        shortcut = x
        if self.use_gx:
            kv = torch.cat([encoder_out, gx.unsqueeze(1)], dim=1)
            if att_mask is None:
                cross_mask = None
            else:
                ones = torch.ones((att_mask.shape[0], 1), dtype=att_mask.dtype, device=att_mask.device)
                cross_mask = torch.cat([att_mask, ones], dim=1).unsqueeze(1)
        else:
            kv = encoder_out
            cross_mask = att_mask.unsqueeze(1) if att_mask is not None else None
        x = self.cross_attn(x, kv, kv, cross_mask)
        x = self.norm2(shortcut + self.dropout(x))

        shortcut = x
        x = self.ff(x)
        return self.norm3(shortcut + self.dropout(x))


class PureTDecoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        dim: int,
        layers: int,
        heads: int,
        dropout: float,
        ff_dropout: float,
        max_positions: int,
    ) -> None:
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.embed_scale = math.sqrt(dim)
        self.register_buffer("pos_embed", _sinusoid_encoding_table(max_positions + 2, dim), persistent=False)
        self.layers = nn.ModuleList(
            [PureTDecoderLayer(dim, heads, dropout, ff_dropout, use_gx=True) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.generator = nn.Linear(dim, vocab_size)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        gx: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = int(decoder_input_ids.shape[1])
        if seq_len + 1 > self.pos_embed.shape[0]:
            raise ValueError(f"Decoder sequence length {seq_len} exceeds configured max positions.")
        pos = self.pos_embed[1 : seq_len + 1].to(device=decoder_input_ids.device, dtype=self.word_embed.weight.dtype)
        x = self.embed_scale * self.word_embed(decoder_input_ids) + pos.unsqueeze(0)
        causal = _subsequent_mask(seq_len, decoder_input_ids.device)
        if decoder_attention_mask is not None:
            seq_mask = decoder_attention_mask.to(dtype=torch.float32).unsqueeze(1) * causal.to(dtype=torch.float32)
        else:
            seq_mask = causal.to(dtype=torch.float32)
        for layer in self.layers:
            x = layer(x, encoder_out, gx, seq_mask, encoder_attention_mask)
        return self.generator(self.dropout(x))


@dataclass
class ByteCaptionLatent:
    decoder_features: torch.Tensor
    attention_mask: torch.Tensor
    payload: torch.Tensor
    latent_mode: str


class ByteCaptionDNACompressor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_mode = config.bytecaption_latent_mode
        self.seq_length = int(config.seq_length)
        self.pad_id = int(config.pad_id)
        self.decoder_start_token_id = int(config.pad_id)
        self.decoder_dim = int(config.bytecaption_decoder_dim)
        self.code_dim = int(config.bytecaption_code_dim)
        self.flatten_bottleneck_dim = int(config.bytecaption_flatten_bottleneck_dim)

        self.byteformer = build_byteformer_encoder(config)
        byteformer_dim = int(getattr(self.byteformer.config, "hidden_size", self.decoder_dim))
        self.byteformer_dim = byteformer_dim
        byteformer_vocab = int(self.byteformer.byteformer.embeddings.num_embeddings)
        if byteformer_vocab != int(config.vocab_size):
            raise ValueError(
                f"ByteFormer vocab size {byteformer_vocab} does not match DNA vocab size {config.vocab_size}."
            )
        self.flatten_input_tokens = byteformer_output_token_count(self.byteformer, self.seq_length)
        self.flatten_input_dim = self.flatten_input_tokens * self.code_dim

        if self.latent_mode == "dense":
            self.encoder_to_decoder = nn.Linear(byteformer_dim, self.decoder_dim)
            self.encoder_to_code = None
            self.code_to_decoder = None
        else:
            self.encoder_to_code = nn.Linear(byteformer_dim, self.code_dim)
            self.code_to_decoder = nn.Linear(self.code_dim, self.decoder_dim)
            self.encoder_to_decoder = None

        if self.latent_mode == "flatten_bottleneck":
            self.flatten_to_bottleneck = nn.Linear(self.flatten_input_dim, self.flatten_bottleneck_dim)
            self.bottleneck_to_flatten = nn.Linear(self.flatten_bottleneck_dim, self.flatten_input_dim)
        else:
            self.flatten_to_bottleneck = None
            self.bottleneck_to_flatten = None

        use_ln = bool(config.bytecaption_bottleneck_layer_norm)
        self.code_ln = nn.LayerNorm(self.code_dim) if use_ln and self.latent_mode != "dense" else nn.Identity()
        self.flatten_ln = nn.LayerNorm(self.flatten_bottleneck_dim) if use_ln and self.latent_mode == "flatten_bottleneck" else nn.Identity()
        self.decoder_input_ln = nn.LayerNorm(self.decoder_dim) if use_ln else nn.Identity()

        self.decoder = PureTDecoder(
            vocab_size=int(config.vocab_size),
            pad_id=int(config.pad_id),
            dim=self.decoder_dim,
            layers=int(config.bytecaption_decoder_layers),
            heads=int(config.bytecaption_decoder_heads),
            dropout=float(config.bytecaption_decoder_dropout),
            ff_dropout=float(config.bytecaption_decoder_ff_dropout),
            max_positions=int(config.seq_length),
        )
        if int(self.decoder.word_embed.num_embeddings) != int(config.vocab_size):
            raise ValueError("PureT decoder vocab size does not match DNA vocab size.")

    def _decoder_inputs(self, labels: torch.Tensor) -> torch.Tensor:
        shifted = torch.full_like(labels, self.pad_id)
        shifted[:, 0] = self.decoder_start_token_id
        previous = labels[:, :-1].masked_fill(labels[:, :-1] == IGNORE_INDEX, self.pad_id)
        shifted[:, 1:] = previous
        return shifted

    def _encode_byteformer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.byteformer(input_ids=input_ids, attention_mask=attention_mask)
        features = output.last_hidden_state
        mask = getattr(output, "encoder_attention_mask", None)
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.float32, device=features.device)
        return features, mask.to(device=features.device, dtype=torch.float32)

    def encode_latent(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> ByteCaptionLatent:
        features, mask = self._encode_byteformer(input_ids, attention_mask)
        if self.latent_mode == "dense":
            decoder_features = self.decoder_input_ln(self.encoder_to_decoder(features))
            return ByteCaptionLatent(decoder_features=decoder_features, attention_mask=mask, payload=decoder_features, latent_mode=self.latent_mode)

        code = self.code_ln(self.encoder_to_code(features))
        code = code * mask.unsqueeze(-1)
        if self.latent_mode == "continuous_bottleneck":
            decoder_features = self.decoder_input_ln(self.code_to_decoder(code))
            return ByteCaptionLatent(decoder_features=decoder_features, attention_mask=mask, payload=code, latent_mode=self.latent_mode)

        if code.shape[1] > self.flatten_input_tokens:
            raise ValueError(
                f"ByteCaption flatten_bottleneck got {code.shape[1]} encoder tokens, "
                f"but seq_length allows {self.flatten_input_tokens}."
            )
        padded = code.new_zeros((code.shape[0], self.flatten_input_tokens, self.code_dim))
        padded[:, : code.shape[1], :] = code
        bottleneck = self.flatten_ln(self.flatten_to_bottleneck(padded.reshape(code.shape[0], -1)))
        restored = self.bottleneck_to_flatten(bottleneck).reshape(code.shape[0], self.flatten_input_tokens, self.code_dim)
        restored = restored[:, : code.shape[1], :] * mask.unsqueeze(-1)
        decoder_features = self.decoder_input_ln(self.code_to_decoder(restored))
        return ByteCaptionLatent(
            decoder_features=decoder_features,
            attention_mask=mask,
            payload=bottleneck,
            latent_mode=self.latent_mode,
        )

    def decode_from_latent(
        self,
        latent_payload: torch.Tensor,
        latent_attention_mask: torch.Tensor,
        *,
        labels: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        if self.latent_mode == "dense":
            decoder_features = latent_payload
        elif self.latent_mode == "continuous_bottleneck":
            decoder_features = self.decoder_input_ln(self.code_to_decoder(latent_payload))
        elif self.latent_mode == "flatten_bottleneck":
            restored = self.bottleneck_to_flatten(latent_payload).reshape(
                latent_payload.shape[0], self.flatten_input_tokens, self.code_dim
            )
            restored = restored[:, : latent_attention_mask.shape[1], :] * latent_attention_mask.unsqueeze(-1)
            decoder_features = self.decoder_input_ln(self.code_to_decoder(restored))
        else:
            raise ValueError(f"Unsupported ByteCaption latent mode: {self.latent_mode}")
        return self.decode_features(
            decoder_features=decoder_features,
            encoder_attention_mask=latent_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def decode_features(
        self,
        *,
        decoder_features: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        denom = encoder_attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gx = (decoder_features * encoder_attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        decoder_input_ids = self._decoder_inputs(labels)
        logits = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_out=decoder_features,
            gx=gx,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX)
        return SimpleNamespace(logits=logits, loss=loss)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        latent = self.encode_latent(input_ids=input_ids, attention_mask=attention_mask)
        output = self.decode_features(
            decoder_features=latent.decoder_features,
            encoder_attention_mask=latent.attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        output.latent = latent
        return output


def build_bytecaption_model(config: ModelConfig) -> ByteCaptionDNACompressor:
    if config.bytecaption_latent_mode not in BYTECAPTION_LATENT_MODES:
        raise ValueError(f"model.bytecaption_latent_mode must be one of: {', '.join(BYTECAPTION_LATENT_MODES)}")
    return ByteCaptionDNACompressor(config)


def load_bytecaption_checkpoint(model: nn.Module, path: str | Path, *, strict: bool = True) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint
