from __future__ import annotations

from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from model.attend import Attend


class Rearrange(nn.Module):
    def __init__(self, pattern: str, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, self.pattern, **self.axes_lengths)


MegabyteConfig = namedtuple(
    "MegabyteConfig",
    [
        "V", "P", "D_G", "D_L", "T_MAX",
        "g_nheads", "g_nlayers",
        "l_nheads", "l_nlayers",
        "attn_dropout", "ff_dropout",
        "input_causal_conv_kernel_size",
        "initializer_range",
        "pad_id", "eos_id",
    ],
    defaults=(0.0, 0.0, 1)
)


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.0,
        flash=False,
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal=True,
            flash=flash,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, attn_bias=None):
        h = self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        out = self.attend(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def FeedForward(*, dim, mult=4, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        flash_attn=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nheads = heads

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

    def alibi_bias(self, length: int) -> torch.Tensor:
        nheads = self.nheads
        slopes = torch.tensor([2 ** ((-8 / nheads) * i) for i in range(1, nheads + 1)])
        positions = torch.arange(length)
        return positions.view(1, 1, length) * slopes.view(nheads, 1, 1)

    def forward(self, x):
        _, length, _ = x.shape
        attn_bias = self.alibi_bias(length).to(x.dtype).to(x.device)
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x


MegabyteOutput = namedtuple(
    "MegabyteOutput",
    [
        "lm_logits", "loss", "metrics",
    ]
)


class CausalConv1d(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"input_causal_conv_kernel_size must be > 0, got {kernel_size}.")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)

        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class Megabyte(nn.Module):
    def __init__(
        self,
        config: MegabyteConfig,
    ):
        super().__init__()
        self.config = config
        patch_size = config.P
        vocab_size = config.V
        global_dim = config.D_G
        local_dim = config.D_L

        self.to_embed = nn.Embedding(vocab_size, global_dim)
        self.input_causal_conv = CausalConv1d(global_dim, config.input_causal_conv_kernel_size)
        self.g_transformer = Transformer(
            dim=patch_size * global_dim,
            layers=config.g_nlayers,
            dim_head=(patch_size * global_dim) // config.g_nheads,
            heads=config.g_nheads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            flash_attn=True,
        )
        self.gl_linear = nn.Sequential(
            Rearrange("... (P D_G) -> ... P D_G", P=patch_size, D_G=global_dim),
            nn.Linear(global_dim, local_dim),
            Rearrange("... P D_L -> (...) P D_L", P=patch_size, D_L=local_dim),
        )

        self.to_l_embed = nn.Linear(global_dim, local_dim)
        self.l_transformer = Transformer(
            dim=local_dim,
            layers=config.l_nlayers,
            dim_head=local_dim // config.l_nheads,
            heads=config.l_nheads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            flash_attn=True,
        )

        self.to_logits = nn.Linear(local_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, ids, return_loss=False, return_metrics=False):
        batch_size, sequence_length = ids.shape
        patch_size = self.config.P
        patch_count = sequence_length // patch_size
        global_dim = self.config.D_G
        loss = None
        metrics = None

        pad_ids = F.pad(
            rearrange(ids, "... (K P) -> ... K P", K=patch_count, P=patch_size),
            (0, 0, 1, 0),
            value=self.config.pad_id,
        )
        pad_embed = self.to_embed(rearrange(pad_ids, "B ... -> B (...)", B=batch_size))
        pad_embed = self.input_causal_conv(pad_embed)
        pad_embed = rearrange(
            pad_embed,
            "... (K P) D_G -> ... K P D_G", K=patch_count + 1, P=patch_size, D_G=global_dim,
        )

        global_in = rearrange(
            pad_embed[:, :patch_count, ...],
            "... K P D_G -> ... K (P D_G)", K=patch_count, P=patch_size, D_G=global_dim,
        )
        global_out = self.g_transformer(global_in)

        if return_metrics:
            metrics = {
                "global_in_norm": global_in.norm(),
                "global_out_norm": global_out.norm(),
                "input_conv_kernel_size": float(self.config.input_causal_conv_kernel_size),
            }

        local_embed = rearrange(
            pad_embed,
            "B K P ... -> B (K P) ...",
            B=batch_size,
            K=patch_count + 1,
            P=patch_size,
        )[:, patch_size - 1:-1, :]
        local_embed = self.to_l_embed(
            rearrange(local_embed, "B (K P) ... -> (B K) P ...", B=batch_size, K=patch_count, P=patch_size)
        )
        local_in = self.gl_linear(global_out) + local_embed
        local_out = self.l_transformer(local_in)

        if return_metrics:
            metrics.update({
                "local_in_norm": local_in.norm(),
                "local_out_norm": local_out.norm(),
            })

        lm_logits = self.to_logits(local_out)

        if return_loss:
            labels = ids
            loss = F.cross_entropy(
                rearrange(lm_logits, "... V -> (...) V", V=self.config.V),
                rearrange(labels, "... -> (...)"),
                ignore_index=self.config.pad_id,
            )

        lm_logits = rearrange(lm_logits, "(B K) P ... -> B (K P) ...", B=batch_size, K=patch_count, P=patch_size)
        return MegabyteOutput(lm_logits=lm_logits, loss=loss, metrics=metrics)
