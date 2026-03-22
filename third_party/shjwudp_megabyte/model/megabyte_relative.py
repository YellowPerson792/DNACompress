# MIT License
# Copyright (c) 2023 Jianbin Chang

"""
Megabyte variant that replaces the original global absolute position embedding
with relative attention bias (ALiBi-style), while preserving the original
local embedding, local shift scheme, and local transformer position handling
from megabyte.py.
"""

# MIT License
# Copyright (c) 2023 Phil Wang

from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from model.attend import Attend


MegabyteConfig = namedtuple(
    "MegabyteConfig",
    [
        "V", "P", "D_G", "D_L", "T_MAX",
        "g_nheads", "g_nlayers",
        "l_nheads", "l_nlayers",
        "attn_dropout", "ff_dropout",
        "initializer_range",
        "pad_id", "eos_id",
    ],
    defaults=(0.0, 0.0)
)


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        h = self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        out = self.attend(q, k, v, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def FeedForward(*, dim, mult = 4, dropout = 0.):
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
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        flash_attn = False,
        use_alibi = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nheads = heads
        self.use_alibi = use_alibi
        if use_alibi:
            slopes = torch.tensor([2 ** ((-8 / heads) * i) for i in range(1, heads + 1)], dtype=torch.float32)
            self.register_buffer("alibi_slopes", slopes, persistent=False)

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

    def alibi_bias(self, length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=dtype)
        slopes = self.alibi_slopes.to(device=device, dtype=dtype)
        return positions.view(1, 1, length) * slopes.view(self.nheads, 1, 1)

    def forward(self, x):
        attn_bias = None
        if self.use_alibi:
            _, length, _ = x.shape
            attn_bias = self.alibi_bias(length, device=x.device, dtype=x.dtype)
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


class Megabyte(nn.Module):
    """
    notation
    V - vocabulary size
    P - patch size
    D_G - global dimension
    D_L - local dimension
    T - sequence length
    """

    def __init__(
        self,
        config: MegabyteConfig,
    ):
        super().__init__()
        self.config = config
        P = config.P
        V = config.V
        D_G = config.D_G
        D_L = config.D_L

        self.g_embedder = nn.Embedding(V, D_G)
        self.g_transformer = Transformer(
            dim=config.P * config.D_G,
            layers=config.g_nlayers,
            dim_head=(config.P * config.D_G) // config.g_nheads,
            heads=config.g_nheads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            flash_attn=True,
            use_alibi=True,
        )
        self.gl_linear = nn.Sequential(
            Rearrange("... (P D_G) -> ... P D_G", P=P, D_G=D_G),
            nn.Linear(D_G, D_L),
            Rearrange("... P D_L -> (...) P D_L", P=P, D_L=D_L),
        )

        self.l_embedder = nn.Embedding(V, D_L)
        self.l_transformer = Transformer(
            dim=config.D_L,
            layers=config.l_nlayers,
            dim_head=config.D_L // config.l_nheads,
            heads=config.l_nheads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            flash_attn=True,
        )

        self.to_logits = nn.Linear(D_L, V)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
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
        B, T = ids.shape
        P = self.config.P
        K = T // P
        D_G = self.config.D_G
        loss = None
        metrics = None

        g_pad_ids = F.pad(
            rearrange(ids, "... (K P) -> ... K P", K=K, P=P),
            (0, 0, 1, -1),
            value=self.config.pad_id
        )
        g_pad_embed = self.g_embedder(rearrange(g_pad_ids, "B ... -> B (...)", B=B))
        global_in = rearrange(
            g_pad_embed,
            "... (K P) D_G -> ... K (P D_G)", K=K, P=P, D_G=D_G,
        )
        global_out = self.g_transformer(global_in)

        if return_metrics:
            metrics = {
                "global_in_norm": global_in.norm(),
                "global_out_norm": global_out.norm(),
            }

        l_input_ids = rearrange(ids, "B (K P) -> (B K) P", B=B, K=K, P=P)
        l_input_ids = F.pad(l_input_ids, (1, -1), value=self.config.pad_id)
        l_embed = self.l_embedder(l_input_ids)
        local_in = self.gl_linear(global_out) + l_embed
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
                rearrange(lm_logits, "... V ->  (...) V", V=self.config.V),
                rearrange(labels, "... -> (...)"),
                ignore_index=self.config.pad_id,
            )

        lm_logits = rearrange(lm_logits, "(B K) P ... -> B (K P) ...", B=B, K=K, P=P)

        return MegabyteOutput(lm_logits=lm_logits, loss=loss, metrics=metrics)
