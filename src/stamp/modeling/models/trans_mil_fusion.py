"""
TransMIL with an additional gated-attention bag pooling branch.

The goal is to preserve the strong contextual modeling of TransMIL while adding
an explicit discriminative bag summary that can focus on informative regions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.models.trans_mil import PPEG, Transformer


class TransMILFusion(nn.Module):
    def __init__(
        self,
        dim_output: int,
        dim_input: int,
        dim_hidden: int,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.pos_layer = PPEG(dim=dim_hidden)
        self.feature_proj = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        self.layer1 = Transformer(dim=dim_hidden)
        self.layer2 = Transformer(dim=dim_hidden)
        self.norm = nn.LayerNorm(dim_hidden)

        self.attn_v = nn.Linear(dim_hidden, dim_hidden)
        self.attn_u = nn.Linear(dim_hidden, dim_hidden)
        self.attn_w = nn.Linear(dim_hidden, 1)

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(dim_hidden, dim_output)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, h: Float[Tensor, "batch tiles dim_input"], **kwargs
    ) -> Float[Tensor, "batch n_classes"]:
        _ = kwargs

        h = self.feature_proj(h)

        n_tiles = h.shape[1]
        side = int(np.ceil(np.sqrt(n_tiles)))
        add_length = side * side - n_tiles
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        batch_size = h.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        h = self.norm(h)

        cls_repr = h[:, 0]
        tile_repr = h[:, 1 : n_tiles + 1]

        attn_v = torch.tanh(self.attn_v(tile_repr))
        attn_u = torch.sigmoid(self.attn_u(tile_repr))
        attn = self.attn_w(attn_v * attn_u)
        attn = F.softmax(attn, dim=1)
        bag_repr = (attn * tile_repr).sum(dim=1)

        fusion_input = torch.cat([cls_repr, bag_repr], dim=-1)
        gate = self.fusion_gate(fusion_input)
        fused_repr = gate * cls_repr + (1 - gate) * bag_repr
        return self.classifier(fused_repr)
