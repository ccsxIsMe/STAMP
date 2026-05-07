"""
Attention-Based MIL (ABMIL), adapted for STAMP.
"""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class ABMIL(nn.Module):
    """Gated Attention-Based MIL aggregator."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(dim_hidden, dim_hidden)
        self.attn_U = nn.Linear(dim_hidden, dim_hidden)
        self.attn_w = nn.Linear(dim_hidden, 1)
        self.classifier = nn.Linear(dim_hidden, dim_output)

    @jaxtyped(typechecker=beartype)
    def encode_bag(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_hidden"]:
        _ = kwargs
        h = self.feature_proj(h)
        attn_v = torch.tanh(self.attn_V(h))
        attn_u = torch.sigmoid(self.attn_U(h))
        attn = self.attn_w(attn_v * attn_u)
        attn = F.softmax(attn, dim=1)
        return (attn * h).sum(dim=1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        return self.classifier(self.encode_bag(h, **kwargs))
