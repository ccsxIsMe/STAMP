"""
Attention-Based MIL (ABMIL) — Ilse et al., NeurIPS 2018
"Attention-based Deep Multiple Instance Learning"
https://arxiv.org/abs/1802.04712

Gated attention variant: A = softmax(W * (tanh(V*H) ⊙ sigmoid(U*H)))
"""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class ABMIL(nn.Module):
    """Gated Attention-Based MIL aggregator.

    Args:
        dim_input:   Feature dimension of each patch (from extractor).
        dim_hidden:  Projection dimension inside attention layers.
        dim_output:  Number of output classes.
        dropout:     Dropout on the patch projection.
    """

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
        # Gated attention: two parallel branches
        self.attn_V = nn.Linear(dim_hidden, dim_hidden)   # tanh branch
        self.attn_U = nn.Linear(dim_hidden, dim_hidden)   # sigmoid gate
        self.attn_w = nn.Linear(dim_hidden, 1)            # scalar attention weight

        self.classifier = nn.Linear(dim_hidden, dim_output)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        # h: (B, N, D)  →  project to hidden dim
        h = self.feature_proj(h)                          # (B, N, H)

        # Gated attention scores
        A_V = torch.tanh(self.attn_V(h))                  # (B, N, H)
        A_U = torch.sigmoid(self.attn_U(h))               # (B, N, H)
        A = self.attn_w(A_V * A_U)                        # (B, N, 1)
        A = F.softmax(A, dim=1)                            # (B, N, 1)

        # Weighted sum → bag representation
        M = (A * h).sum(dim=1)                            # (B, H)

        return self.classifier(M)                          # (B, dim_output)
