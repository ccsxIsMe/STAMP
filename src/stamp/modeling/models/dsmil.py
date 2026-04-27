"""
Dual-Stream Multiple Instance Learning (DSMIL) style MIL backbone.

Adapted for the STAMP single-head classification interface from:
Li et al., CVPR 2021
"Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification"
https://arxiv.org/abs/2011.08939
"""

import math

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class DSMIL(nn.Module):
    """Critical-instance driven MIL classifier.

    The model first predicts instance logits for every tile, then selects the
    highest-scoring critical instance for each class and aggregates the bag with
    class-specific attention centered on that instance.
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.feature_proj = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.instance_classifier = nn.Linear(dim_hidden, dim_output)
        self.attention_norm = nn.LayerNorm(dim_hidden)
        self.bag_classifier = nn.Parameter(torch.empty(dim_output, dim_hidden))
        self.bag_bias = nn.Parameter(torch.zeros(dim_output))

        nn.init.xavier_uniform_(self.bag_classifier)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        _ = kwargs

        feats = self.feature_proj(h)  # (B, N, H)
        instance_logits = self.instance_classifier(feats)  # (B, N, C)

        critical_indices = instance_logits.argmax(dim=1)  # (B, C)
        critical_feats = torch.gather(
            feats,
            dim=1,
            index=critical_indices.unsqueeze(-1).expand(-1, -1, self.dim_hidden),
        )  # (B, C, H)

        normalized_feats = self.attention_norm(feats)
        normalized_critical = self.attention_norm(critical_feats)
        attn_scores = torch.einsum(
            "bnh,bch->bnc",
            normalized_feats,
            normalized_critical,
        ) / math.sqrt(self.dim_hidden)
        attn = F.softmax(attn_scores, dim=1)

        bag_repr = torch.einsum("bnc,bnh->bch", attn, feats)  # (B, C, H)
        bag_logits = (
            torch.einsum("bch,ch->bc", bag_repr, self.bag_classifier)
            + self.bag_bias
        )

        max_instance_logits = instance_logits.max(dim=1).values
        return 0.5 * (bag_logits + max_instance_logits)
