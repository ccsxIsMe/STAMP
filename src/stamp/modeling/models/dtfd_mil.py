"""
DTFD-MIL style two-tier distillation MIL backbone.

This implementation follows the core idea of distilling informative instances
from multiple sub-bags, then performing a second-stage bag prediction on the
distilled set.
"""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class _GatedAttentionPool(nn.Module):
    def __init__(self, dim_hidden: int):
        super().__init__()
        self.attn_v = nn.Linear(dim_hidden, dim_hidden)
        self.attn_u = nn.Linear(dim_hidden, dim_hidden)
        self.attn_w = nn.Linear(dim_hidden, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_v = torch.tanh(self.attn_v(x))
        attn_u = torch.sigmoid(self.attn_u(x))
        attn = self.attn_w(attn_v * attn_u)
        attn = F.softmax(attn, dim=1)
        pooled = (attn * x).sum(dim=1)
        return pooled, attn


class DTFDMIL(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        dropout: float = 0.25,
        n_groups: int = 4,
        distill_topk: int = 4,
        distill_bottomk: int = 2,
        aux_loss_blend: float = 0.5,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.n_groups = n_groups
        self.distill_topk = distill_topk
        self.distill_bottomk = distill_bottomk
        self.aux_loss_blend = aux_loss_blend

        self.feature_proj = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.group_pool = _GatedAttentionPool(dim_hidden)
        self.group_classifier = nn.Linear(dim_hidden, dim_output)
        self.instance_classifier = nn.Linear(dim_hidden, dim_output)

        self.distill_pool = _GatedAttentionPool(dim_hidden)
        self.distill_classifier = nn.Linear(dim_hidden, dim_output)

    @staticmethod
    def _split_groups(x: Tensor, n_groups: int) -> list[Tensor]:
        group_sizes = [x.size(1) // n_groups] * n_groups
        for i in range(x.size(1) % n_groups):
            group_sizes[i] += 1
        return [chunk for chunk in torch.split(x, group_sizes, dim=1) if chunk.size(1) > 0]

    @staticmethod
    def _gather_indices(x: Tensor, indices: Tensor) -> Tensor:
        return torch.gather(
            x,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )

    def _distill_instances(
        self,
        group_feats: Tensor,
        instance_logits: Tensor,
    ) -> Tensor:
        confidence = instance_logits.softmax(dim=-1).max(dim=-1).values

        n_instances = group_feats.size(1)
        k_top = min(self.distill_topk, n_instances)
        top_indices = confidence.topk(k_top, dim=1).indices
        selected = [self._gather_indices(group_feats, top_indices)]

        if self.distill_bottomk > 0 and n_instances > k_top:
            k_bottom = min(self.distill_bottomk, n_instances - k_top)
            bottom_indices = confidence.topk(k_bottom, dim=1, largest=False).indices
            selected.append(self._gather_indices(group_feats, bottom_indices))

        return torch.cat(selected, dim=1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        _ = kwargs

        feats = self.feature_proj(h)
        groups = self._split_groups(feats, self.n_groups)

        group_logits = []
        distilled_feats = []

        for group_feats in groups:
            group_repr, _ = self.group_pool(group_feats)
            group_logits.append(self.group_classifier(group_repr))

            instance_logits = self.instance_classifier(group_feats)
            distilled_feats.append(
                self._distill_instances(group_feats, instance_logits)
            )

        distilled = torch.cat(distilled_feats, dim=1)
        bag_repr, _ = self.distill_pool(distilled)
        bag_logits = self.distill_classifier(bag_repr)

        aux_logits = torch.stack(group_logits, dim=0).mean(dim=0)
        return (1 - self.aux_loss_blend) * bag_logits + self.aux_loss_blend * aux_logits
