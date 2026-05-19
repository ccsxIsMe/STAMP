import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.models.abmil import ABMIL


class ABMILDomain(nn.Module):
    """ABMIL with optional clinical fusion and domain-specific classifier heads.

    The last `domain_dim` entries of the clinical vector are expected to encode
    the domain/cohort indicator as a one-hot vector. Those entries are used to
    route each sample through its domain-specific classification head.
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        clinical_dim: int,
        domain_dim: int,
        clinical_hidden_dim: int = 128,
        dropout: float = 0.25,
        fusion_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if clinical_dim <= 0:
            raise ValueError("ABMILDomain requires clinical_dim > 0.")
        if domain_dim <= 1:
            raise ValueError("ABMILDomain requires domain_dim > 1.")
        if domain_dim > clinical_dim:
            raise ValueError("domain_dim cannot exceed clinical_dim.")

        self.domain_dim = domain_dim
        self.pathology_encoder = ABMIL(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            dropout=dropout,
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, clinical_hidden_dim),
            nn.LayerNorm(clinical_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(clinical_hidden_dim, dim_hidden),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid(),
        )
        self.classifier_heads = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_output) for _ in range(domain_dim)]
        )

    @jaxtyped(typechecker=beartype)
    def encode_bag(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_hidden"]:
        return self.pathology_encoder.encode_bag(h, **kwargs)

    @jaxtyped(typechecker=beartype)
    def encode_multimodal(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        *,
        clinical: Float[Tensor, "batch clinical_dim"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_hidden"]:
        bag_repr = self.encode_bag(h, **kwargs)
        clinical_repr = self.clinical_proj(clinical)
        fusion_input = torch.cat([bag_repr, clinical_repr], dim=-1)
        gate = self.fusion_gate(fusion_input)
        return gate * bag_repr + (1.0 - gate) * clinical_repr

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        *,
        clinical: Float[Tensor, "batch clinical_dim"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        fused_repr = self.encode_multimodal(h, clinical=clinical, **kwargs)
        domain_onehot = clinical[:, -self.domain_dim :]
        domain_idx = domain_onehot.argmax(dim=-1).long()

        logits_per_domain = torch.stack(
            [head(fused_repr) for head in self.classifier_heads],
            dim=1,
        )
        batch_idx = torch.arange(
            fused_repr.size(0),
            device=fused_repr.device,
        )
        return logits_per_domain[batch_idx, domain_idx]
