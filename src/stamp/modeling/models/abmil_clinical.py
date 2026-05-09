import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.models.abmil import ABMIL


class ABMILClinical(nn.Module):
    """ABMIL pathology encoder with late fusion of shared clinical features."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        clinical_dim: int,
        clinical_hidden_dim: int = 128,
        dropout: float = 0.25,
        fusion_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if clinical_dim <= 0:
            raise ValueError("ABMILClinical requires clinical_dim > 0.")

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
        self.classifier = nn.Linear(dim_hidden, dim_output)

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
        return self.classifier(fused_repr)
