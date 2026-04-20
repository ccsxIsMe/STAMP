try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "phikon dependencies not installed."
        " Please run `pip install transformers`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor


class _PhikonWrapper(torch.nn.Module):
    """Wrap HuggingFace ViT so it returns CLS token as a plain tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]  # CLS token → (B, 768)


def phikon() -> Extractor:
    """Phikon — iBOT ViT-B/16 trained on TCGA by Owkin (2024).
    HuggingFace: owkin/phikon
    Feature dim: 768
    """
    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    hf_model = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    hf_model.eval()

    model = _PhikonWrapper(hf_model)

    # Build a torchvision-compatible transform from the HF processor
    from torchvision import transforms

    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", 224)

    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return Extractor(
        model=model,
        transform=transform,
        identifier=ExtractorName.PHIKON,  # type: ignore
    )
