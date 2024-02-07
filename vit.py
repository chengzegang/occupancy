from torch import Tensor, nn
from torchvision import models  # type: ignore
from torchvision.models.feature_extraction import (  # type: ignore
    create_feature_extractor,
)


class VisionTransformer(nn.Module):
    def __init__(self, feat_dim: int, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        self.encoder.requires_grad_(False)
        for i in range(8):
            self.encoder.encoder.layers[-i].requires_grad_(True)
        self.encoder = create_feature_extractor(
            self.encoder, {"encoder.layers.encoder_layer_23.add_1": "encoder_output"}
        )
        self.pre_avg_proj = nn.Linear(1024, feat_dim)
        self.out_norm = nn.InstanceNorm1d(feat_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)["encoder_output"]
        x = self.pre_avg_proj(x)
        x = x.mean(dim=-2)
        x = self.out_norm(x)
        return x
