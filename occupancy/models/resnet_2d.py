__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

from torchvision import models
from torch import nn
from torch import Tensor


def create_resenet_feature_extractor(model_id: str, **kwargs) -> nn.Module:
    resnet = models.get_model(
        model_id, weights=models.get_model_weights(model_id).DEFAULT, **kwargs
    )
    model = ModuleDictWrapper(
        key_order=[
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ],
        layers={
            "conv1": resnet.conv1,
            "bn1": resnet.bn1,
            "relu": resnet.relu,
            "maxpool": resnet.maxpool,
            "layer1": resnet.layer1,
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        },
    )
    return model


class ModuleDictWrapper(nn.ModuleDict):
    def __init__(
        self,
        key_order: list[str],
        layers: dict[str, nn.Module],
    ):
        super().__init__(layers)
        self.key_order = key_order

    def forward(self, input: Tensor) -> Tensor:
        for key in self.key_order:
            input = self[key](input)
        return input


def resnet18(**kwargs) -> nn.Module:
    return create_resenet_feature_extractor("resnet18", **kwargs)


def resnet34(**kwargs) -> nn.Module:
    return create_resenet_feature_extractor("resnet34", **kwargs)


def resnet50(**kwargs) -> nn.Module:
    return create_resenet_feature_extractor("resnet50", **kwargs)


def resnet101(**kwargs) -> nn.Module:
    return create_resenet_feature_extractor("resnet101", **kwargs)


def resnet152(**kwargs) -> nn.Module:
    return create_resenet_feature_extractor("resnet152", **kwargs)
