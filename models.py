import torch
from torchvision.models import ResNet, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, VisionTransformer, vit_b_16, ViT_B_16_Weights
from torchvision.models.resnet import ImageClassification
from typing import Tuple, Callable

def hone_resnet(classes: int) -> Callable[[ResNet], ResNet]:
    def focused(model: ResNet) -> ResNet:
        ins = model.fc.in_features
        model.fc = torch.nn.Linear(ins, classes)
        return model
    return focused

def hone_vit(classes: int) -> Callable[[VisionTransformer], VisionTransformer]:
    def focused(model: VisionTransformer) -> VisionTransformer:
        model.heads = torch.nn.Linear(model.hidden_dim, classes)
        return model
    return focused

TASK_RESNET = {
    "cifar100": hone_resnet(100),
    "cifar10": hone_resnet(10),
    "fashionMNIST": hone_resnet(10),
}

TASK_VIT = {
    "cifar100": hone_vit(100),
    "cifar10": hone_vit(10),
    "fashionMNIST": hone_vit(10),
}

def resnet(config: str, dataset: str, pretrained: bool = True) -> Tuple[ResNet, ImageClassification]:
    if config == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        if pretrained:
            model = resnet18(weights=weights)
        else:
            model = resnet18()
    elif config == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        if pretrained:
            model = resnet50(weights=weights)
        else:
            model = resnet50()
    else:
        raise ValueError(f"Unknown config: {config}")

    return TASK_RESNET[dataset](model), weights.transforms()


def vit(config: str, dataset: str, pretrained: bool = True) -> Tuple[VisionTransformer, ImageClassification]:
    if config == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT
        if pretrained:
            model = vit_b_16(weights=weights)
        else:
            model = vit_b_16()
    else:
        raise ValueError(f"Unknown config: {config}")

    return TASK_VIT[dataset](model), weights.transforms()
