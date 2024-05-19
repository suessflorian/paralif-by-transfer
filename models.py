import torch
from torchvision.models import ResNet, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models.resnet import ImageClassification
from typing import Tuple, Callable

def hone(classes: int) -> Callable[[ResNet], ResNet]:
    def focused(model: ResNet) -> ResNet:
        ins = model.fc.in_features
        model.fc = torch.nn.Linear(ins, classes)
        return model
    return focused

TASK = {
    "cifar100": hone(100),
    "cifar10": hone(10),
    "fashionMNIST": hone(10),
}

def resnet(config: str, dataset: str) -> Tuple[ResNet, ImageClassification]:
    print(f"ARCH: {config}")
    if config == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif config == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown config: {config}")

    return TASK[dataset](model), weights.transforms()
