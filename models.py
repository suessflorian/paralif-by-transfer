import torch
from torchvision.models import ResNet, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights


def resnet(config: str) -> ResNet:
    if config == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # model = freeze(model)
        return cifar100(model)
    if config == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = freeze(model)
        return cifar100(model)
    else:
        raise ValueError(f"Unknown config: {config}")


def freeze(model: ResNet) -> ResNet:
    for param in model.parameters():
        param.requires_grad = False
    return model


def cifar100(model: ResNet) -> ResNet:
    ins = model.fc.in_features
    model.fc = torch.nn.Linear(ins, 100)
    return model
