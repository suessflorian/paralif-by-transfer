import torch
from torchvision.models import ResNet, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models.resnet import ImageClassification
from typing import Tuple
from enum import Enum, auto
from tabulate import tabulate


def resnet(config: str) -> Tuple[ResNet, ImageClassification]:
    print(f"...configuration chosen: {config}")
    if config == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model = freeze(model, depth=FreezeDepth.LAYER3)
        return cifar100(model), weights.transforms()
    if config == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model = freeze(model, depth=FreezeDepth.LAYER3)
        return cifar100(model), weights.transforms()
    else:
        raise ValueError(f"Unknown config: {config}")

class FreezeDepth(Enum):
    NONE = auto()
    LAYER1 = auto()
    LAYER2 = auto()
    LAYER3 = auto()
    ALL = auto()

def freeze(model: ResNet, depth: FreezeDepth = FreezeDepth.ALL) -> ResNet:
    layers_to_freeze = {
        FreezeDepth.NONE: [],
        FreezeDepth.LAYER1: ["conv1", "bn1", "layer1"],
        FreezeDepth.LAYER2: ["conv1", "bn1", "layer1", "layer2"],
        FreezeDepth.LAYER3: ["conv1", "bn1", "layer1", "layer2", "layer3"],
        FreezeDepth.ALL: ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    }
    depth_to_layers = {tuple(v): k for k, v in layers_to_freeze.items()}
    
    prefixes = layers_to_freeze[depth]
    table = []
    total_params = 0

    for name, param in model.named_parameters():
        layer_depth = "NONE"
        for prefix, freeze_depth in depth_to_layers.items():
            if any(name.startswith(p) for p in prefix):
                layer_depth = freeze_depth.name
                break
        
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = False
            status = "FROZEN"
        else:
            param.requires_grad = True
            status = "TRAINABLE"
        
        param_count = param.numel()
        table.append([layer_depth, name,  status, param_count])
        total_params += param_count
    
    headers = ["Parameter Name", "Depth", "Number of Parameters", "Status"]
    print(tabulate(table, headers=headers))
    print(f"==> Total parameters: {total_params}")
    
    return model


def cifar100(model: ResNet) -> ResNet:
    ins = model.fc.in_features
    model.fc = torch.nn.Linear(ins, 100)
    return model

def fashionMNIST(model: ResNet) -> ResNet:
    ins = model.fc.in_features
    model.fc = torch.nn.Linear(ins, 10)
    return model
