from torchvision.models import ResNet
from enum import Enum, auto
from tabulate import tabulate
from convert import LIFResNetDecoder, ParaLIFResNetDecoder

class Depth(Enum):
    NONE = auto()
    LAYER1 = auto()
    LAYER2 = auto()
    LAYER3 = auto()
    ALL = auto()

def freeze(model: ResNet | LIFResNetDecoder | ParaLIFResNetDecoder, depth: Depth = Depth.ALL) -> ResNet | LIFResNetDecoder | ParaLIFResNetDecoder:
    if isinstance(model, ResNet):
        return resnet(model, depth)
    elif isinstance(model, LIFResNetDecoder) or isinstance(model, ParaLIFResNetDecoder):
        model.encoder = resnet(model.encoder, depth=Depth.LAYER3)
        return model
    else:
        raise ValueError(f"cannot freeze type {type(model).__name__}")


def resnet(model: ResNet, depth: Depth = Depth.ALL) -> ResNet:
    layers_to_freeze = {
        Depth.NONE: [],
        Depth.LAYER1: ["conv1", "bn1", "layer1"],
        Depth.LAYER2: ["conv1", "bn1", "layer1", "layer2"],
        Depth.LAYER3: ["conv1", "bn1", "layer1", "layer2", "layer3"],
        Depth.ALL: ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
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
