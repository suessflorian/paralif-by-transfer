from torchvision.models import ResNet, VisionTransformer
from enum import Enum, auto
from tabulate import tabulate
from convert import LIFResNetDecoder, LIFViTDecoder, ParaLIFResNetDecoder

class Depth(Enum):
    NONE = auto()
    ONE = auto()
    TWO = auto()
    THREE = auto()
    ALL = auto()

def freeze(model: ResNet | VisionTransformer | LIFResNetDecoder | ParaLIFResNetDecoder, depth: Depth = Depth.ALL) -> ResNet | VisionTransformer | LIFResNetDecoder | ParaLIFResNetDecoder:
    if isinstance(model, ResNet):
        return resnet(model, depth)
    elif isinstance(model, LIFResNetDecoder) or isinstance(model, ParaLIFResNetDecoder):
        model.encoder = resnet(model.encoder, depth=Depth.THREE)
        return model
    elif isinstance(model, LIFViTDecoder):
        model.encoder = vit(model.encoder, depth=Depth.THREE)
        return model
    elif isinstance(model, VisionTransformer):
        return vit(model, depth)
    else:
        raise ValueError(f"cannot freeze type {type(model).__name__}")

def vit(model: VisionTransformer, depth: Depth = Depth.ALL) -> VisionTransformer:
    layers_to_freeze = {
        Depth.NONE: [],
        Depth.ONE: [
            "conv_proj",
            "encoder.pos_embedding",
            "encoder.layers.encoder_layer_0.",
            "encoder.layers.encoder_layer_1.",
            "encoder.layers.encoder_layer_2.",
        ],
        Depth.TWO: [
            "conv_proj",
            "encoder.pos_embedding",
            "encoder.layers.encoder_layer_0.",
            "encoder.layers.encoder_layer_1.",
            "encoder.layers.encoder_layer_2.",
            "encoder.layers.encoder_layer_3.",
            "encoder.layers.encoder_layer_4.",
            "encoder.layers.encoder_layer_5.",
        ],
        Depth.THREE: [
            "conv_proj",
            "encoder.pos_embedding",
            "encoder.layers.encoder_layer_0.",
            "encoder.layers.encoder_layer_1.",
            "encoder.layers.encoder_layer_2.",
            "encoder.layers.encoder_layer_3.",
            "encoder.layers.encoder_layer_4.",
            "encoder.layers.encoder_layer_5.",
            "encoder.layers.encoder_layer_6.",
            "encoder.layers.encoder_layer_7.",
            "encoder.layers.encoder_layer_8.",
        ],
        Depth.ALL: ["encoder"],
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

def resnet(model: ResNet, depth: Depth = Depth.ALL) -> ResNet:
    layers_to_freeze = {
        Depth.NONE: [],
        Depth.ONE: ["conv1", "bn1", "layer1"],
        Depth.TWO: ["conv1", "bn1", "layer1", "layer2"],
        Depth.THREE: ["conv1", "bn1", "layer1", "layer2", "layer3"],
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
