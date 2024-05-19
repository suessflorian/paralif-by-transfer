import torch
import os
from convert import LIFResNetDecoder
from dataclasses import dataclass
from typing import Tuple
from torchvision.models import ResNet

CACHE = "./checkpoint"


@dataclass
class Metadata:
    name: str
    epoch: int
    accuracy: float
    loss: float


def cache(model: torch.nn.Module, dataset: str, converted: bool, metadata: Metadata):
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    path = f"{CACHE}/{dataset}-{metadata.name}-checkpoint.pth"
    if converted:
        path = f"{CACHE}/{dataset}-lif-{metadata.name}-checkpoint.pth"
    torch.save(checkpoint, path)
    print("-> checkpoint saved")


def load(model: ResNet | LIFResNetDecoder, name: str, dataset: str, converted: bool = False) -> Tuple[bool, ResNet | LIFResNetDecoder, Metadata]:
    path = f"{CACHE}/{dataset}-{name}-checkpoint.pth"
    if converted:
        path = f"{CACHE}/{dataset}-lif-{name}-checkpoint.pth"

    if not os.path.exists(path):
        return False, model, Metadata(name, 0, 0, float("inf"))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = checkpoint["metadata"]

    return True, model, metadata
