import torch
import os
from convert import LIFResNetDecoder, ParaLIFResNetDecoder
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


def cache(model: torch.nn.Module, dataset: str, metadata: Metadata, variant: str = ""):
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    path = f"{CACHE}/{dataset}-{metadata.name}-checkpoint.pth"
    if variant != "":
        path = f"{CACHE}/{dataset}-{variant}-{metadata.name}-checkpoint.pth"
    torch.save(checkpoint, path)
    print("-> checkpoint saved")


def load(model: ResNet | LIFResNetDecoder | ParaLIFResNetDecoder, name: str, dataset: str, variant: str = "") -> Tuple[bool, ResNet | LIFResNetDecoder | ParaLIFResNetDecoder, Metadata]:
    path = f"{CACHE}/{dataset}-{name}-checkpoint.pth"
    if variant != "":
        path = f"{CACHE}/{dataset}-{variant}-{name}-checkpoint.pth"

    if not os.path.exists(path):
        return False, model, Metadata(name, 0, 0, float("inf"))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = checkpoint["metadata"]

    return True, model, metadata
