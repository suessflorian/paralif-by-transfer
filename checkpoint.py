import torch
import os
from dataclasses import dataclass
from typing import Tuple

CACHE = "./checkpoint"


@dataclass
class Metadata:
    name: str
    epoch: int
    accuracy: float
    loss: float


def cache(model: torch.nn.Module, metadata: Metadata):
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(checkpoint, f"{CACHE}/{metadata.name}-checkpoint.pth")
    print("-> checkpoint saved")


def load(model: torch.nn.Module, name: str) -> Tuple[torch.nn.Module, Metadata]:
    path = f"{CACHE}/{name}-checkpoint.pth"
    if not os.path.exists(path):
        return model, Metadata(name, 0, 0, float("inf"))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = checkpoint["metadata"]

    return model, metadata
