import torch
import os
import io
import upload
from convert import LIFResNetDecoder, ParaLIFResNetDecoder
from dataclasses import dataclass
from typing import Tuple
from torchvision.models import ResNet, VisionTransformer
from google.cloud import storage

CACHE = "./checkpoint"
PREFIX = "checkpoint"
BUCKET_NAME = "florians_results"


@dataclass
class Metadata:
    name: str
    epoch: int
    accuracy: float
    loss: float


def cache(model: torch.nn.Module, dataset: str, metadata: Metadata, variant: str = "", scratch: bool = False, gcs: bool = False):
    if scratch:
        dataset = f"{dataset}[scratch]"

    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    path = f"{CACHE}/{dataset}-{metadata.name}-checkpoint.pth"
    gcs_path = f"{PREFIX}/{dataset}-{metadata.name}-checkpoint.pth"
    if variant != "":
        path = f"{CACHE}/{dataset}-{variant}-{metadata.name}-checkpoint.pth"
        gcs_path = f"{PREFIX}/{dataset}-{variant}-{metadata.name}-checkpoint.pth"

    torch.save(checkpoint, path)

    if gcs:
        upload.gcs(path, gcs_path)


def load(model: ResNet | VisionTransformer | LIFResNetDecoder | ParaLIFResNetDecoder,
         name: str, dataset: str, variant: str = "", scratch: bool = False, gcs: bool = False) -> Tuple[bool, ResNet | VisionTransformer | LIFResNetDecoder | ParaLIFResNetDecoder, Metadata]:
    if scratch:
        dataset = f"{dataset}[scratch]"

    path = f"{CACHE}/{dataset}-{name}-checkpoint.pth"
    gcs_path = f"{PREFIX}/{dataset}-{name}-checkpoint.pth"
    if variant != "":
        path = f"{CACHE}/{dataset}-{variant}-{name}-checkpoint.pth"
        gcs_path = f"{PREFIX}/{dataset}-{variant}-{name}-checkpoint.pth"

    if gcs:
        print("-> loading from GCS")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            print("-> nothing found in GCS")
            return False, model, Metadata(name, 0, 0, float("inf"))

        checkpoint_data = blob.download_as_bytes()
        checkpoint = torch.load(io.BytesIO(checkpoint_data))
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint["metadata"]
        return True, model, metadata
    else:
        print("-> loading from disk")
        if not os.path.exists(path):
            print("-> nothing found on disk")
            return False, model, Metadata(name, 0, 0, float("inf"))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint["metadata"]
        return True, model, metadata
