import torch.nn as nn
import snntorch as snn
import torch
import encoding
import paralif
from torchvision.models import ResNet, VisionTransformer

def min_max_norm(batch: torch.Tensor) -> torch.Tensor:
    min_vals = batch.amin(dim=1, keepdim=True)
    max_vals = batch.amax(dim=1, keepdim=True)
    normalized_batch = (batch - min_vals) / (max_vals - min_vals + 1e-8) # NOTE: eps here to avoid div/zero.
    return normalized_batch

class ParaLIFResNetDecoder(nn.Module):
    def __init__(self, model: ResNet, num_classes: int):
        super(ParaLIFResNetDecoder, self).__init__()
        self.encoder = model
        fm = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.steps = 20
        self.paralif = paralif.ParaLIF(fm, num_classes, "mps", "SB", tau_mem=0.02, tau_syn=0.02)

    def forward(self, x):
        x = self.encoder(x)
        train = encoding.rate(min_max_norm(x), num_steps=self.steps)
        train = torch.swapaxes(train, 0, 1)
        x = self.paralif(train)
        return torch.mean(x,1)

    def to(self, device):
        self.encoder.to(device)
        self.paralif.to(device)
        return self

class LIFResNetDecoder(nn.Module):
    def __init__(self, model: ResNet, num_classes: int):
        super(LIFResNetDecoder, self).__init__()
        self.encoder = model
        fm = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.steps = 20
        self.fc1 = nn.Linear(fm, 256)
        self.lif = snn.Leaky(beta=0.95)
        self.mem = self.lif.init_leaky()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        self.mem = self.lif.init_leaky()

        x = self.encoder(x)
        train = encoding.rate(min_max_norm(x), num_steps=self.steps)
        for i in range(self.steps):
            spike = train[i]
            spike = self.fc1(spike)
            spike, self.mem = self.lif(spike, self.mem)
            spike = self.fc2(spike)
        return x

    def to(self, device):
        self.encoder.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.mem = self.mem.to(device)
        self.lif.to(device)
        return self

class LIFViTDecoder(nn.Module):
    def __init__(self, model: VisionTransformer, num_classes: int):
        super(LIFViTDecoder, self).__init__()
        self.encoder = model
        fm = self.encoder.hidden_dim
        self.encoder.heads = torch.nn.Identity()

        self.steps = 20
        self.fc1 = nn.Linear(fm, 256)
        self.lif = snn.Leaky(beta=0.95)
        self.mem = self.lif.init_leaky()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        self.mem = self.lif.init_leaky()

        x = self.encoder(x)
        train = encoding.rate(min_max_norm(x), num_steps=self.steps)
        for i in range(self.steps):
            spike = train[i]
            spike = self.fc1(spike)
            spike, self.mem = self.lif(spike, self.mem)
            spike = self.fc2(spike)
        return x

    def to(self, device):
        self.encoder.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.mem = self.mem.to(device)
        self.lif.to(device)
        return self

def convert(model: ResNet | VisionTransformer | LIFResNetDecoder | ParaLIFResNetDecoder, dataset: str, dest: str = "LIF") -> LIFResNetDecoder | ParaLIFResNetDecoder:
    if isinstance(model, ResNet):
        if dest == "LIF":
            return LIFResNetDecoder(model, num_classes=100 if dataset == "cifar100" else 10)
        elif dest == "ParaLIF":
            return ParaLIFResNetDecoder(model, num_classes=100 if dataset == "cifar100" else 10)
        else:
            raise ValueError(f"Unknown conversion destination: {dest}")
    if isinstance(model, VisionTransformer):
        if dest == "LIF":
            return LIFViTDecoder(model, num_classes=100 if dataset == "cifar100" else 10)
        elif dest == "ParaLIF":
            raise NotImplementedError("ParaLIF convert not implemented for ViT")
        else:
            raise ValueError(f"Unknown conversion destination: {dest}")

    return model
