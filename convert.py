import torch.nn as nn
import snntorch as snn
import torch
import paralif
from snntorch import spikegen
from torchvision.models import ResNet

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
        train = spikegen.rate(x, num_steps=self.steps)
        train = torch.swapaxes(train, 0, 1)
        x = self.paralif(train)
        return torch.mean(x,1)

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
        x = self.encoder(x)
        train = spikegen.rate(x, num_steps=self.steps)
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

def convert(model: ResNet | LIFResNetDecoder | ParaLIFResNetDecoder, dataset: str, dest: str = "LIF") -> LIFResNetDecoder | ParaLIFResNetDecoder:
    print(f"VARIANT: {dest}")
    if dest == "LIF":
        if isinstance(model, ResNet):
            if dataset == "cifar10":
                return LIFResNetDecoder(model, num_classes=10)
            elif dataset == "cifar100":
                return LIFResNetDecoder(model, num_classes=100)
            elif dataset == "fashionMNIST":
                return LIFResNetDecoder(model, num_classes=10)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
    elif dest == "ParaLIF":
        if isinstance(model, ResNet):
            if dataset == "cifar10":
                return ParaLIFResNetDecoder(model, num_classes=10)
            elif dataset == "cifar100":
                return ParaLIFResNetDecoder(model, num_classes=100)
            elif dataset == "fashionMNIST":
                return ParaLIFResNetDecoder(model, num_classes=10)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
    else:
        raise ValueError(f"Unknown conversion destination: {dest}")

    return model
