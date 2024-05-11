import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from typing import Tuple

DATA = os.path.join(os.path.dirname(__file__), "data")


def cifar100(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]
    )

    train_dataset = datasets.CIFAR100(root=DATA, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=DATA, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
