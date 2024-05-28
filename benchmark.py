import torch
from typing import Callable

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def benchmark(
    model: torch.nn.Module,
    criterion: Callable,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
):
    model.train()
    model.to(device)

    (images, labels) = next(iter(train_loader))
    batch = images.shape[0]

    if device == "cuda":
        scaler = GradScaler()
    else:
        scaler = None


    for _ in range(1, epochs + 1):
        correct, total = 0, 0
        with tqdm(train_loader, unit="images", unit_scale=batch) as progress:
            for images, labels in progress:
                progress.set_description(f"Benchmark")
                images, labels = images.to(device), labels.to(device)
                if scaler:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                progress.set_postfix(train_accuracy=f"{(correct/total):.2f}")

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            with tqdm(test_loader, desc="Evaluation", unit="batch") as progress:
                for images, labels in progress:
                    images, labels = images.to(device), labels.to(device)
                    if scaler:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    progress.set_postfix(test_accuracy=(correct/total))
