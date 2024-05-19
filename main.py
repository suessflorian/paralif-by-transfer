import models
import freeze
import convert
import checkpoint
import torch
import data
import argparse
from typing import Tuple, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(69)

parser = argparse.ArgumentParser(description="Exploring CIFAR-100 S-NN models")
subparsers = parser.add_subparsers(dest="command", help="sub-command help")

train_parser = subparsers.add_parser("train", help="training models")
train_parser.add_argument("--epochs", type=int, default=200, help="number of epochs to train for")
train_parser.add_argument("--batch", type=int, default=64, help="input batch size for training")
train_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to train for")
train_parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
train_parser.add_argument("--model", type=str, default="resnet18", help="the architecture")
train_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
train_parser.add_argument("--convert", type=bool, default=False, help="if the model should be converted to snn")

test_parser = subparsers.add_parser("test", help="testing models")
test_parser.add_argument("--model", type=str, default="resnet18", help="the architecture")
test_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
test_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to train for")

args = parser.parse_args()

def train(
    model: torch.nn.Module,
    name: str,
    dataset: str,
    converted: bool,
    criterion: Callable,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cpu",
    metadata: checkpoint.Metadata = checkpoint.Metadata("resnet18", 0, 0, float("inf")),
):
    model.train()
    best_accuracy, best_loss = metadata.accuracy, metadata.loss
    for i in range(epochs):
        if (i) % 5 == 0:
            loss, accuracy = evaluate(model, criterion, test_loader, device)
            if accuracy >= best_accuracy:
                checkpoint.cache(
                    model,
                    dataset,
                    converted,
                    checkpoint.Metadata(
                        name=name, epoch=i + metadata.epoch, accuracy=best_accuracy, loss=best_loss
                    ),
                )
            print(f"Epoch: {i + metadata.epoch}, Loss: {loss}, Accuracy: {accuracy}")
        for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(
    model: torch.nn.Module,
    criterion: Callable,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluation", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return average_loss, accuracy


if not args.command:
    parser.print_help()
    raise ValueError("no command specified")
elif args.command == "train":
    model, preprocess = models.resnet(args.model, args.dataset)
    loaded, vanilla, metadata = checkpoint.load(model, args.model, args.dataset, converted=False)

    if args.convert:
        if not loaded:
            raise ValueError("must train vanilla model first... abort")
        model = convert.convert(vanilla, args.dataset)
        # NOTE: check if we already have a converted model checkpointed ready to continue on
        _, model, metadata = checkpoint.load(model, args.model, args.dataset, converted=True)
    else:
        model = vanilla

    model = freeze.freeze(model, depth=freeze.Depth.LAYER3)
    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    train_loader, test_loader = data.loader(args.dataset, preprocess, args.batch)

    train(
        model,
        args.model,
        args.dataset,
        args.convert,
        criterion,
        train_loader,
        test_loader,
        optimizer,
        args.epochs,
        args.device,
        metadata,
    )
elif args.command == "test":
    model, preprocess = models.resnet(args.model, args.dataset)
    if args.convert:
        model = convert.convert(model, args.dataset)
    loaded, model, metadata = checkpoint.load(model, args.model, args.dataset, converted=args.convert)

    if not loaded:
        raise ValueError(f"no trained {args.model} for {args.dataset}... abort")

    model = freeze.freeze(model, depth=freeze.Depth.ALL)
    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = data.loader(args.dataset, preprocess, 64)

    loss, accuracy = evaluate(
        model,
        criterion,
        test_loader,
        args.device,
    )

    print(f"Epoch: {metadata.epoch}, Loss: {loss}, Accuracy: {accuracy}")
else:
    raise ValueError("invalid command")
