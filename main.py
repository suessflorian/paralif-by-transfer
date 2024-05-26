import models
import freeze
import convert
import checkpoint
import torch
import data
import attacks
import results
import upload
import argparse
import os
import csv
from typing import Tuple, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(69)

parser = argparse.ArgumentParser(description="Exploring CIFAR-100 S-NN models")
subparsers = parser.add_subparsers(dest="command")

# NOTE: resnet trainer
rtrainer_parser = subparsers.add_parser("rtrainer", help="training resnet models")
rtrainer_parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train for")
rtrainer_parser.add_argument("--batch", type=int, default=64, help="input batch size for training")
rtrainer_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to train for")
rtrainer_parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
rtrainer_parser.add_argument("--arch", type=str, default="resnet18", help="the architecture")
rtrainer_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
rtrainer_parser.add_argument("--lif", action="store_true", help="if the model should be converted a lif decoder variant")
rtrainer_parser.add_argument("--paralif", action="store_true", help="if the model should be converted to a paralif decoder variant")
rtrainer_parser.add_argument("--gcs", type=bool, default=False, help="indication of whether source/store models gcsly or not")

# NOTE: vision transformer trainer
vtrainer_parser = subparsers.add_parser("vtrainer", help="training vision transformer models")
vtrainer_parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for")
vtrainer_parser.add_argument("--warmup", type=int, default=2, help="number of warmup epochs")
vtrainer_parser.add_argument("--warmup-decay", type=int, default=0.033, help="warmup decay factor")
vtrainer_parser.add_argument("--weight-decay", type=int, default=0.3, help="warmup decay factor")
vtrainer_parser.add_argument("--batch", type=int, default=128, help="input batch size for training")
vtrainer_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to train for")
vtrainer_parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
vtrainer_parser.add_argument("--arch", type=str, default="vit_b_16", help="the architecture")
vtrainer_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
vtrainer_parser.add_argument("--gcs", type=bool, default=False, help="indication of whether source/store models from gcs or not")

test_parser = subparsers.add_parser("test", help="testing models")
test_parser.add_argument("--model", type=str, default="resnet18", help="the architecture")
test_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
test_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to train for")
test_parser.add_argument("--lif", action="store_true", help="the lif variant of the model")
test_parser.add_argument("--paralif", action="store_true", help="the paralif variant of the model")
test_parser.add_argument("--gcs", type=bool, default=False, help="indication of whether source/store models from gcs or not")

scratch_parser = subparsers.add_parser("scratch", help="testing models from scratch")
scratch_parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train for")
scratch_parser.add_argument("--batch", type=int, default=64, help="input batch size for training")
scratch_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to train for")
scratch_parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
scratch_parser.add_argument("--model", type=str, default="resnet18", help="the architecture")
scratch_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
scratch_parser.add_argument("--eval", action="store_true", help="if you want to evaluate the model instead of train")
scratch_parser.add_argument("--lif", action="store_true", help="if the model should be converted a lif decoder variant")
scratch_parser.add_argument("--paralif", action="store_true", help="if the model should be converted to a paralif decoder variant")
scratch_parser.add_argument("--gcs", type=bool, default=False, help="indication of whether source/store models from gcs or not")

attack_parser = subparsers.add_parser("attack", help="testing robustness of models")
attack_parser.add_argument("--model", type=str, default="resnet18", help="the architecture")
attack_parser.add_argument("--device", type=str, default="mps", help="device to lay tensor work over")
attack_parser.add_argument("--dataset", type=str, default="cifar100", help="dataset to use for attacking")
attack_parser.add_argument("--lif", action="store_true", help="the lif variant of the model")
attack_parser.add_argument("--paralif", action="store_true", help="the paralif variant of the model")
attack_parser.add_argument("--attack", type=str, default="fgsm", help="the type of attack to perform")
attack_parser.add_argument("--gcs", type=bool, default=False, help="indication of whether source/store models from gcs or not")

results_parser = subparsers.add_parser("results", help="rendering the diagrams of the results gathered")
results_subparser = results_parser.add_subparsers(dest="type", help="which type of results to show")

attack_results_parser = results_subparser.add_parser("attack", help="rendering the diagrams of the attack results")
attack_results_parser.add_argument("--attack", type=str, default="fgsm", help="which attack robustness results graphed")
attack_results_parser.add_argument("--dataset", type=str, default="cifar100", help="which dataset to show attack results for")

scratch_results_parser = results_subparser.add_parser("scratch", help="rendering the diagrams of the transfer learning effectiveness results")
scratch_results_parser.add_argument("--dataset", type=str, default="cifar100", help="which dataset to show for")

training_results_parser = results_subparser.add_parser("training", help="rendering the diagrams relevant to training")
training_results_parser.add_argument("--dataset", type=str, default="cifar100", help="which dataset to show training results for")

args = parser.parse_args()

if args.command != "results" and args.command != "vtrainer" and (args.lif and args.paralif):
    raise ValueError("cannot convert to both LIF and ParaLIF... abort")

def rtrain(
    model: torch.nn.Module,
    name: str,
    dataset: str,
    criterion: Callable,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    metadata: checkpoint.Metadata,
    variant: str = "",
    device: str = "cpu",
    gcs: bool = True,
):
    model.train()
    best_accuracy, best_loss = metadata.accuracy, metadata.loss

    report = f"./results/train/{dataset}-{variant}-{name}.csv"
    report_gcs = f"results/train/{dataset}-{variant}-{name}.csv"

    def write_to_csv(path, data):
        if not os.path.exists(path):
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "loss", "accuracy"])

        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    best_loss, best_accuracy = evaluate(model, criterion, test_loader, device)

    if metadata.epoch == 0:
        write_to_csv(report, [0, best_loss, best_accuracy])
        if gcs:
            upload.gcs(report, report_gcs)
        if variant == "ParaLIF":
            for _ in range(3):
                loss, accuracy = evaluate(model, criterion, test_loader, device)
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                write_to_csv(report, [0, loss, accuracy])
                if gcs:
                    upload.gcs(report, report_gcs)

    print(f"Current: {metadata.epoch}, Loss: {best_loss}, Accuracy: {best_accuracy}")
    for i in range(1, epochs + 1):
        for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if variant == "ParaLIF":
            num_runs = 5
            total_loss, total_accuracy = 0, 0
            for _ in range(num_runs):
                loss, accuracy = evaluate(model, criterion, test_loader, device)
                total_loss += loss
                total_accuracy += accuracy
                write_to_csv(report, [i + metadata.epoch, loss, accuracy])
                if gcs:
                    upload.gcs(report, report_gcs)

            avg_loss = total_loss / num_runs
            avg_accuracy = total_accuracy / num_runs

            if avg_loss >= best_accuracy:
                best_accuracy = avg_accuracy
                best_loss = avg_loss
                checkpoint.cache(
                    model,
                    dataset,
                    checkpoint.Metadata(
                        name=name, epoch=i + metadata.epoch, accuracy=best_accuracy, loss=best_loss
                    ),
                    variant=variant,
                    gcs=gcs,
                )
            print(f"Epoch: {i + metadata.epoch}, Avg_Loss: {avg_loss}, Avg_Accuracy: {avg_accuracy}")
        else:
            loss, accuracy = evaluate(model, criterion, test_loader, device)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_loss = loss
                checkpoint.cache(
                    model,
                    dataset,
                    checkpoint.Metadata(
                        name=name, epoch=i + metadata.epoch, accuracy=best_accuracy, loss=best_loss
                    ),
                    variant=variant,
                    gcs=gcs,
                )
            print(f"Epoch: {i + metadata.epoch}, Loss: {loss}, Accuracy: {accuracy}")

            write_to_csv(report, [i + metadata.epoch, loss, accuracy])
            if gcs:
                upload.gcs(report, report_gcs)

def vtrain(
    model: torch.nn.Module,
    name: str,
    dataset: str,
    criterion: Callable,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    metadata: checkpoint.Metadata,
    variant: str = "",
    device: str = "cpu",
    gcs: bool = True,
):
    model.train()
    best_accuracy, best_loss = metadata.accuracy, metadata.loss

    report = f"./results/train/{dataset}-{variant}-{name}.csv"
    report_gcs = f"results/train/{dataset}-{variant}-{name}.csv"

    def write_to_csv(path, data):
        if not os.path.exists(path):
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "loss", "accuracy"])

        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    best_loss, best_accuracy = evaluate(model, criterion, test_loader, device)

    if metadata.epoch == 0:
        write_to_csv(report, [0, best_loss, best_accuracy])
        if gcs:
            upload.gcs(report, report_gcs)
        if variant == "ParaLIF":
            for _ in range(3):
                loss, accuracy = evaluate(model, criterion, test_loader, device)
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                write_to_csv(report, [0, loss, accuracy])
                if gcs:
                    upload.gcs(report, report_gcs)

    for i in range(1, epochs + 1):
        correct, total = 0, 0
        with tqdm(train_loader, unit="batch") as progress:
            for images, labels in progress:
                progress.set_description(f"Epoch {i + metadata.epoch}")
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_postfix(train_accuracy=f"{(correct/total):.2f}")

        if variant == "ParaLIF":
            num_runs = 5
            total_loss, total_accuracy = 0, 0
            for _ in range(num_runs):
                loss, accuracy = evaluate(model, criterion, test_loader, device)
                total_loss += loss
                total_accuracy += accuracy
                write_to_csv(report, [i + metadata.epoch, loss, accuracy])
                if gcs:
                    upload.gcs(report, report_gcs)

            avg_loss = total_loss / num_runs
            avg_accuracy = total_accuracy / num_runs

            if avg_loss >= best_accuracy:
                best_accuracy = avg_accuracy
                best_loss = avg_loss
                checkpoint.cache(
                    model,
                    dataset,
                    checkpoint.Metadata(
                        name=name, epoch=i + metadata.epoch, accuracy=best_accuracy, loss=best_loss
                    ),
                    variant=variant,
                    gcs=gcs,
                )
            print(f"Epoch: {i + metadata.epoch}, Avg_Loss: {avg_loss}, Avg_Accuracy: {avg_accuracy}")
        else:
            loss, accuracy = evaluate(model, criterion, test_loader, device)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_loss = loss
                checkpoint.cache(
                    model,
                    dataset,
                    checkpoint.Metadata(
                        name=name, epoch=i + metadata.epoch, accuracy=best_accuracy, loss=best_loss
                    ),
                    variant=variant,
                    gcs=gcs,
                )
            print(f"Epoch: {i + metadata.epoch}, Loss: {loss}, Accuracy: {accuracy}")

            write_to_csv(report, [i + metadata.epoch, loss, accuracy])
            if gcs:
                upload.gcs(report, report_gcs)

def evaluate(
    model: torch.nn.Module,
    criterion: Callable,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        with tqdm(data_loader, desc="Evaluation", unit="batch") as progress:
            for images, labels in progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress.set_postfix(test_accuracy=(correct/total))

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return average_loss, accuracy


if not args.command:
    parser.print_help()
    raise ValueError("no command specified")
elif args.command == "rtrainer":
    model, preprocess = models.resnet(args.arch, args.dataset)
    loaded, vanilla, metadata = checkpoint.load(model, args.arch, args.dataset, gcs=args.gcs)

    if args.lif or args.paralif:
        if not loaded:
            raise ValueError("must train vanilla model first... abort")
        model = convert.convert(vanilla, args.dataset, dest="LIF" if args.lif else "ParaLIF")
        # NOTE: check if we already have a converted model checkpointed ready to continue on
        loaded, model, metadata = checkpoint.load(model, args.arch, args.dataset, variant="LIF" if args.lif else "ParaLIF", gcs=args.gcs)
        print("training LIF decoder model...")
    else:
        print("training vanilla model...")
        model = vanilla

    model = freeze.freeze(model, depth=freeze.Depth.THREE)
    model = model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    train_loader, test_loader = data.loader(args.dataset, preprocess, args.batch)

    rtrain(
        model,
        args.arch,
        args.dataset,
        criterion,
        train_loader,
        test_loader,
        optimizer,
        args.epochs,
        metadata,
        variant="LIF" if args.lif else "ParaLIF" if args.paralif else "",
        device=args.device,
        gcs=args.gcs,
    )
elif args.command == "vtrainer":
    model, preprocess = models.vit(args.arch, args.dataset)
    loaded, vanilla, metadata = checkpoint.load(model, args.arch, args.dataset, gcs= args.gcs)

    model = freeze.freeze(model, depth=freeze.Depth.THREE)
    model = model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    train_loader, test_loader = data.loader(args.dataset, preprocess, args.batch)

    vtrain(
        model,
        args.arch,
        args.dataset,
        criterion,
        train_loader,
        test_loader,
        optimizer,
        args.epochs,
        metadata,
        variant="",
        device=args.device,
        gcs=args.gcs,
    )
elif args.command == "scratch":
    model, preprocess = models.resnet(args.model, args.dataset, pretrained=False)
    if args.lif or args.paralif:
        model = convert.convert(model, args.dataset, dest="LIF" if args.lif else "ParaLIF")
    loaded, model, metadata = checkpoint.load(model, args.model, args.dataset, variant="LIF" if args.lif else "ParaLIF" if args.paralif else "", scratch=True, gcs=args.gcs)
    model = model.to(args.device)

    train_loader, test_loader = data.loader(args.dataset, preprocess, args.batch)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    rtrain(
        model,
        args.model,
        f"{args.dataset}[scratch]",
        criterion,
        train_loader,
        test_loader,
        optimizer,
        args.epochs,
        metadata,
        variant="LIF" if args.lif else "ParaLIF" if args.paralif else "",
        device=args.device,
        gcs=args.gcs,
    )
elif args.command == "test":
    model, preprocess = models.resnet(args.model, args.dataset)
    if args.lif or args.paralif:
        model = convert.convert(model, args.dataset, dest="LIF" if args.lif else "ParaLIF")
    loaded, model, metadata = checkpoint.load(model, args.model, args.dataset, variant="LIF" if args.lif else "ParaLIF" if args.paralif else "", gcs=args.gcs)
    if not loaded:
        raise ValueError(f"no trained {args.model} for {args.dataset}... abort")


    model = model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = data.loader(args.dataset, preprocess, 64)

    loss, accuracy = evaluate(
        model,
        criterion,
        test_loader,
        args.device,
    )

    print(f"Epoch: {metadata.epoch}, Loss: {loss}, Accuracy: {accuracy}")
elif args.command == "attack":
    model, preprocess = models.resnet(args.model, args.dataset)
    if args.lif or args.paralif:
        model = convert.convert(model, args.dataset, dest="LIF" if args.lif else "ParaLIF")
    loaded, model, metadata = checkpoint.load(model, args.model, args.dataset, variant="LIF" if args.lif else "ParaLIF" if args.paralif else "", gcs=args.gcs)

    if not loaded:
        raise ValueError(f"no trained {args.model} for {args.dataset}... abort")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())
    _, test_loader = data.loader(args.dataset, preprocess, 64)

    model = model.to(device=args.device)
    attacks.perform(
        model, args.model, "LIF" if args.lif else "ParaLIF" if args.paralif else "",
        criterion,
        optimizer,
        test_loader,
        attack=args.attack,
        dataset=args.dataset,
        device=args.device,
    )
elif args.command == "results":
    if args.type == "training":
        results.plot_training(args.dataset)
    if args.type == "scratch":
        results.transfer_learning(args.dataset)
    elif args.type == "attack":
        results.plot_attack(args.dataset, args.attack)
    else:
        raise ValueError("invalid results type")

else:
    raise ValueError("invalid command")
