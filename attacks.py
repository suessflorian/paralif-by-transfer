import torch
import random
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy
import csv

def sampled(loader: DataLoader, model: torch.nn.Module, device: str, sample: int = 200) -> DataLoader:
    model = model.to(device)
    model.eval()

    all_images = []
    all_labels = []

    for images, labels in tqdm(loader, desc=f"Creating sampled loader, finding candidates", unit="batch"):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct_indices = (predicted == labels).nonzero(as_tuple=True)[0]
            all_images.append(images[correct_indices])
            all_labels.append(labels[correct_indices])

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    indices = random.sample(range(len(all_images)), min(sample, len(all_images)))
    sample_images = all_images[indices]
    sample_labels = all_labels[indices]

    print(f"Randomly sampled {sample} correctly classified images from {len(all_images)}...")
    return DataLoader(TensorDataset(sample_images, sample_labels), batch_size=10)

def perform(
        model: torch.nn.Module,
        name: str,
        variant: str,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        attack: str = "fgsm",
        dataset: str = "cifar10",
        device: str = "mps",
    ):

    sampled_loader = sampled(loader, model, device)

    classes = 100
    if dataset == "cifar10" or dataset == "fashionMNIST":
        classes = 10
            
    images, _ = next(iter(loader))
    input_shape = images.shape[1:]

    cpuModel = copy.deepcopy(model)
    cpuModel = cpuModel.to("cpu")

    if attack == "fgsm":
        classifier = PyTorchClassifier(
            model = cpuModel,
            loss = criterion,
            optimizer = optimizer,
            input_shape= input_shape,
            nb_classes= classes,
        )

        model = model.to(device)
        results = []

        for epsilon in [
            0.0005,
            0.001,
            0.005,
            0.01,
            0.1,
            0.3,
            0.5,
            0.7,
            1.0,
        ]:
            method = FastGradientMethod(estimator=classifier, eps=epsilon)

            correct = 0
            total = 0
            for images, labels in tqdm(sampled_loader, desc=f"FGSM(epsilon={epsilon})", unit="batch"):
                adv = method.generate(x=images.cpu().numpy())
                adv = torch.tensor(adv).to(device)
                outputs = model(adv)
                correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total += labels.size(0)
            accuracy = 100 * correct / total
            results.append([epsilon, accuracy])

        persist(name, variant, dataset, "FGSM", header=["epsilon", "accuracy"], results=results)
    else:
        raise ValueError(f"Unknown attack: {attack}")

def persist(model: str, variant: str, dataset: str, attack: str, header = [], results = []):
    filename = f"./results/{dataset}-{model}-{attack}.csv"
    if variant != "":
        filename = f"./results/{dataset}-{variant}-{model}-{attack}.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

    print(f"...results saved to {filename}")
