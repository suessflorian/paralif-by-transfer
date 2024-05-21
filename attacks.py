import torch
import random
from art.attacks.evasion import FastGradientMethod, DeepFool, SquareAttack
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import copy
import csv

def sampled(loader: DataLoader, model: torch.nn.Module, device: str, sample: int = 300) -> DataLoader:
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

cifar10_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

cifar100_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

fashion_mnist_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def get_label(dataset, label):
    if dataset == "cifar10":
        return cifar10_labels[label]
    elif dataset == "cifar100":
        return cifar100_labels[label]
    elif dataset == "fashionMNIST":
        return fashion_mnist_labels[label]
    else:
        return "Unknown"

def denormalize(tensor):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    mean = np.array(imagenet_mean)
    std = np.array(imagenet_std)
    tensor = tensor.cpu().numpy().transpose((1, 2, 0))
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor

def plot(original, adv, label, dataset="cifar10"):
    original = denormalize(original)
    adv = denormalize(adv)

    similarity = ssim(
        original,
        adv,
        data_range=1.0,
        multichannel=True,
        channel_axis=2
    )

    label_name = get_label(dataset, label)

    _, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(original, interpolation='nearest')
    axes[0].set_title(f"Original Image\n(label={label_name})")
    axes[0].axis('off')

    axes[1].imshow(adv, interpolation='nearest')
    axes[1].set_title(f"Adversarial Image\n(SSIM={similarity:.4f})")
    axes[1].axis('off')

    plt.show()

def firstSSIM(original, adv):
    return ssim(
        denormalize(original[0]) ,
        denormalize(adv[0]),
        data_range=1.0,
        multichannel=True,
        channel_axis=2
    )


def find_clip_values(loader: DataLoader):
    min_val, max_val = float('inf'), float('-inf')
    for images, _ in tqdm(loader, desc="Finding Clip Values", unit="batch"):
        batch_min = torch.min(images)
        batch_max = torch.max(images)

        if batch_min < min_val:
            min_val = batch_min.item()
        if batch_max > max_val:
            max_val = batch_max.item()

    return min_val, max_val

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

    sampled_loader = sampled(loader, model, device, sample=100)

    classes = 100
    if dataset == "cifar10" or dataset == "fashionMNIST":
        classes = 10

    images, _ = next(iter(loader))
    input_shape = images.shape[1:]

    cpuModel = copy.deepcopy(model)
    cpuModel = cpuModel.to("cpu")

    classifier = PyTorchClassifier(
        model = cpuModel,
        loss = criterion,
        optimizer = optimizer,
        input_shape= input_shape,
        nb_classes= classes,
        clip_values=find_clip_values(loader)
    )

    model = model.to(device)

    results = []
    if attack == "fgsm":
        for epsilon in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.5, 1]:
            method = FastGradientMethod(estimator=classifier, eps=epsilon)
            if variant == "ParaLIF":
                num_steps = 5
                for _ in range(num_steps):
                    correct, total = 0, 0
                    ssim_sample = []
                    for images, labels in tqdm(sampled_loader, desc=f"fgsm(epsilon={epsilon})", unit="batch"):
                        adv = method.generate(x=images.cpu().numpy())
                        adv = torch.tensor(adv).to(device)
                        outputs = model(adv)
                        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                        total += labels.size(0)
                        ssim_sample.append(firstSSIM(images, adv))
                    accuracy = 100 * correct / total
                    perturbed_ssim = np.mean(ssim_sample)
                    results.append([epsilon, accuracy, perturbed_ssim])
            else:
                correct, total = 0, 0
                ssim_sample = []
                for images, labels in tqdm(sampled_loader, desc=f"fgsm(epsilon={epsilon})", unit="batch"):
                    adv = method.generate(x=images.cpu().numpy())
                    adv = torch.tensor(adv).to(device)
                    outputs = model(adv)
                    correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    total += labels.size(0)
                    ssim_sample.append(firstSSIM(images, adv))
                accuracy = 100 * correct / total
                perturbed_ssim = np.mean(ssim_sample)
                results.append([epsilon, accuracy, perturbed_ssim])

        persist(name, variant, dataset, "fgsm", header=["epsilon", "accuracy", "ssim"], results=results)
    elif attack == "deepfool":
        for epsilon in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.5, 1]:
            method = DeepFool(classifier=classifier, epsilon=epsilon, verbose=False)
            if variant == "ParaLIF":
                num_steps = 5
                for _ in range(num_steps):
                    correct, total = 0, 0
                    ssim_sample = []
                    for images, labels in tqdm(sampled_loader, desc=f"deepfool(Epsilon: {epsilon})"):
                        adv = method.generate(x=images.cpu().numpy())
                        adv = torch.tensor(adv).to(device)
                        outputs = model(adv)
                        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                        total += labels.size(0)
                        ssim_sample.append(firstSSIM(images, adv))
                    accuracy = 100 * correct / total
                    perturbed_ssim = np.mean(ssim_sample)
                    results.append([epsilon, accuracy, perturbed_ssim])
            else:
                correct, total = 0, 0
                ssim_sample = []
                for images, labels in tqdm(sampled_loader, desc=f"deepfool(Epsilon: {epsilon})"):
                    adv = method.generate(x=images.cpu().numpy())
                    adv = torch.tensor(adv).to(device)
                    outputs = model(adv)
                    correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    total += labels.size(0)
                    ssim_sample.append(firstSSIM(images, adv))
                accuracy = 100 * correct / total
                perturbed_ssim = np.mean(ssim_sample)
                results.append([epsilon, accuracy, perturbed_ssim])

        persist(name, variant, dataset, "deepfool", header=["epsilon", "accuracy", "ssim"], results=results)
    elif attack == "square@0.1":
        for iterations in [1, 2, 3, 5, 10, 15, 20, 25, 50, 100]:
            method = SquareAttack(estimator=classifier, max_iter=iterations, eps=0.1, verbose=False)

            if variant == "ParaLIF":
                num_steps = 5
                for _ in range(num_steps):
                    correct, total = 0, 0
                    ssim_sample = []
                    for images, labels in tqdm(sampled_loader, desc=f"square(Iter: {iterations})"):
                        adv = method.generate(x=images.cpu().numpy())
                        adv = torch.tensor(adv).to(device)
                        outputs = model(adv)
                        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                        total += labels.size(0)
                        ssim_sample.append(firstSSIM(images, adv))
                    accuracy = 100 * correct / total
                    perturbed_ssim = np.mean(ssim_sample)
                    print(f"Iterations: {iterations}, Accuracy: {accuracy}, SSIM: {perturbed_ssim}")
                    results.append([iterations, accuracy, perturbed_ssim])
            else:
                correct, total = 0, 0
                ssim_sample = []
                for images, labels in tqdm(sampled_loader, desc=f"square(Iter: {iterations})"):
                    adv = method.generate(x=images.cpu().numpy())
                    adv = torch.tensor(adv).to(device)
                    outputs = model(adv)
                    correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    total += labels.size(0)
                    ssim_sample.append(firstSSIM(images, adv))
                accuracy = 100 * correct / total
                perturbed_ssim = np.mean(ssim_sample)
                print(f"Iterations: {iterations}, Accuracy: {accuracy}, SSIM: {perturbed_ssim}")
                results.append([iterations, accuracy, perturbed_ssim])

        persist(name, variant, dataset, "square@0.1", header=["max_iterations", "accuracy", "ssim"], results=results)
    else:
        raise ValueError(f"Unknown attack: {attack}")

def persist(model: str, variant: str, dataset: str, attack: str, header = [], results = []):
    filename = f"./results/{dataset}-{variant}-{model}-{attack}.csv"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

    print(f"...results saved to {filename}")
