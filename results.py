import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_training(dataset: str):
    results_dir = "./results/train"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith(dataset + "-") and f.endswith(".csv")]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for csv_file in csv_files:
        file_path = os.path.join(results_dir, csv_file)
        data = pd.read_csv(file_path)

        base_name = os.path.splitext(csv_file)[0]
        _, variant, model = base_name.split("-", 2)

        label = f"{model} ({variant})" if variant != "" else model
        if variant == "ParaLIF":
            grouped_data = data.groupby('epoch')['accuracy'].agg(['min', 'max'])
            if "resnet18" in model:
                ax1.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=f'Accuracy - {label}')
            elif "resnet50" in model:
                ax2.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=f'Accuracy - {label}')
        else:
            if "resnet18" in model:
                ax1.plot(data['epoch'], data['accuracy'], label=f'Accuracy - {label}')
            elif "resnet50" in model:
                ax2.plot(data['epoch'], data['accuracy'], label=f'Accuracy - {label}')

    ax1.set_title(f"Test Accuracy: {dataset} - ResNet-18 Models")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f"Test Accuracy: {dataset} - ResNet-50 Models")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def transfer_learning(dataset: str):
    results_dir = "./results/train"
    csv_files = [f for f in os.listdir(results_dir) if (f.startswith(dataset + "[scratch]-ParaLIF") or f.startswith(dataset + "-ParaLIF")) and f.endswith(".csv")]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 1]})

    for csv_file in csv_files:
        file_path = os.path.join(results_dir, csv_file)
        data = pd.read_csv(file_path)

        base_name = os.path.splitext(csv_file)[0]
        dataset_name, variant, model = base_name.split("-", 2)

        scratch = "[scratch]" in dataset_name
        label = f"{model} ({variant})" if variant != "" else model
        if scratch:
            label = f"{label} from scratch"

        grouped_data = data.groupby('epoch')['accuracy'].agg(['min', 'max'])
        if scratch:
            ax = ax2
            ax.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=label)
        else:
            ax = ax1
            ax.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=label)

    ax1.set_ylim(0.8, 1)
    ax2.set_ylim(0.25, 0.45)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.suptitle(f"Contrast Transfered Models with Scratch Models On {dataset}", fontsize=16)
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Accuracy')

    ax1.legend()
    ax2.legend()

    ax1.grid(True)
    ax2.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()

def log_format(x, pos):
    if x == 0:
        return "0"
    else:
        return f'$10^{{{int(np.log10(x))}}}$'

def plot_attack(dataset: str, attack: str):
    results_dir = "./results"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith(dataset + "-") and f.endswith(f"{attack}.csv")]

    if attack == "square@0.1":
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        for csv_file in csv_files:
            file_path = os.path.join(results_dir, csv_file)
            data = pd.read_csv(file_path)

            data["success_rate"] = (100 - data["accuracy"]) / 100

            base_name = os.path.splitext(csv_file)[0]
            _, variant, model, _ = base_name.split("-", 3)

            label = f"{model} ({variant})" if variant != "" else model
            if variant == "ParaLIF":
                grouped_data = data.groupby('max_iterations')['success_rate'].agg(['min', 'max'])
                if "resnet18" in model:
                    ax1.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=label)
                elif "resnet50" in model:
                    ax2.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=label)
            else:
                if "resnet18" in model:
                    ax1.plot(data['max_iterations'], data['success_rate'], label=label)
                elif "resnet50" in model:
                    ax2.plot(data['max_iterations'], data['success_rate'], label=label)

        ax1.set_title(f"Attack success rate: {dataset} - ResNet-18 Models")
        ax1.set_xlabel('max_iterations')
        ax1.set_ylabel('success_rate')
        ax1.legend()
        ax1.grid(True)

        ax2.set_title(f"Attack success rate: {dataset} - ResNet-50 Models")
        ax2.set_xlabel('max_iterations')
        ax2.set_ylabel('success_rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
    elif attack=="deepfool" or attack=="fgsm" or attack=="square":
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        for csv_file in csv_files:
            file_path = os.path.join(results_dir, csv_file)
            data = pd.read_csv(file_path)

            if attack == "deepfool":
                data["success_rate"] = (1 - data["accuracy"])
            if attack == "fgsm":
                data["success_rate"] = (100 - data["accuracy"]) / 100
            if attack == "square":
                data["success_rate"] = (100 - data["accuracy"]) / 100

            base_name = os.path.splitext(csv_file)[0]
            _, variant, model, _ = base_name.split("-", 3)

            label = f"{model} ({variant})" if variant != "" else model
            if variant == "ParaLIF":
                grouped_data = data.groupby('epsilon')['success_rate'].agg(['min', 'max'])
                if "resnet18" in model:
                    ax1.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=label)
                elif "resnet50" in model:
                    ax2.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=label)
            else:
                if "resnet18" in model:
                    ax1.plot(data['epsilon'], data['success_rate'], label=label)
                elif "resnet50" in model:
                    ax2.plot(data['epsilon'], data['success_rate'], label=label)

        ax1.set_title(f"Attack success rate: {dataset} - ResNet-18 Models")
        ax1.set_xlabel(r'$\epsilon$')
        ax1.set_xscale('log')
        ax1.xaxis.set_major_formatter(FuncFormatter(log_format))
        ax1.set_ylabel('Success Rate')
        ax1.legend()
        ax1.grid(True)

        ax2.set_title(f"Attack success rate: {dataset} - ResNet-50 Models")
        ax2.set_xlabel(r'$\epsilon$')
        ax2.set_xscale('log')
        ax2.xaxis.set_major_formatter(FuncFormatter(log_format))
        ax2.set_ylabel('Success Rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        raise ValueError(f"not implemented for {attack}")
