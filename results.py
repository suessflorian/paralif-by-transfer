import os
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_attack(dataset: str, attack: str):
    results_dir = "./results"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith(dataset + "-") and f.endswith(f"{attack}.csv")]

    if attack == "square":
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        for csv_file in csv_files:
            file_path = os.path.join(results_dir, csv_file)
            data = pd.read_csv(file_path)

            data["success_rate"] = (100 - data["accuracy"]) / 100

            base_name = os.path.splitext(csv_file)[0]
            _, variant, model, _ = base_name.split("-", 3)

            label = f"{model} ({variant})" if variant != "" else model
            print(label)
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
    elif attack=="deepfool" or attack=="fgsm":
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        for csv_file in csv_files:
            file_path = os.path.join(results_dir, csv_file)
            data = pd.read_csv(file_path)

            data["success_rate"] = (100 - data["accuracy"]) / 100

            base_name = os.path.splitext(csv_file)[0]
            _, variant, model, _ = base_name.split("-", 3)

            label = f"{model} ({variant})" if variant != "" else model
            print(label)
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
        ax1.set_xlabel('epsilon')
        ax1.set_ylabel('success_rate')
        ax1.legend()
        ax1.grid(True)

        ax2.set_title(f"Attack success rate: {dataset} - ResNet-50 Models")
        ax2.set_xlabel('epsilon')
        ax2.set_ylabel('success_rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        raise ValueError(f"not implemented for {attack}")
