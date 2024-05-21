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
        _, variant, name = base_name.split("-", 2)

        label = f"{name} ({variant})" if variant != "" else name
        if variant == "ParaLIF":
            grouped_data = data.groupby('epoch')['accuracy'].agg(['min', 'max'])
            if "resnet18" in name:
                ax1.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=f'Accuracy - {label}')
            elif "resnet50" in name:
                ax2.fill_between(grouped_data.index, grouped_data['min'], grouped_data['max'], alpha=0.3, label=f'Accuracy - {label}')
        else:
            if "resnet18" in name:
                ax1.plot(data['epoch'], data['accuracy'], label=f'Accuracy - {label}')
            elif "resnet50" in name:
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
    result_files = [f for f in os.listdir('./results') if f.endswith(f"-{attack}.csv")]
    if not result_files:
        print(f"No result files found for attack: {attack}")
        return

    _, axs = plt.subplots(2, 2, figsize=(14, 14))

    for file in result_files:
        base_name = os.path.splitext(file)[0]
        _, variant, name = base_name.split("-", 2)

        label = f"{name} ({variant})" if variant != "" else name
        data = pd.read_csv(os.path.join('./results', file))

        if attack == "fgsm" or attack == "deepfool" or attack == "fmn":
            data['success_rate'] = (100 - data['accuracy']) / 100

            if variant == "ParaLIF":
                grouped_data = data.groupby('epsilon').agg({'success_rate': ['min', 'max'], 'ssim': ['mean']})
                if "resnet18" in name:
                    axs[0, 0].fill_between(grouped_data.index, grouped_data['success_rate']['min'], grouped_data['success_rate']['max'], alpha=0.3, label=label)
                    axs[1, 0].plot(grouped_data.index, grouped_data['ssim']['mean'], label=label)
                elif "resnet50" in name:
                    axs[0, 1].fill_between(grouped_data.index, grouped_data['success_rate']['min'], grouped_data['success_rate']['max'], alpha=0.3, label=label)
                    axs[1, 1].plot(grouped_data.index, grouped_data['ssim']['mean'], label=label)
            else:
                if "resnet18" in name:
                    axs[0, 0].plot(data['epsilon'], data['success_rate'], label=label)
                    axs[1, 0].plot(data['epsilon'], data['ssim'], label=label)
                elif "resnet50" in name:
                    axs[0, 1].plot(data['epsilon'], data['success_rate'], label=label)
                    axs[1, 1].plot(data['epsilon'], data['ssim'], label=label)

            axs[0, 0].set_title(f"{attack} on {dataset} - ResNet18 Attack Success Rate")
            axs[0, 0].set_xlabel('Epsilon')
            axs[0, 0].set_ylabel('Success Rate')
            axs[0, 0].legend()
            axs[0, 0].grid(True)

            axs[0, 1].set_title(f"{attack} on {dataset} - ResNet50 Attack Success Rate")
            axs[0, 1].set_xlabel('Epsilon')
            axs[0, 1].set_ylabel('Success Rate')
            axs[0, 1].legend()
            axs[0, 1].grid(True)

            axs[1, 0].set_title(f"Training Data for {dataset} - ResNet18 Models (SSIM)")
            axs[1, 0].set_xlabel('Epsilon')
            axs[1, 0].set_ylabel('SSIM')
            axs[1, 0].legend()
            axs[1, 0].grid(True)

            axs[1, 1].set_title(f"Training Data for {dataset} - ResNet50 Models (SSIM)")
            axs[1, 1].set_xlabel('Epsilon')
            axs[1, 1].set_ylabel('SSIM')
            axs[1, 1].legend()
            axs[1, 1].grid(True)

            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Attack {attack} not yet supported for plotting")
