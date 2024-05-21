import os
import csv
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

    ax1.set_title(f"Training Data for {dataset} - ResNet18 Models")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f"Training Data for {dataset} - ResNet50 Models")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_attack(attack: str):
    result_files = [f for f in os.listdir('./results') if f.endswith(f"-{attack}.csv")]
    if not result_files:
        print(f"no result files found for attack: {attack}")
        return

    data_dict = {}
    for file in result_files:
        spec = file.replace('.csv', '').split('-')

        # NOTE: we will deduce these given the results filename...
        dataset, model, variant, attack = "", "", "", ""
        if len(spec) == 3:
            dataset, model, attack = spec
        if len(spec) == 4:
            dataset, variant, model, attack = spec

        label = f"{model} ({variant})" if variant != "" else model

        with open(os.path.join('./results', file), mode='r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            data = [(float(row[0]), float(row[1])) for row in reader]

        if dataset not in data_dict:
            data_dict[dataset] = {}
        data_dict[dataset][label] = data

    for dataset, results in data_dict.items():
        plt.figure()
        for label, data in results.items():
            epsilons, accuracies = zip(*data)
            plt.plot(epsilons, accuracies, label=label)
        plt.xlabel('Epsilon') # TODO: hardcoded for FGSM
        plt.ylabel('Accuracy') # TODO: hardcoded for FGSM
        plt.title(f'Attack: {attack} on Dataset: {dataset}')
        plt.legend()
        plt.grid(True)
        plt.show()
