import os
import csv
import matplotlib.pyplot as plt

def plot(attack):
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

    print(f"Results plotted and saved in ./results/ for attack: {attack}")
