import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def parse_results(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    results = {
        'Algo1': [],
        'Algo2': []
    }

    tau = None
    burn_in = None

    for line in lines:
        line = line.strip()

        tau_burn_match = re.match(r'Running with tau=([\d\.e\-]+) and burn_in=(\d+)', line)
        if tau_burn_match:
            tau = float(tau_burn_match.group(1))
            burn_in = int(tau_burn_match.group(2))
            continue

        algo1_match = re.match(r'Mean probability for Algo1:.*?tensor\(\[\[([0-9.eE\-]+)\]\]', line)
        if algo1_match and tau is not None and burn_in is not None:
            val = float(algo1_match.group(1))
            results['Algo1'].append((tau, burn_in, val))
            continue

        algo2_match = re.match(r'Mean probability for Algo2:.*?tensor\(\[\[([0-9.eE\-]+)\]\]', line)
        if algo2_match and tau is not None and burn_in is not None:
            val = float(algo2_match.group(1))
            results['Algo2'].append((tau, burn_in, val))

    return results

def prepare_heatmap_data(entries):
    # Extract unique tau and burn_in values
    taus = sorted(set([entry[0] for entry in entries]))
    burn_ins = sorted(set([entry[1] for entry in entries]))

    tau_idx = {tau: i for i, tau in enumerate(taus)}
    burn_idx = {b: i for i, b in enumerate(burn_ins)}

    heatmap = np.full((len(taus), len(burn_ins)), np.nan)

    for tau, burn_in, val in entries:
        i = tau_idx[tau]
        j = burn_idx[burn_in]
        heatmap[i, j] = val

    return heatmap, taus, burn_ins

def plot_comparison_heatmaps(results):
    algo1_data, xticks, yticks = prepare_heatmap_data(results['Algo1'])
    algo2_data, _, _ = prepare_heatmap_data(results['Algo2'])

    diff = algo1_data - algo2_data

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.heatmap(algo1_data, xticklabels=yticks, yticklabels=xticks, annot=True, ax=axes[0], cmap='YlGnBu')
    axes[0].set_title("Algo1 Mean Probabilities")
    axes[0].set_xlabel("Burn-in")
    axes[0].set_ylabel("Tau")

    sns.heatmap(algo2_data, xticklabels=yticks, yticklabels=xticks, annot=True, ax=axes[1], cmap='YlGnBu')
    axes[1].set_title("Algo2 Mean Probabilities")
    axes[1].set_xlabel("Burn-in")
    axes[1].set_ylabel("Tau")

    sns.heatmap(diff, xticklabels=yticks, yticklabels=xticks, annot=True, ax=axes[2], center=0, cmap='coolwarm')
    axes[2].set_title("Difference (Algo1 - Algo2)")
    axes[2].set_xlabel("Burn-in")
    axes[2].set_ylabel("Tau")

    plt.tight_layout()
    plt.savefig("heatmap_prob.png")
    plt.show()

if __name__ == '__main__':
    log_file = 'out'  # Replace with your actual file name
    results = parse_results(log_file)
    print("Parsed keys:", results.keys())
    print("Algo1 entries:", len(results['Algo1']))
    print("Algo2 entries:", len(results['Algo2']))
    plot_comparison_heatmaps(results)
