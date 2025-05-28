import pandas as pd
import matplotlib.pyplot as plt
import os

def summarize_batch(csv_files):
    metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    summary = {m: [] for m in metrics}

    for file in csv_files:
        df = pd.read_csv(file)
        for m in metrics:
            summary[m].append(df[m].values)

    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    for i, m in enumerate(metrics):
        data = summary[m]
        stacked = pd.DataFrame(data).T
        axs[i//2][i%2].plot(stacked.mean(axis=1), label="Mean")
        axs[i//2][i%2].fill_between(
            stacked.index,
            stacked.mean(axis=1) - stacked.std(axis=1),
            stacked.mean(axis=1) + stacked.std(axis=1),
            alpha=0.3
        )
        axs[i//2][i%2].set_title(m.replace("_", " ").title())
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csvs = [f for f in os.listdir() if f.startswith("segregation_metrics") and f.endswith(".csv")]
    summarize_batch(csvs)
