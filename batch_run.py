import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def process_batch(run_count=10):
    all_metrics = {
        'clusters': [],
        'switch_rate': [],
        'distance': [],
        'mix_deviation': [],
        'share': [],
        'ghetto_rate': []
    }

    for i in range(run_count):
        os.system("python SchellingSim.py --headless")  # Simulated run (headless mode must be supported separately)
        df = pd.read_csv("segregation_metrics.csv")
        for key in all_metrics:
            all_metrics[key].append(df[key].values)

    summary = {}
    for key in all_metrics:
        values = np.array(all_metrics[key])
        mean_vals = np.mean(values, axis=0)
        std_vals = np.std(values, axis=0)
        summary[key] = (mean_vals, std_vals)

        plt.figure()
        plt.plot(mean_vals, label="Mean")
        plt.fill_between(np.arange(len(mean_vals)), mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)
        plt.title(f"{key.replace('_', ' ').title()} Over Time")
        plt.xlabel("Step")
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()
        plt.savefig(f"batch_{key}.pdf")
        plt.close()

    print("Batch analysis complete. PDFs saved.")

if __name__ == "__main__":
    process_batch(5)
