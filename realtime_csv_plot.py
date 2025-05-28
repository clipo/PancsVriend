import pandas as pd
import matplotlib.pyplot as plt
import time
import os

CSV_FILE = "segregation_metrics.csv"
METRICS = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']

def monitor_csv():
    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    while True:
        if not os.path.exists(CSV_FILE):
            time.sleep(1)
            continue
        try:
            df = pd.read_csv(CSV_FILE)
            for i, m in enumerate(METRICS):
                axs[i//2][i%2].clear()
                axs[i//2][i%2].plot(df[m])
                axs[i//2][i%2].set_title(m.replace("_", " ").title())
            plt.pause(1)
        except Exception as e:
            print("Waiting for file access...")
            time.sleep(1)

if __name__ == "__main__":
    monitor_csv()
