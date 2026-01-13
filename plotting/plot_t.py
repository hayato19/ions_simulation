import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_t(t, T, T_min, M, mag, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"energy_vs_time_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)

    n = len(T) // mag
    T = T[:n]
    t = t[:n]

    plt.figure(figsize=(10, 6))
    for j in range(M):
        plt.plot(t, T[:, j], label=f"temperature {j}")

    plt.axhline(T_min, color="black", linestyle="-", linewidth=1, label=r"$\T_min$")

    plt.xlabel("time [s]")
    plt.ylabel("temperature [K]")
    plt.title("Temperature vs Time")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=250)
    plt.show()

    print("Saved energy figure:", save_path)
