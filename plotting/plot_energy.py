import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_energy(t, e, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"energy_vs_time_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(t, e, linewidth=1.2)

    plt.xlabel("time [s]")
    plt.ylabel("total energy [J]")
    plt.title("Total Energy vs Time")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=250)
    plt.show()

    print("Saved energy figure:", save_path)
