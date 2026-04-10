import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
plt.rcParams["font.size"] = 15

def plot_precision(dt_list, err_euler, err_rk4, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"precision_scan_{timestamp}.png")

    plt.figure(figsize=(9,6))

    plt.loglog(dt_list, err_euler, "o-", label="Euler")
    plt.loglog(dt_list, err_rk4, "o-", label="RK4")

    plt.gca().invert_xaxis()  # dt が小さいほど右 → 左に流れるように反転

    plt.xlabel("dt (log scale)")
    plt.ylabel("energy drift ΔE (log scale)")
    plt.title("Precision Comparison: Euler vs RK4")
    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)
