import matplotlib.pyplot as plt
import numpy as np
import math
import os
from datetime import datetime

plt.rcParams["font.size"] = 15

def plot_spectroscopy(omega_sp, rho_int, omega_0, f_modes_hz, mode="sum", save_dir="./figs"):

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"spectroscopy_{mode}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    plt.figure(figsize=(8, 5))

    omega_sp -= omega_0

    if mode == "sum":
        rho_plot = np.sum(rho_int, axis=1)
        plt.plot(omega_sp, rho_plot, "-", label="sum over particles")
    elif mode == "mean":
        rho_plot = np.mean(rho_int, axis=1)
        plt.plot(omega_sp, rho_plot, "-", label="mean over particles")
    elif mode == "each":
        M = rho_int.shape[1]
        for k in range(M):
            plt.plot(omega_sp, rho_int[:, k], "-", label=f"particle {k}")
    else:
        raise ValueError("mode must be 'sum', 'mean', or 'each'")

    plt.xlabel(r"angular frequency $\omega$ [rad/s]")
    plt.ylabel(r"integrated $\rho_{\mathrm{sp}}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)