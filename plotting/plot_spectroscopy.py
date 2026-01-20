import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams["font.size"] = 15

def plot_spectroscopy(omega_sp, rho_int, omega_0, f_modes_hz, mode="sum"):

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

    # ω0
    # plt.axvline(0, color="black", linestyle="--", linewidth=1, label=r"$\omega_0$")

    # サイドバンド候補：ω0 ± 2π f_n
    # for f in f_modes_hz:
    #     w = omega_0 + 2*math.pi*f
    #     plt.axvline(w, color="gray", linestyle="--", linewidth=1)
    #     w = omega_0 - 2*math.pi*f
    #     plt.axvline(w, color="gray", linestyle="--", linewidth=1)

    plt.xlabel(r"angular frequency $\omega$ [rad/s]")
    plt.ylabel(r"integrated $\rho_{\mathrm{sp}}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
