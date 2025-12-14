import matplotlib.pyplot as plt
import numpy as np

def plot_spectroscopy(omega_sp, rho_int, omega_0, mode="sum"):

    plt.figure(figsize=(8, 5))

    if mode == "sum":
        rho_plot = np.sum(rho_int, axis=1)
        plt.plot(omega_sp, rho_plot, "-o", label="sum over particles")

    elif mode == "mean":
        rho_plot = np.mean(rho_int, axis=1)
        plt.plot(omega_sp, rho_plot, "-o", label="mean over particles")

    elif mode == "each":
        M = rho_int.shape[1]
        for k in range(M):
            plt.plot(omega_sp, rho_int[:, k], "-o", label=f"particle {k}")

    else:
        raise ValueError("mode must be 'sum', 'mean', or 'each'")

    plt.axvline(
        omega_0,
        color="red",
        linestyle="-",
        linewidth=1,
        label=r"$\omega = \omega_0$"
    )

    plt.xlabel(r"angular frequency $\omega$ [rad/s]")
    plt.ylabel(r"integrated $\rho_{\mathrm{sp}}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
