import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_full_x(t, xM, save_dir="./figs"):
    """
    全粒子の x(t) をプロット
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"x_vs_time_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(20, 6))
    M = xM.shape[1]

    for j in range(M):
        plt.plot(t, xM[:, j], label=f"particle {j}")

    plt.title("Multi-particle trajectory (x vs t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position x(t)")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    print("Saved x-t figure:", save_path)



def plot_full_f(t, f, save_dir="./figs"):
    """
    全粒子の radiation force f(t) をプロット
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"f_vs_time_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(20, 6))
    M = f.shape[1]

    for j in range(M):
        plt.plot(t, f[:, j], label=f"particle {j}")

    plt.title("Radiation force (f vs t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Force f(t)")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    print("Saved f-t figure:", save_path)
