import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime

from simulation.params import ht


def load_rho_ee_and_plot(
        rho_ee_path,
        detuning_num,
        dt_value,
        mode="each",
        save_dir="./figs"
):
    # ==============================
    # rho_ee(t) の読み込み
    # ==============================
    rho_ee_all = np.load(rho_ee_path, mmap_mode="r")

    # shape = (detuning_num, M, time_steps)
    if rho_ee_all.shape[0] != detuning_num:
        raise ValueError(
            f"detuning_num mismatch: "
            f"given={detuning_num}, file={rho_ee_all.shape[0]}"
        )

    _, M, _ = rho_ee_all.shape

    # ==============================
    # detuning 軸を再構成
    # ==============================
    scale = 0.5e7 * 2 * math.pi / 5
    delta = np.linspace(0, scale, 300)

    # ==============================
    # rho_int を再計算
    # 元コード: sum_rho += rho_ee * dt_rec
    # ==============================
    rho_int = np.nansum(rho_ee_all, axis=2) * dt_value

    # ==============================
    # 図の保存先
    # ==============================
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"spectroscopy_from_rho_ee_{mode}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    # ==============================
    # プロット
    # ==============================
    plt.figure(figsize=(8, 5))

    if mode == "sum":
        rho_plot = np.sum(rho_int, axis=1)
        plt.plot(
            delta,
            rho_plot,
            "-",
            label=f"Signal strength(ht = {ht})"
        )

    elif mode == "mean":
        rho_plot = np.mean(rho_int, axis=1)
        plt.plot(
            delta,
            rho_plot,
            "-",
            label="mean over particles"
        )

    elif mode == "each":
        for k in range(M):
            plt.plot(
                delta,
                rho_int[:, k],
                "-",
                linewidth=0.7,
                label=f"particle {k}"
            )

    else:
        raise ValueError("mode must be 'sum', 'mean', or 'each'")

    plt.xlabel(r"angular frequency $\omega$ [rad/s]")
    plt.ylabel(r"integrated $\rho_{\mathrm{sp}}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Loaded:", rho_ee_path)
    print("rho_ee_all shape:", rho_ee_all.shape)
    print("Saved:", save_path)

    return rho_int, save_path


if __name__ == "__main__":
    rho_ee_path = "../data/rho_ee_time_20260602_164533.npy"

    detuning_num = 300
    dt_value = 50e-9  # または dt * w

    load_rho_ee_and_plot(
        rho_ee_path=rho_ee_path,
        detuning_num=detuning_num,
        dt_value=dt_value,
        mode="each"
    )