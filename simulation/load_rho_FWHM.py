import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime

from simulation.params import ht


def calculate_fwhm(x, y):
    """
    単峰スペクトル y(x) の FWHM を計算する。
    x: detuning 軸 [rad/s]
    y: スペクトル強度
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # NaNを除外
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("Not enough valid data points to calculate FWHM")

    peak_index = np.argmax(y)
    y_max = y[peak_index]
    half_max = y_max / 2.0

    # 左側
    left_indices = np.where(y[:peak_index] < half_max)[0]
    if len(left_indices) == 0:
        raise ValueError("Left half-maximum point not found")

    i1 = left_indices[-1]
    i2 = i1 + 1

    x_left = x[i1] + (half_max - y[i1]) * (x[i2] - x[i1]) / (y[i2] - y[i1])

    # 右側
    right_indices = np.where(y[peak_index:] < half_max)[0]
    if len(right_indices) == 0:
        raise ValueError("Right half-maximum point not found")

    i2 = peak_index + right_indices[0]
    i1 = i2 - 1

    x_right = x[i1] + (half_max - y[i1]) * (x[i2] - x[i1]) / (y[i2] - y[i1])

    fwhm = x_right - x_left

    return fwhm, x_left, x_right, half_max


def load_rho_int_and_plot(
        rho_int_path,
        detuning_num,
        mode="sum",
        save_dir="./figs"
):
    rho_int = np.load(rho_int_path)

    if rho_int.shape[0] != detuning_num:
        raise ValueError(
            f"detuning_num mismatch: "
            f"given={detuning_num}, file={rho_int.shape[0]}"
        )

    _, M = rho_int.shape

    scale = 0.5e7 * 2 * math.pi / 5
    delta = np.linspace(scale * 0.5, scale * 1.5, detuning_num)

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"spectroscopy_from_rho_int_{mode}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    plt.figure(figsize=(8, 5))

    # ==============================
    # プロット用スペクトルを作成
    # ==============================
    if mode == "sum":
        rho_plot = np.nansum(rho_int, axis=1)
        label = f"Signal strength(ht = {ht})"

    elif mode == "mean":
        rho_plot = np.nanmean(rho_int, axis=1)
        label = "mean over particles"

    elif mode == "each":
        # FWHM計算には全粒子和を使用
        rho_plot = np.nansum(rho_int, axis=1)
        label = "sum over particles"

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

    if mode in ["sum", "mean"]:
        plt.plot(delta, rho_plot, "-", label=label)

    # ==============================
    # FWHM計算
    # ==============================
    fwhm, x_left, x_right, half_max = calculate_fwhm(delta, rho_plot)

    plt.axhline(half_max, linestyle="--", linewidth=0.8, label="half maximum")
    plt.axvline(x_left, linestyle="--", linewidth=0.8)
    plt.axvline(x_right, linestyle="--", linewidth=0.8)

    plt.xlabel(r"angular frequency $\omega$ [rad/s]")
    plt.ylabel(r"integrated $\rho_{\mathrm{ee}}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Loaded:", rho_int_path)
    print("rho_int shape:", rho_int.shape)
    print("FWHM [rad/s]:", fwhm)
    print("FWHM / 2π [Hz]:", fwhm / (2 * math.pi))
    print("left half max [rad/s]:", x_left)
    print("right half max [rad/s]:", x_right)
    print("Saved:", save_path)

    return rho_int, fwhm, save_path


if __name__ == "__main__":
    rho_int_path = "../data/rho_int_20260616_111102.npy"

    detuning_num = 300

    load_rho_int_and_plot(
        rho_int_path=rho_int_path,
        detuning_num=detuning_num,
        mode="sum"
    )