import numpy as np
import math
from simulation.params import dt, N, w
from simulation.bloph import phase_shift_x, phase_shift_v, bloph_timeeq_x, bloph_timeeq_v
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.rcParams["font.size"] = 15

def calculate_spec_bloch(M, x, v):
    c = 299_792_458.0
    ramda = 313e-9
    omega_0 = 2 * math.pi * c / ramda
    t_max = N * w
    scale = 2e6 * 2 * math.pi
    delta = np.linspace(scale, -scale, 300)
    rho_int = np.zeros((len(delta), M))

    mode = "sum"
    save_dir = "./figs"

    dt_rec = dt * w

    particle = 0  # Mを上書きしないために変更

    for i in range(len(delta)):  # レーザー周波数ごと
        sum_rho = 0.0

        rho_gg = 1.0
        rho_ge = 0.0 + 0.0j

        Omega = 1

        for step in range(1, N):  # 全ステップ
            if step < N/2 :
                continue
            else:
                try:
                    # x利用==========
                    # rho_ee, rho_ge = bloph_timeeq_x(
                    #     rho_gg,
                    #     rho_ge,
                    #     x[step-1, particle],
                    #     delta[i]
                    # )
                    #    ==========

                    #v利用==========
                    rho_ee, rho_ge, Omega = bloph_timeeq_v(
                        rho_gg,
                        rho_ge,
                        v[step-1, particle],
                        delta[i],
                        Omega
                    )
                    #     ==========

                    # ---- エラーチェック ----
                    if not np.isfinite(rho_gg):
                        raise FloatingPointError("rho_gg became non-finite")

                    if not np.isfinite(rho_ge):
                        raise FloatingPointError("rho_ge became non-finite")

                    if not np.isfinite(rho_ee):
                        raise FloatingPointError("rho_ee became non-finite")

                    if abs(rho_gg) > 1e6 or abs(rho_ge) > 1e6:
                        raise FloatingPointError("rho value diverged")

                    sum_rho += rho_ee * dt_rec

                except FloatingPointError as e:
                    print("Numerical error detected")
                    print("reason =", e)
                    print("i =", i)
                    print("step =", step)
                    print("delta =", delta[i])
                    print("x =", x[step-1, particle])
                    print("rho_gg =", rho_gg)
                    print("rho_ge =", rho_ge)
                    print("rho_ee =", 1 - rho_gg)

                    sum_rho = np.nan
                    break

        rho_int[i, particle] = sum_rho

        if i % 50 == 0:
            print(f"calc {i}/{len(delta)}")

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"spectroscopy_{mode}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    plt.figure(figsize=(8, 5))

    omega_l = delta  # レーザー周波数の変化量を x 軸にする

    if mode == "sum":
        rho_plot = np.sum(rho_int, axis=1)
        plt.plot(omega_l, rho_plot, "-", label="Signal strength")
    elif mode == "mean":
        rho_plot = np.mean(rho_int, axis=1)
        plt.plot(omega_l, rho_plot, "-", label="mean over particles")
    elif mode == "each":
        M = rho_int.shape[1]
        for k in range(M):
            plt.plot(omega_l, rho_int[:, k], "-", label=f"particle {k}")
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

    return 0