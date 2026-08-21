import numpy as np
import os
from datetime import datetime
from simulation.params import dt, N, w, alpha, eps, ips, ht, hbar, set_particle_params
from simulation.initialize import initialize_arrays_multi
from plotting.plot_x_range import plot_x_range
from plotting.plot_full import plot_full_x, plot_full_f, plot_full_rho
from plotting.plot_fft import plot_fft_all_particles
from plotting.plot_energy import plot_energy
from simulation.spectroscopy import calculate_rho_sp
from plotting.plot_spectroscopy import plot_spectroscopy
from simulation.calculation_t import T_ratio_with_and_without_COM, kB
from plotting.plot_t import plot_t
from simulation.solver_rk4 import rk4_step_multi
from simulation.bloch_spec import calculate_spec_bloch

#
N_WORKERS = None


def main():
    # ======================================
    # 初期条件設定
    # ======================================

    M = 2
    x0s = np.linspace(-12e-6, 12e-6, M)
    v0s = 0.0

    m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr = set_particle_params(M)
    t, xM, vM, f = initialize_arrays_multi(M, N, w, dt, x0s, v0s)

    xM, vM, f, heating_log, r, e = rk4_step_multi(
        m_arr, k_arr, xM, vM, f, dt, N, w,
        alpha, eps, S0_arr, kl_arr, gamma_arr, delta_arr,
        ips, ht
    )

    print(f"t_final = {t[-1]:.3e} s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"x_data_{timestamp}"), xM)
    np.save(os.path.join(save_dir, f"v_data_{timestamp}"), vM)

    # 可視化例（粒子2、時間範囲3e-5〜t_end）
    # plot_x_range(t, xM, t_start=3e-5, t_end=t[-1], particle_index=2)

    # 可視化(全粒子位置、全時間範囲)
    # plot_full_x(t, xM, save_dir="./figs")

    # 可視化(全粒子の受ける力、全時間範囲)
    # plot_full_f(t, xM, save_dir="./figs")

    # 可視化(全粒子ρ、全時間範囲)
    # plot_full_rho(t, r, save_dir="./figs")

    # 可視化(全粒子のFFT、指定周波数範囲)
    # f_lines = plot_fft_all_particles(t, xM, dt, save_dir="./figs")

    # 可視化(総エネルギー、全範囲)
    # plot_energy(t, e, save_dir="./figs")

    # 温度による冷却の評価
    T, T_min, n_sum = T_ratio_with_and_without_COM(
        v=vM,
        m=m_arr[0],
        Gamma=gamma_arr[0],  # [rad/s]
        s0=S0_arr[0],  # dimensionless
    )
    plot_t(t, T, T_min, M, n_sum, 2)

    # 分光信号のシミュレーション
    calculate_spec_bloch(M, xM, vM)



    with open("tex/sections/params.tex","w",encoding="utf-8") as f:
        f.write(rf"\newcommand{{\Mval}}{{{M}}}" + "\n")
        f.write(rf"\newcommand{{\dtval}}{{{dt}}}" + "\n")
        f.write(rf"\newcommand{{\hbarval}}{{{hbar}}}" + "\n")
        f.write(rf"\newcommand{{\kval}}{{{k_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\kBval}}{{{kB}}}" + "\n")
        f.write(rf"\newcommand{{\klval}}{{{kl_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\mval}}{{{m_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\Sval}}{{{S0_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\alphaval}}{{{alpha}}}" + "\n")
        f.write(rf"\newcommand{{\gammaval}}{{{gamma_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\htval}}{{{ht}}}" + "\n")
        f.write(rf"\newcommand{{\deltaval}}{{{delta_arr[0]}}}" + "\n")
        # f.write(rf"\newcommand{{\gammadval}}{{{gammad}}}" + "\n")
        # f.write(rf"\newcommand{{\Sspval}}{{{s_sp}}}" + "\n")
        # f.write(rf"\newcommand{{\fvala}}{{{f_lines[0]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvalb}}{{{f_lines[1]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvalc}}{{{f_lines[2]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvald}}{{{f_lines[3]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvale}}{{{f_lines[4]}}}" + "\n")

if __name__ == "__main__":
    main()