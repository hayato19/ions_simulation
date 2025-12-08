import time
import numpy as np

from simulation.params import dt, N, w, alpha, eps, ips, ht
from simulation.initialize import initialize_arrays_multi
from simulation.particle_params import set_particle_params
from plotting.plot_x_range import plot_x_range
from plotting.plot_full import plot_full_x, plot_full_f

# ======================================
#  シミュレーション方式の選択
# ======================================
# "euler" または "rk4"
# USE_SOLVER = "euler"
USE_SOLVER = "rk4"

if USE_SOLVER == "euler":
    from simulation.solver_euler import euler_step_multi
elif USE_SOLVER == "rk4":
    from simulation.solver_rk4 import rk4_step_multi
else:
    raise ValueError("USE_SOLVER must be 'euler' or 'rk4'")


# ======================================
# 初期条件設定
# ======================================

M = 5
x0s = np.linspace(-12e-6, 12e-6, M)
v0s = 0.0
posit = [1, 1, 1, 1, 1]#1 : Be+

start_time = time.time()

m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr = set_particle_params(M, posit)
t, xM, vM, f = initialize_arrays_multi(M, N, w, dt, x0s, v0s)


# ======================================
# シミュレーション実行（方式切替）
# ======================================

if USE_SOLVER == "euler":
    print("=== Solver: Euler法 ===")
    xM, vM, f, heating_log = euler_step_multi(
        m_arr, k_arr, xM, vM, f, dt, N, w,
        alpha, eps, S0_arr, kl_arr, gamma_arr, delta_arr,
        ips, ht
    )
    print("Heating executed:", len(heating_log))

elif USE_SOLVER == "rk4":
    print("=== Solver: RK4法 ===")
    xM, vM, f, heating_log = rk4_step_multi(
        m_arr, k_arr, xM, vM, f, dt, N, w,
        alpha, eps, S0_arr, kl_arr, gamma_arr, delta_arr,
        ips, ht
    )


end_time = time.time()
print(f"Execution time = {end_time - start_time:.3f} s")
print(f"t_final = {t[-1]:.3e} s")


# 可視化例（粒子2、時間範囲3e-5〜t_end）
plot_x_range(t, xM, t_start=3e-5, t_end=t[-1], particle_index=2)

#可視化(全粒子位置、全時間範囲)
plot_full_x(t, xM, save_dir="./figs")

#可視化(全粒子の受ける力、全時間範囲)
# plot_full_f(t, xM, save_dir="./figs")
