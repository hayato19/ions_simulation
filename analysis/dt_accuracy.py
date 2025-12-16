import numpy as np
import matplotlib.pyplot as plt
import os

from simulation.params import alpha, eps, ips, ht
from simulation.initialize import initialize_arrays_multi
from simulation.particle_params import set_particle_params
from simulation.solver_euler import euler_step_multi
from simulation.solver_rk4 import rk4_step_multi


# -------------------------------------------------------
# dt 候補を広いオーダーでスキャンする
# -------------------------------------------------------

dt_list = np.array([
    1e-10,   # 非常に小さい dt（限界の下限を確認）
    3e-10,
    1e-9,
    3e-9,
    1e-8,    # 収束領域の中心（最適候補）
    3e-8,
    1e-7     # 不安定領域の入口（限界確認）
])


# ===========================================
# 初期エネルギー
# ===========================================
def initial_energy(k, x0):
    return 0.5 * k * x0**2


# ===========================================
# エネルギー誤差
# ===========================================
def energy_error(E, E0):
    delta = np.max(np.abs(E - E0)) / abs(E0)
    dE = delta * abs(E0)
    return delta, dE


# ===========================================
# initialize の安全処理
# ===========================================
def safe_initialize(N_eff, w_eff, dt, x0, v0):

    M = 1
    x0s = np.array([x0])
    v0s = np.array([v0])

    out = initialize_arrays_multi(M, N_eff, w_eff, dt, x0s, v0s)

    x = v = f = t = None

    for arr in out:
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.size == N_eff+1:
            t = arr
        elif arr.ndim == 2 and arr.shape[0] == N_eff+1:
            if x is None:
                x = arr
            elif v is None:
                v = arr
            else:
                f = arr

    return x, v, f, t


# ===========================================
# dt で 1 回計算
# ===========================================
def run_once(dt, method):

    M = 1
    posit = np.array([1])

    m_arr, k_arr, S0, kl, gamma, delta = set_particle_params(M, posit)
    m = m_arr[0]
    k = k_arr[0]

    # 初期条件（保存系）
    x0 = 10e-6
    v0 = 0.0

    E0 = initial_energy(k, x0)

    # 保存系誤差比較用（長め）
    N_eff = 50000
    w_eff = 10

    x, v, f, t = safe_initialize(N_eff, w_eff, dt, x0, v0)

    # 冷却・加熱はOFF（既に0のため何も変更しない）

    if method == "euler":
        xM, vM, fM, heating_log, rM, E = euler_step_multi(
            m_arr, k_arr, x, v, f, dt, N_eff, w_eff,
            alpha, eps, S0, kl, gamma, delta, ips, ht, 0
        )
    else:
        xM, vM, fM, heating_log, rM, E = rk4_step_multi(
            m_arr, k_arr, x, v, f, dt, N_eff, w_eff,
            alpha, eps, S0, kl, gamma, delta, ips, ht, 0
        )

    return E, E0


# ===========================================
# dt スキャン
# ===========================================
def scan_dt(method):

    dt_vals = []
    dE_vals = []
    delta_vals = []

    print(f"\n=== {method} dt scan ===")

    for dt in dt_list:
        print(f"dt={dt} ...")

        E, E0 = run_once(dt, method)
        delta, dE = energy_error(E, E0)

        print(f"  δE = {delta}")
        print(f"  dE = {dE}")

        dt_vals.append(dt)
        delta_vals.append(delta)
        dE_vals.append(dE)

    return np.array(dt_vals), np.array(delta_vals), np.array(dE_vals)


# ===========================================
# log–log プロット
# ===========================================
def plot_results(dt_euler, delta_euler, dt_rk4, delta_rk4):

    os.makedirs("./figs", exist_ok=True)
    save_path = "./figs/dt_accuracy_preserved_large_dt.png"

    plt.figure(figsize=(8, 6))

    plt.loglog(dt_euler, delta_euler, "o-", label="Euler")
    plt.loglog(dt_rk4, delta_rk4, "s-", label="RK4")

    plt.xlabel("time step dt")
    plt.ylabel("relative energy error δE")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print("Saved:", save_path)

# ===========================================
# Main
# ===========================================
if __name__ == "__main__":

    dt_euler, delta_euler, dE_euler = scan_dt("euler")
    dt_rk4, delta_rk4, dE_rk4 = scan_dt("rk4")

    plot_results(dt_euler, delta_euler, dt_rk4, delta_rk4)
