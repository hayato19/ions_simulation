import numpy as np
import math
import random
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import time

from gmpy2 import square

getcontext().prec = 50  # 計算精度

# --- params ---
dt = 1e-10 / 2        #0.05ns
N  = 5000
w  = 200              # 記録間隔（w ステップに 1 回記録）

alpha = 2.3e-28
eps = 1e-7

# ---- heating parameters ----
ht  = 1000      # heating 判定間隔 = dt * ht
ips = 0         # scattering param
v_th = 1e-5       # v=0の閾値

# 物理定数
hbar = Decimal("1.054e-34")
NA   = Decimal("6.02214076e23")

# --- 初期化 ---
def initialize_arrays_multi(M, N, w, dt, x0s, v0s=0.0):

    total_steps = N * w           # ← 計算する全ステップ数
    t_end = total_steps * dt      # ← 記録された最終時刻

    t = np.linspace(0.0, t_end, N + 1)   # ← 記録される N+1 個の時刻

    x0s = np.asarray(x0s, dtype=float)
    v0s = np.asarray(v0s, dtype=float)

    if x0s.ndim == 0: x0s = np.full(M, x0s)
    if v0s.ndim == 0: v0s = np.full(M, v0s)

    x = np.empty((N+1, M))
    v = np.empty((N+1, M))
    f = np.empty((N+1, M))

    x[0, :] = x0s
    v[0, :] = v0s
    f[0, :] = 0.0

    return t, x, v, f

# --- 粒子種別パラメータ設定 ---
def set_particle_params(M, posit):
    m_arr     = np.empty(M)
    k_arr     = np.empty(M)
    kl_arr    = np.empty(M)
    gamma_arr = np.empty(M)
    S0_arr    = np.empty(M)
    delta_arr = np.empty(M)

    two_pi = 2.0 * math.pi

    for i in range(M):
        if posit[i] == 1:
            M_mol = Decimal("9e-3")
            m_arr[i] = float(M_mol / NA)

            k_arr[i]     = (two_pi * 1e6)**2 * m_arr[i]
            kl_arr[i]    = two_pi / 313e-9
            gamma_arr[i] = 20.0e6 * two_pi
            S0_arr[i]    = 10
            delta_arr[i] = -40.0e6 * two_pi

        else:
            m_arr[i]     = 1.0
            k_arr[i]     = 1.0
            kl_arr[i]    = 0.0
            gamma_arr[i] = 0.0
            S0_arr[i]    = 0.0
            delta_arr[i] = -40.0e6 * two_pi

    return m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr

def calculate_rho(v, S0, kl, gamma, delta):
    rho = S0 / 2 / (S0 + 1.0 + 4.0 / (gamma ** 2) * (delta - kl * v) ** 2)
    return rho

# --- 冷却力 ---
def cooling_step(v, S0, kl, gamma, delta, h):
    rho = calculate_rho(v, S0, kl, gamma, delta)
    return h * gamma * rho * kl

def heating_step(v, S0, kl, gamma, delta, h, m, ips, ht, dt):
    rho = calculate_rho(v, S0, kl, gamma, delta)
    E = (h * kl) ** 2 / (2 * m) * gamma * rho * (1 + ips)
    o = np.sqrt(2 * E * ht * dt / m)
    u = math.cos(random.uniform(0, 2 * math.pi))
    return o * u


# --- オイラー法（最適化バージョン） ---
def euler_step_multi(m, k, x, v, f, dt, N, w, alpha, eps,
                     S0, kl, gamma, delta, ips, ht, v_th):

    M = x.shape[1]
    hbar_f = float(hbar)

    x_now = x[0, :].copy()
    v_now = v[0, :].copy()
    f_now = np.zeros(M, dtype=float)

    record_index = 1

    # -------------------------------------
    # ★ heating のログを保存するリスト
    # (step, particle_index, dv_value)
    # -------------------------------------
    heating_log = []

    for step in range(1, N*w + 1):

        # ① 調和力
        a = -(k / m) * x_now

        # ② 粒子間相互作用
        dx = x_now[np.newaxis, :] - x_now[:, np.newaxis]
        np.fill_diagonal(dx, 0.0)
        r3 = np.abs(dx)**3 + eps**3
        a_int = -(alpha / m) * np.sum(dx / r3, axis=1)
        a += a_int

        # ③ 冷却力
        for i in range(M):
            f_now[i] = cooling_step(v_now[i], S0[i], kl[i], gamma[i], delta[i], hbar_f)

        a += f_now / m

        # ④ Euler 更新
        v_now = v_now + dt * a
        x_now = x_now + dt * v_now

        # ⑤ Heating：一定ステップごと
        if step % ht == 0:
            for i in range(M):
                if abs(v_now[i]) < v_th:

                    # heating 値を計算
                    dv = heating_step(
                        v_now[i], S0[i], kl[i], gamma[i], delta[i],
                        hbar_f, m[i], ips, ht, dt
                    )

                    # apply heating
                    v_now[i] += dv

                    # ★ログに追加 (step, particle, dv)
                    heating_log.append((step, i, float(dv)))

        # ⑥ 記録
        if step % w == 0:
            idx = record_index
            x[idx, :] = x_now
            v[idx, :] = v_now
            f[idx, :] = f_now
            record_index += 1

    # ★heating_log を返す
    return x, v, f, heating_log
# --- 4次ルンゲクッタ法 ---
def rk4_step_multi(m, k, x, v, f, dt, N, w, alpha, eps, S0, kl, gamma, delta):

    M = x.shape[1]
    hbar_f = float(hbar)

    # 現在値（1×M）
    x_now = x[0, :].copy()
    v_now = v[0, :].copy()
    f_now = np.zeros(M, dtype=float)

    # 記録インデックス
    record_index = 1

    # ---- 内部で使う加速度計算関数 ----
    def calc_accel(xn, vn):
        # ① 調和力
        a = -(k / m) * xn

        # ② 粒子間相互作用（ベクトル化）
        dx = xn[np.newaxis, :] - xn[:, np.newaxis]
        np.fill_diagonal(dx, 0.0)
        r3 = np.abs(dx)**3 + eps**3
        a_int = -(alpha / m) * np.sum(dx / r3, axis=1)
        a += a_int

        # ③ 冷却力
        f_tmp = np.zeros(M)
        for i in range(M):
            f_tmp[i] = cooling_step(vn[i], S0[i], kl[i], gamma[i], delta[i], hbar_f)

        a += f_tmp / m
        return a, f_tmp

    # ---- 時間発展 ----
    total_steps = N * w
    for step in range(1, total_steps + 1):

        # ----- k1 -----
        a1, f1 = calc_accel(x_now, v_now)
        k1x = v_now
        k1v = a1

        # ----- k2 -----
        x2 = x_now + 0.5 * dt * k1x
        v2 = v_now + 0.5 * dt * k1v
        a2, _ = calc_accel(x2, v2)
        k2x = v2
        k2v = a2

        # ----- k3 -----
        x3 = x_now + 0.5 * dt * k2x
        v3 = v_now + 0.5 * dt * k2v
        a3, _ = calc_accel(x3, v3)
        k3x = v3
        k3v = a3

        # ----- k4 -----
        x4 = x_now + dt * k3x
        v4 = v_now + dt * k3v
        a4, f4 = calc_accel(x4, v4)
        k4x = v4
        k4v = a4

        # ---- RK4 更新 ----
        x_now = x_now + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        v_now = v_now + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

        # 冷却力は k4 の f を代表値とする
        f_now = f4.copy()

        # ---- 記録処理 ----
        if step % w == 0:
            idx = record_index
            x[idx, :] = x_now
            v[idx, :] = v_now
            f[idx, :] = f_now
            record_index += 1

    return x, v, f

def plot_x_range(t, xM, t_start, t_end, particle_index,
                 save_dir="./figs"):
    """
    ★ 指定した粒子 1 個だけをプロットする関数 ★

    Parameters:
        t  : 時間配列 (N+1)
        xM : 位置配列 (N+1, M)
        t_start, t_end: 描画したい時間範囲 [s]
        particle_index: プロット対象の粒子番号 (0 ～ M-1)
        save_dir: 保存先ディレクトリ
    """

    # --- 粒子番号チェック ---
    M = xM.shape[1]
    if not (0 <= particle_index < M):
        print(f"particle_index={particle_index} が範囲外 (0〜{M-1})")
        return

    # --- 時間インデックス抽出 ---
    idx = np.where((t >= t_start) & (t <= t_end))[0]
    if len(idx) == 0:
        print("指定した時間範囲にデータがありません")
        return

    # --- 保存名 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = (
        f"x_particle{particle_index}"
        f"_{t_start:.2e}_{t_end:.2e}_{timestamp}.png"
    )
    save_path = os.path.join(save_dir, save_name)

    # --- プロット ---
    plt.figure(figsize=(16,5))

    plt.plot(
        t[idx],
        xM[idx, particle_index],
        label=f"particle {particle_index}",
        lw=1.5
    )

    plt.title(
        f"x(t) for particle {particle_index} "
        f"(range {t_start:.2e}–{t_end:.2e} s)"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    print("Saved range figure:", save_path)
# --- 実行 ---
M = 5
x0s = np.linspace(-12e-6, 12e-6, M)
v0s = 0.0
posit = [1, 1, 1, 1, 1]

start_time = time.time()

m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr = set_particle_params(M, posit)
t, xM, vM, f = initialize_arrays_multi(M, N, w, dt, x0s, v0s)

xM, vM, f, heating_log = euler_step_multi(
    m_arr, k_arr, xM, vM, f, dt, N, w,
    alpha, eps, S0_arr, kl_arr, gamma_arr, delta_arr,
    ips, ht, v_th
)

print("Heating executed:", len(heating_log), "times")
# xM, vM, f = rk4_step_multi(
#     m_arr, k_arr, xM, vM, f, dt, N, w, alpha, eps, S0_arr, kl_arr, gamma_arr, delta_arr
# )
print(f"t_final = {t[-1]:.3f}  |  x_last (per particle) = {xM[-1, :]}")

print("Heating log (first 10 entries):")
for item in heating_log[:10]:
    step, i, dv = item
    print(f" step={step}, particle={i}, dv={dv:.3e}")


import os
from datetime import datetime
import matplotlib.pyplot as plt

# --- 時間依存のファイル名を生成 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVE_DIR = "./figs"
SAVE_NAME_X = f"x_vs_time_{timestamp}.png"   # x(t)
SAVE_NAME_F = f"f_vs_time_{timestamp}.png"   # f(t)

os.makedirs(SAVE_DIR, exist_ok=True)
save_path_x = os.path.join(SAVE_DIR, SAVE_NAME_X)
save_path_f = os.path.join(SAVE_DIR, SAVE_NAME_F)

plt.rcParams['figure.figsize'] = [20, 6]
plt.rcParams['figure.dpi'] = 200

# --- x(t) プロット ---
fig_x = plt.figure()
for j in range(M):
    plt.plot(t, xM[:, j], label=f'particle {j}')
plt.title('Multi-particle trajectory (x vs t)')
plt.xlabel('Time t [s]')
plt.ylabel('Position x(t)')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
fig_x.savefig(save_path_x, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
plt.show()

print(f"Saved x-t figure to: {save_path_x}")

# # --- f(t) プロット ---
# fig_f = plt.figure()
# for j in range(M):
#     plt.plot(t, f[:, j], label=f'particle {j}')
# plt.title('Radiation force (f vs t)')
# plt.xlabel('Time t [s]')
# plt.ylabel('f(t)')
# plt.grid(True)
# plt.legend(ncol=2)
# plt.tight_layout()
# fig_f.savefig(save_path_f, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
# plt.show()
#
# print(f"Saved f-t figure to: {save_path_f}")

total_steps = N * w
T_total = total_steps * dt
print(f"Total simulated time = {T_total:.6e} s")

end_time = time.time()
elapsed = end_time - start_time
print(f"Execution time = {elapsed:.3f} s")

plot_x_range(t, xM, t_start=3e-5, t_end=t[-1], particle_index=2)

# from ipywidgets import interact

# view_max = 50
# view_step = 10
# @interact(start=(0, len(t)-view_max, view_step)) ##最小値、最大値、ステップサイズ
# def plot_window(start=0):
#     end = start + view_max ##描画範囲ステップ
#     plt.figure(figsize=(14,4))
#     plt.xlabel("time [s]")
#     plt.ylabel("x")
#     plt.title("x vs time ")
#     for j in range(M):
#         plt.plot(t[start:end], xM[start:end, j], lw=1, label=f"particle {j}")
#     plt.grid(True)
#     plt.legend()
#     plt.show()