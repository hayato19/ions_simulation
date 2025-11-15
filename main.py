import numpy as np
import math
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

getcontext().prec = 50  # 計算精度（桁数）を設定

# --- params ---
dt = 1e-10
N  = 100000
alpha = 2.3e-28
eps = 1e-7

# 物理定数
hbar = Decimal("1.054e-34")
NA   = Decimal("6.02214076e23")

# --- 初期化 ---
def initialize_arrays_multi(M, N, dt, x0s, v0s=0.0):
    t = np.linspace(0.0, N*dt, N+1)
    x0s = np.asarray(x0s, dtype=float)
    v0s = np.asarray(v0s, dtype=float)

    if x0s.ndim == 0:
        x0s = np.full(M, x0s, dtype=float)
    if v0s.ndim == 0:
        v0s = np.full(M, v0s, dtype=float)

    x = np.empty((N+1, M), dtype=float)
    v = np.empty((N+1, M), dtype=float)
    x[0, :] = x0s
    v[0, :] = v0s

    return t, x, v

# --- 粒子種別パラメータ設定 ---
def set_particle_params(M, posit):
    m_arr     = np.empty(M, dtype=float)
    k_arr     = np.empty(M, dtype=float)
    kl_arr    = np.empty(M, dtype=float)
    gamma_arr = np.empty(M, dtype=float)
    S0_arr    = np.empty(M, dtype=float)
    delta_arr = np.empty(M, dtype=float)

    two_pi = 2.0 * math.pi
    base_delta = -40.0e6 * two_pi

    for i in range(M):
        if posit[i] == 1:
            M_mol = Decimal("9e-3")
            m_decimal = M_mol / NA
            m_arr[i]  = float(m_decimal)

            k_arr[i]     = (two_pi * 1e6) ** 2 * m_arr[i]
            kl_arr[i]    = 2.0 * math.pi / 313e-6
            gamma_arr[i] = 20.0e6 * two_pi
            S0_arr[i]    = 0
            delta_arr[i] = base_delta
        else:
            m_arr[i]     = 1.0
            k_arr[i]     = 1.0
            kl_arr[i]    = 0.0
            gamma_arr[i] = 0.0
            S0_arr[i]    = 0.0
            delta_arr[i] = base_delta

    return m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr

# --- オイラー法 ---
def euler_step_multi(m, k, x, v, dt, N, alpha, eps, S0, kl, gamma, delta):

    M = x.shape[1]
    hbar_f = float(hbar)
    f = np.zeros((N+1, M), dtype=float)# 時間発展の f(t,i) を保存する 2D 配列

    for n in range(N):
        a = -(k[:] / m[:]) * x[n, :].copy()

        for i in range(M):
            ai = 0.0
            xi = x[n, i]
            mi = m[i]

            for j in range(M):
                if j == i:
                    continue
                dx = x[n, j] - xi
                r3 = abs(dx ** 3)
                ai += -(alpha / m[i]) * dx / (r3 + eps**3)

            a[i] += ai
            f = cooling_step(v[n,i], S0[i], kl[i], gamma[i], delta[i], hbar_f)
            a[i] += f / m[i]

        v[n+1, :] = v[n, :] + dt * a[:]
        x[n+1, :] = x[n, :] + dt * v[n+1, :]

    return x, v

# --- 4次ルンゲクッタ法 ---
def rk4_step_multi(m, k, x, v, dt, N, S0, kl, gamma, delta):

    hbar_f = float(hbar)
    def f(xn, vn, S0, kl, gamma, delta):

        dxdt = vn
        # dx[i, j] = x_j - x_i
        dx = xn[np.newaxis, :] - xn[:, np.newaxis]   # (M, M)
        np.fill_diagonal(dx, 0.0)                    # 自己項を0に
        r3 = abs(dx ** 3)                            # 相対変位３乗
        a_int = -(alpha/m) * np.sum(dx / (r3 + eps ** 3), axis=1)
        f = cooling_step(dxdt,S0[:],kl[:],gamma[:],delta[:], hbar_f)
        dvdt = -(k/m) * xn + a_int
        return dxdt, dvdt

    for n in range(N):
        xn, vn = x[n, :], v[n, :]

        k1x, k1v = f(xn, vn, S0, kl, gamma, delta)
        k2x, k2v = f(xn + 0.5*dt*k1x, vn + 0.5*dt*k1v, S0, kl, gamma, delta)
        k3x, k3v = f(xn + 0.5*dt*k2x, vn + 0.5*dt*k2v, S0, kl, gamma, delta)
        k4x, k4v = f(xn + dt*k3x,     vn + dt*k3v, S0, kl, gamma, delta)

        x[n+1, :] = xn + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        v[n+1, :] = vn + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)

    return x, v

def cooling_step(v, S0, kl, gamma, delta, h):
    denom = (S0 + 1.0 + 4.0 / gamma ** 2 * (delta - kl * v) ** 2)
    f = h * gamma * S0 / 2.0 / denom * kl
    return f

# --- 実行 ---
M = 5
x0s = np.linspace(-4e-5, 4e-5, M)
v0s = 0.0
posit = [1, 1, 1, 1, 1]

m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr = set_particle_params(M, posit)
t, xM, vM = initialize_arrays_multi(M, N, dt, x0s, v0s)

# xM, vM, f = euler_step_multi(
#     m_arr, k_arr, xM, vM, dt, N,
#     alpha=alpha, eps=eps,
#     S0=S0_arr, kl=kl_arr, gamma=gamma_arr, delta=delta_arr
# )

xM, vM = rk4_step_multi(
    m_arr, k_arr, xM, vM, dt, N,
    S0=S0_arr, kl=kl_arr, gamma=gamma_arr, delta=delta_arr
)
print(f"t_final = {t[-1]:.3f}  |  x_last (per particle) = {xM[-1, :]}")


import os
import matplotlib.pyplot as plt

SAVE_DIR = "./figs"                 # 保存先フォルダ
SAVE_NAME = "x_vs_time.png"         # 任意のファイル名

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, SAVE_NAME)

# 描画設定
plt.rcParams['figure.figsize'] = [20, 6]
plt.rcParams['figure.dpi'] = 200

# 描画設定を調整して画質を向上
plt.rcParams['figure.figsize'] = [20, 6]   # 横長の20インチ×6インチ
plt.rcParams['figure.dpi'] = 200           # 高解像度

# --- 可視化（各粒子の x(t) を重ね描き） ---
plt.figure()  # figsizeは上で設定したので不要
for j in range(M):
    plt.plot(t, xM[:, j], label=f'particle {j}')
plt.title('Multi-particle trajectory (expanded view)')
plt.xlabel('Time t [s]')
plt.ylabel('Position x(t)')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

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