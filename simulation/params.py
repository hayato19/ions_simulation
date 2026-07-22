# パラメータ格納

import math
import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 50  # 計算精度

# --- params ---
dt = 25e-9        # 時間刻み（0.05ns）
N  = 600000   # 記録数, 寿命1.6us/dt * 10000
w  = 1         # 記録間隔（w ステップに 1 回記録）
                     # ステップ数　= N * w

alpha = 2.3e-28       # クーロン反発の係数
eps   = 1e-7          # 発散防止のための微小数

# ---- heating parameters ----
ht  = 1      # heating 判定間隔
ips = 0         # scattering param

# ---- operation parameters ----
cool_time = 50e-6
spec_time = 1e-6
cycle_time = cool_time + spec_time

# --- physical constants ---
hbar = Decimal("1.054e-34")     # ディラック定数
NA   = Decimal("6.02214076e23") # アボガドロ定数

def set_particle_params(M):
    """
    粒子ごとのパラメータを種類(posit)に応じて設定する
    posit[i] == 1 : Beイオン
    posit[i] != 1 : テスト粒子
    """

    m_arr     = np.empty(M)
    k_arr     = np.empty(M) # ポテンシャル拘束の係数
    kl_arr    = np.empty(M) # ρ計算時の係数k
    gamma_arr = np.empty(M)
    S0_arr    = np.empty(M)
    delta_arr = np.empty(M)

    two_pi = 2.0 * math.pi

    for i in range(M):

        M_mol = Decimal("9e-3")  # 9 mg/mol
        m_arr[i] = float(M_mol / NA)
        trap_f = 1e6
        k_arr[i]     = (two_pi * trap_f)**2 * m_arr[i]
        kl_arr[i]    = two_pi / 313e-9
        gamma_arr[i] = 80.0e6 * two_pi
        S0_arr[i]    = 1
        delta_arr[i] = -40.0e6 * two_pi

    return m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr