# simulation/particle_params.py

import math
import numpy as np
from decimal import Decimal

# 物理定数
NA = Decimal("6.02214076e23")

def set_particle_params(M, posit):
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

        # Ion (posit = 1)
        if posit[i] == 1:
            M_mol = Decimal("9e-3")  # 9 mg/mol
            m_arr[i] = float(M_mol / NA)

            k_arr[i]     = (two_pi * 1e6)**2 * m_arr[i]
            kl_arr[i]    = two_pi / 313e-9
            gamma_arr[i] = 20.0e6 * two_pi
            S0_arr[i]    = 10
            delta_arr[i] = -40.0e6 * two_pi

        # Other particles
        else:
            m_arr[i]     = 1.0
            k_arr[i]     = 1.0
            kl_arr[i]    = 0.0
            gamma_arr[i] = 0.0
            S0_arr[i]    = 0.0
            delta_arr[i] = -40.0e6 * two_pi

    return m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr
