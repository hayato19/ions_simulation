import numpy as np
from simulation.solver_euler import euler_step_multi
from simulation.solver_rk4 import rk4_step_multi

def run_test(dt, method="euler", N=2000, w=1):
    """
    dt を指定して１回のシミュレーションを実行し、
    総エネルギー誤差 δE を返す。
    保存系テストのため、冷却・加熱は OFF にする。
    """

    M = 1  # 単一粒子で十分（解析が簡単）

    # パラメータ（必要最低限）
    m = np.array([1.0])
    k = np.array([1.0])
    alpha = 0.0
    eps = 1e-6

    # 初期条件（適当に振動させる）
    x0 = np.array([1.0])
    v0 = np.array([0.0])

    # 記録配列
    x = np.zeros((N+1, M))
    v = np.zeros((N+1, M))
    f = np.zeros((N+1, M))
    x[0], v[0] = x0, v0

    # 冷却・加熱は OFF する
    S0 = np.array([0.0])
    kl = np.array([0.0])
    gamma = np.array([0.0])
    delta = np.array([0.0])
    ips = 0
    ht = 1

    if method == "euler":
        _, _, _, _, _, e = euler_step_multi(
            m, k, x, v, f, dt, N, w, alpha, eps,
            S0, kl, gamma, delta, ips, ht
        )
    else:
        _, _, _, _, _, e = rk4_step_multi(
            m, k, x, v, f, dt, N, w, alpha, eps,
            S0, kl, gamma, delta, ips, ht
        )

    E0 = e[0]
    dE = np.max(np.abs(e - E0))
    return dE
