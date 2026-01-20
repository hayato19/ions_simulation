import numpy as np
from simulation.params import dt, w, N

# ---- physical constants ----
kB   = 1.380649e-23        # J/K
hbar = 1.054571817e-34     # J·s

def T_ratio_with_and_without_COM(
    v: np.ndarray,
    m: float,
    Gamma,
    s0,
    j=2/5,
):
    Gamma = float(Gamma)
    s0 = float(s0)
    j = float(j)
    m = float(m)

    M = 5

    trap_f = 1e6
    w = 50 // 3
    n_cycle = 1000 // w  #約1サイクル(互換性なし)
    n_sum = N // n_cycle  #総ステップ≒n_cycle * n_sum

    v2 = np.zeros_like(v)
    v2 = v2[:n_sum]
    T = np.zeros_like(v)
    T = T[:n_sum]


    for i in range(n_sum-1):
        for j in range(n_cycle-1):
            step = n_cycle * i + j
            v2[i][:] += v[step][:] ** 2
        T[:][:] = m / kB * v2[:][:]

    # T = np.zeros_like(v)
    # T[:][:] = m / kB * v[:][:] ** 2

    T_min = (hbar * Gamma * np.sqrt(1.0 + s0) / (4.0 * kB)) * (1.0 + j)

    return T, T_min, n_sum