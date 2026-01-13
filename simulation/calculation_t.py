import numpy as np

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

    T = np.zeros_like(v)
    T[:][:] = m / kB * v[:][:] ** 2

    T_min = (hbar * Gamma * np.sqrt(1.0 + s0) / (4.0 * kB)) * (1.0 + j)

    return T, T_min