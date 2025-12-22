import numpy as np
import math
from simulation.params import dt, N, w

def calculate_rho_sp(M, v):
    c = 299_792_458.0          # 光速 [m/s]
    ramda = 313e-9             # Be+ 波長 [m]
    gamma = 100e3 * 2 * math.pi
    s_sp = 1.0
    omega_0 = 2 * math.pi * c / ramda
    print(f"ω0 = {omega_0}")
    MHz = 1e6 * 2 * math.pi

    omega_sp = np.linspace(60180560e8, 60180562.5e8, 300)

    delta_sp = omega_sp - omega_0
    k_sp = omega_sp / c

    v = np.asarray(v)
    if v.ndim != 2 or v.shape[0] != N + 1 or v.shape[1] != M:
        raise ValueError(f"v must be shape (N+1, M) = ({N+1}, {M}), got {v.shape}")

    rho_int = np.zeros((len(omega_sp), M))

    dt_rec = dt * w  # 記録間隔

    for i in range(len(omega_sp)):
        sum_rho = np.zeros(M)

        for j in range(N + 1):
            v_now = v[j, :]  # (M,)
            rho_sp = s_sp / 2 / (s_sp + 1 + 4 / (gamma ** 2) * (delta_sp[i] - k_sp[i] * v_now) ** 2)
            sum_rho += rho_sp * dt_rec

        rho_int[i, :] = sum_rho

        if i % 50 == 0:
            print(f"calc {i}")

    return omega_sp, rho_int,omega_0
