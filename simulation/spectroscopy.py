import numpy as np
import math
from simulation.params import dt, N, w

def calculate_rho_sp(M, v):
    c = 299_792_458.0
    ramda = 313e-9
    gamma = 100e3 * 2 * math.pi   # 100 kHz (rad/s)
    s_sp = 1.0
    omega_0 = 2 * math.pi * c / ramda

    # サイドバンド候補（Hz）
    f_modes = np.array([1.0, math.sqrt(3.0)]) * 1e6
    f_max = np.max(f_modes)

    # 掃引範囲：最大側帯 + 余裕（±5 MHz 推奨）
    span_hz = 5e6
    omega_sp = np.linspace(omega_0 - 2*math.pi*span_hz, omega_0 + 2*math.pi*span_hz, 1000)

    delta_sp = omega_sp - omega_0
    k_sp = omega_sp / c

    v = np.asarray(v)
    if v.ndim != 2 or v.shape[0] != N + 1 or v.shape[1] != M:
        raise ValueError(f"v must be shape (N+1, M) = ({N+1}, {M}), got {v.shape}")

    rho_int = np.zeros((len(omega_sp), M))
    dt_rec = dt * w

    for i in range(len(omega_sp)):
        sum_rho = np.zeros(M)
        for j in range(N + 1):
            v_now = v[j, :]
            rho_sp = s_sp / 2 / (s_sp + 1 + 4 / (gamma ** 2) * (delta_sp[i] - k_sp[i] * v_now) ** 2)
            sum_rho += rho_sp * dt_rec

        rho_int[i, :] = sum_rho
        if i % 50 == 0:
            print(f"calc {i}/{len(omega_sp)}")

    return omega_sp, rho_int, omega_0, f_modes
