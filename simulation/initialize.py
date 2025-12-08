import numpy as np

def initialize_arrays_multi(M, N, w, dt, x0s, v0s=0.0):##粒子の位置、速度の初期化

    total_steps = N * w
    t_end = total_steps * dt

    t = np.linspace(0.0, t_end, N + 1)

    x0s = np.asarray(x0s, dtype=float)
    v0s = np.asarray(v0s, dtype=float)

    if x0s.ndim == 0:
        x0s = np.full(M, x0s)
    if v0s.ndim == 0:
        v0s = np.full(M, v0s)

    x = np.empty((N+1, M))
    v = np.empty((N+1, M))
    f = np.empty((N+1, M))

    x[0, :] = x0s
    v[0, :] = v0s
    f[0, :] = 0.0

    return t, x, v, f
