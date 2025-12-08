import numpy as np
from simulation.forces import cooling_step, heating_step

def euler_step_multi(m, k, x, v, f, dt, N, w, alpha, eps,
                     S0, kl, gamma, delta, ips, ht):

    M = x.shape[1]
    x_now = x[0, :].copy()
    v_now = v[0, :].copy()
    f_now = np.zeros(M)

    record_index = 1
    heating_log = []

    for step in range(1, N*w + 1):

        a = -(k/m) * x_now

        dx = x_now[np.newaxis, :] - x_now[:, np.newaxis]
        np.fill_diagonal(dx, 0.0)
        r3 = np.abs(dx)**3 + eps**3
        a_int = -(alpha/m)*np.sum(dx/r3, axis=1)
        a += a_int

        for i in range(M):
            f_now[i] = cooling_step(v_now[i], S0[i], kl[i], gamma[i], delta[i])
        a += f_now / m

        v_now = v_now + dt*a
        x_now = x_now + dt*v_now

        if step % ht == 0:
            for i in range(M):
                dv = heating_step(v_now[i], S0[i], kl[i], gamma[i], delta[i],
                                  m[i], ips, ht, dt)
                v_now[i] += dv
                heating_log.append((step, i, float(dv)))

        if step % w == 0:
            x[record_index, :] = x_now
            v[record_index, :] = v_now
            f[record_index, :] = f_now
            record_index += 1

    return x, v, f, heating_log
