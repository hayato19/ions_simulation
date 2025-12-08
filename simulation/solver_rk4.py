import numpy as np
from simulation.forces import cooling_step, heating_step, calculate_rho


def rk4_step_multi(m, k, x, v, f, dt, N, w, alpha, eps,
                   S0, kl, gamma, delta, ips, ht):

    M = x.shape[1]
    x_now = x[0, :].copy()
    v_now = v[0, :].copy()
    f_now = np.zeros(M)
    r = np.zeros_like(x)

    record_index = 1
    heating_log = []

    # --------------------------------------
    # Acceleration function for RK4
    # --------------------------------------
    def calc_accel(xn, vn):
        a = -(k/m) * xn
        dx = xn[np.newaxis, :] - xn[:, np.newaxis]
        np.fill_diagonal(dx, 0.0)
        r3 = np.abs(dx)**3 + eps**3
        a_int = -(alpha/m)*np.sum(dx/r3, axis=1)
        a += a_int

        f_tmp = np.zeros(M)
        for i in range(M):
            f_tmp[i] = cooling_step(vn[i], S0[i], kl[i], gamma[i], delta[i])
        a += f_tmp / m
        return a, f_tmp

    # --------------------------------------
    # Time evolution
    # --------------------------------------
    total_steps = N * w
    for step in range(1, total_steps + 1):

        # k1
        a1, f1 = calc_accel(x_now, v_now)
        k1x = v_now
        k1v = a1

        # k2
        x2 = x_now + 0.5 * dt * k1x
        v2 = v_now + 0.5 * dt * k1v
        a2, _ = calc_accel(x2, v2)
        k2x = v2
        k2v = a2

        # k3
        x3 = x_now + 0.5 * dt * k2x
        v3 = v_now + 0.5 * dt * k2v
        a3, _ = calc_accel(x3, v3)
        k3x = v3
        k3v = a3

        # k4
        x4 = x_now + dt * k3x
        v4 = v_now + dt * k3v
        a4, f4 = calc_accel(x4, v4)
        k4x = v4
        k4v = a4

        # RK4 update
        x_now = x_now + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        v_now = v_now + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)

        f_now = f4.copy()

        # -------------------------
        # Heating
        # -------------------------
        if step % ht == 0:
            for i in range(M):
                dv = heating_step(
                    v_now[i], S0[i], kl[i], gamma[i], delta[i],
                    m[i], ips, ht, dt
                )
                v_now[i] += dv
                heating_log.append((step, i, float(dv)))

        # -------------------------
        # Recording
        # -------------------------
        if step % w == 0:
            x[record_index, :] = x_now
            v[record_index, :] = v_now
            f[record_index, :] = f_now
            r[record_index, :] = calculate_rho(v_now[:], S0[:], kl[:], gamma[:], delta[:])
            record_index += 1

    return x, v, f, heating_log