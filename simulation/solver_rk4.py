import numpy as np
from simulation.forces import cooling_step

def rk4_step_multi(m, k, x, v, f, dt, N, w, alpha, eps, S0, kl, gamma, delta):

    M = x.shape[1]
    x_now = x[0, :].copy()
    v_now = v[0, :].copy()
    f_now = np.zeros(M)

    record_index = 1
    total_steps = N*w

    def calc_accel(xn, vn):
        a = -(k/m)*xn
        dx = xn[np.newaxis, :] - xn[:, np.newaxis]
        np.fill_diagonal(dx, 0.0)
        r3 = np.abs(dx)**3 + eps**3
        a += -(alpha/m)*np.sum(dx/r3, axis=1)

        f_tmp = np.array([cooling_step(vn[i], S0[i], kl[i], gamma[i], delta[i])
                          for i in range(M)])
        a += f_tmp/m
        return a, f_tmp

    for step in range(1, total_steps+1):

        a1, f1 = calc_accel(x_now, v_now)
        k1x, k1v = v_now, a1

        a2, _ = calc_accel(x_now+0.5*dt*k1x, v_now+0.5*dt*k1v)
        k2x, k2v = v_now+0.5*dt*k1v, a2

        a3, _ = calc_accel(x_now+0.5*dt*k2x, v_now+0.5*dt*k2v)
        k3x, k3v = v_now+0.5*dt*k2v, a3

        a4, f4 = calc_accel(x_now+dt*k3x, v_now+dt*k3v)
        k4x, k4v = v_now+dt*k3v, a4

        x_now += (dt/6)*(k1x+2*k2x+2*k3x+k4x)
        v_now += (dt/6)*(k1v+2*k2v+2*k3v+k4v)

        f_now = f4

        if step % w == 0:
            x[record_index, :] = x_now
            v[record_index, :] = v_now
            f[record_index, :] = f_now
            record_index += 1

    return x, v, f
