# ２準位系の時間発展

import math
import numpy as np
from matplotlib import image

from simulation.params import dt

Gamma = 2 * math.pi * 100e3 # 自然放出率(rad/s)
Omega_0 = 1 # 位相を含まないラビ振動数(rad/s)
ramda = 313e-9 # 波長(m)
k = 2 * math.pi / ramda # 端数(rad/m)
c = 299_792_458.0 # 光速(m/s)
somega_0 = k * c # 永年周波数(rad/s)

def phase_shift_x(Omega, x):
    Omega = Omega_0 * np.exp(1j * k * x)
    return Omega

def phase_shift_v(Omega, v):
    Omega = Omega * np.exp(1j * k * v * dt)
    return Omega

def bloph_timeeq_x(rho_gg, rho_ge, x, delta):
    Omega = phase_shift_x(Omega_0, x)

    def f(rho_gg, rho_ge):

        drho_gg = Gamma * (1.0 - rho_gg) + Omega / 2 * (2 * rho_ge.imag)

        drho_eg = -(Gamma / 2 + 1j * delta) * rho_ge.conjugate() + 1j * Omega / 2 * (rho_gg - (1.0 - rho_gg))

        return drho_gg, drho_eg.conjugate()

    k1_gg, k1_ge = f(rho_gg, rho_ge)

    k2_gg, k2_ge = f(
        rho_gg + 0.5 * dt * k1_gg,
        rho_ge + 0.5 * dt * k1_ge
    )

    k3_gg, k3_ge = f(
        rho_gg + 0.5 * dt * k2_gg,
        rho_ge + 0.5 * dt * k2_ge
    )

    k4_gg, k4_ge = f(
        rho_gg + dt * k3_gg,
        rho_ge + dt * k3_ge
    )

    rho_gg += dt / 6 * (k1_gg + 2*k2_gg + 2*k3_gg + k4_gg)
    rho_ge += dt / 6 * (k1_ge + 2*k2_ge + 2*k3_ge + k4_ge)

    rho_ee = np.real(1 - rho_gg)

    return rho_ee, rho_ge

def bloph_timeeq_v(rho_gg, rho_ge, v, delta, Omega):
    Omega = phase_shift_v(Omega, v)

    def f(rho_gg, rho_ge):

        drho_gg = Gamma * (1.0 - rho_gg) - 1j * Omega / 2 * (rho_ge - rho_ge.conjugate())

        drho_eg = -(Gamma / 2 + 1j * delta) * rho_ge.conjugate() + 1j * Omega / 2 * (rho_gg - (1.0 - rho_gg))

        return drho_gg, drho_eg.conjugate()

    k1_gg, k1_ge = f(rho_gg, rho_ge)

    k2_gg, k2_ge = f(
        rho_gg + 0.5 * dt * k1_gg,
        rho_ge + 0.5 * dt * k1_ge
    )

    k3_gg, k3_ge = f(
        rho_gg + 0.5 * dt * k2_gg,
        rho_ge + 0.5 * dt * k2_ge
    )

    k4_gg, k4_ge = f(
        rho_gg + dt * k3_gg,
        rho_ge + dt * k3_ge
    )

    rho_gg += dt / 6 * (k1_gg + 2*k2_gg + 2*k3_gg + k4_gg)
    rho_ge += dt / 6 * (k1_ge + 2*k2_ge + 2*k3_ge + k4_ge)

    rho_ee = np.real(1 - rho_gg)

    return rho_ee, rho_ge, Omega