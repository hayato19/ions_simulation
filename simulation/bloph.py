# ２準位系の時間発展

import math
import numpy as np
from matplotlib import image

from simulation.params import dt

s_gamma = 10
Gamma = 2 * math.pi * s_gamma * 1000  # 自然放出率(rad/s)
Omega_0 = 1  # 位相を含まないラビ振動数(rad/s)
ramda = 313e-9  # 波長(m)
k = 2 * math.pi / ramda  # 波数(rad/m)
c = 299_792_458.0  # 光速(m/s)
somega_0 = k * c  # 永年周波数(rad/s)


def phase_shift_x(Omega, x):
    Omega = Omega_0 * np.exp(1j * k * x)
    return Omega


def phase_shift_v(Omega, v):
    Omega = Omega * np.exp(1j * k * v * dt)
    return Omega


def bloph_timeeq_x(rho_ee, rho_ge, x, delta):

    Omega = phase_shift_x(Omega_0, x)

    def f(rho_ee_value, rho_ge_value):
        population_difference = 1.0 - 2.0 * rho_ee_value

        drho_ee = (
                -Gamma * rho_ee_value
                + 0.5j
                * (
                        Omega * rho_ge_value
                        - Omega.conjugate() * rho_ge_value.conjugate()
                )
        )
        drho_ee = float(np.real(drho_ee))

        drho_ge = (
                -(Gamma / 2.0 - 1j * delta) * rho_ge_value
                - 0.5j * Omega.conjugate() * population_difference
        )

        return drho_ee, drho_ge

    k1_ee, k1_ge = f(rho_ee, rho_ge)

    k2_ee, k2_ge = f(
        rho_ee + 0.5 * dt * k1_ee,
        rho_ge + 0.5 * dt * k1_ge,
        )

    k3_ee, k3_ge = f(
        rho_ee + 0.5 * dt * k2_ee,
        rho_ge + 0.5 * dt * k2_ge,
        )

    k4_ee, k4_ge = f(
        rho_ee + dt * k3_ee,
        rho_ge + dt * k3_ge,
        )

    rho_ee += dt / 6.0 * (
            k1_ee + 2.0 * k2_ee + 2.0 * k3_ee + k4_ee
    )
    rho_ge += dt / 6.0 * (
            k1_ge + 2.0 * k2_ge + 2.0 * k3_ge + k4_ge
    )

    return rho_ee, rho_ge


def bloph_timeeq_v(rho_ee, rho_ge, v, delta, Omega):

    Omega = phase_shift_v(Omega, v)

    def f(rho_ee_value, rho_ge_value):
        population_difference = 1.0 - 2.0 * rho_ee_value

        drho_ee = (
                -Gamma * rho_ee_value
                + 0.5j
                * (
                        Omega * rho_ge_value
                        - Omega.conjugate() * rho_ge_value.conjugate()
                )
        )
        drho_ee = float(np.real(drho_ee))

        drho_ge = (
                -(Gamma / 2.0 - 1j * delta) * rho_ge_value
                - 0.5j * Omega.conjugate() * population_difference
        )

        return drho_ee, drho_ge

    k1_ee, k1_ge = f(rho_ee, rho_ge)

    k2_ee, k2_ge = f(
        rho_ee + 0.5 * dt * k1_ee,
        rho_ge + 0.5 * dt * k1_ge,
        )

    k3_ee, k3_ge = f(
        rho_ee + 0.5 * dt * k2_ee,
        rho_ge + 0.5 * dt * k2_ge,
        )

    k4_ee, k4_ge = f(
        rho_ee + dt * k3_ee,
        rho_ge + dt * k3_ge,
        )

    rho_ee += dt / 6.0 * (
            k1_ee + 2.0 * k2_ee + 2.0 * k3_ee + k4_ee
    )
    rho_ge += dt / 6.0 * (
            k1_ge + 2.0 * k2_ge + 2.0 * k3_ge + k4_ge
    )

    return rho_ee, rho_ge, Omega