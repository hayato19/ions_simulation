import numpy as np
import random
import math
from simulation.params import hbar

def calculate_rho(v, S0, kl, gamma, delta):##ρeeの計算
    return S0 / 2 / (S0 + 1 + 4/(gamma**2)*(delta - kl*v)**2)

def cooling_step(v, S0, kl, gamma, delta):##放射圧冷却力の計算
    rho = calculate_rho(v, S0, kl, gamma, delta)
    return float(hbar) * gamma * rho * kl

def heating_step(v, S0, kl, gamma, delta, m, ips, ht, dt):##励起時の加熱の計算
    rho = calculate_rho(v, S0, kl, gamma, delta)
    E = (float(hbar)*kl)**2 / (2*m) * gamma * rho * (1+ips)
    o = np.sqrt(2 * E * ht * dt / m)
    u = math.cos(random.uniform(0, 2*math.pi))
    return o * u
