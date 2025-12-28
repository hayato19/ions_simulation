import numpy as np

# ---- physical constants ----
kB   = 1.380649e-23        # J/K
hbar = 1.054571817e-34     # J·s

def T_ratio_with_and_without_COM(
    v: np.ndarray,
    m: float,
    Gamma,
    s0,
    steps=None,
    j=2/5,
):
    Gamma = float(Gamma)
    s0 = float(s0)
    j = float(j)
    m = float(m)
    """
    Compute T_th / T_min for 1D classical multi-particle trajectories (kinetic only).

    Parameters
    ----------
    v : ndarray, shape (N_steps, M)
        velocities [m/s]
    m : float
        particle mass [kg]
    Gamma : float
        natural linewidth (angular) [rad/s]
    s0 : float
        saturation parameter (dimensionless), s = s0
    steps : None, slice, or 1D index array
        time indices to evaluate (None -> all)
    j : float
        geometry factor (default 2/5 for dipole radiation)
    remove_com : bool
        subtract COM velocity at each step (recommended)

    Returns
    -------
    ratio : float
        T_th / T_min
    T_th : float
        thermal temperature [K]
    T_min : float
        Doppler minimum temperature [K]
    """
    # --- select steps ---
    vv = v[steps] if steps is not None else v

    # ---- raw ----
    v2_raw = np.mean(vv ** 2)
    T_raw = (m / kB) * v2_raw

    # ---- COM removed ----
    v_cm = vv.mean(axis=1, keepdims=True)
    dv = vv - v_cm
    v2_th = np.mean(dv ** 2)
    T_th = (m / kB) * v2_th

    # ---- Doppler limit ----
    T_min = (hbar * Gamma * np.sqrt(1.0 + s0) / (4.0 * kB)) * (1.0 + j)

    return {
        "T_raw": T_raw,
        "T_th": T_th,
        "T_min": T_min,
        "ratio_raw": T_raw / T_min,
        "ratio_th": T_th / T_min,
    }