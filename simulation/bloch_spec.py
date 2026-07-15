import numpy as np
import time
import math
from simulation.params import dt, N, w, ht
from simulation.bloph import phase_shift_x, phase_shift_v, bloph_timeeq_x, bloph_timeeq_v, Gamma
import matplotlib.pyplot as plt
import os
from datetime import datetime

def print_log(message):
    """現在時刻を付けてログを表示する"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)

def calculate_spec_bloch(M, x, v):
    c = 299_792_458.0
    ramda = 313e-9
    omega_0 = 2 * math.pi * c / ramda

    scale = 0.5e7 * 2 * math.pi
    delta = np.linspace(-scale, scale, 300)

    mode = "each"
    dt_rec = dt * w

    start_step = N // 15
    end_step = N
    n_steps_save = end_step - start_step

    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 積分値
    rho_int = np.zeros((len(delta), M))
    rho_ee_path = os.path.join(save_dir, f"rho_ee_time_{timestamp}.npy")

    rho_ee_all = np.lib.format.open_memmap(
        rho_ee_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(delta), M, n_steps_save)
    )

    # delta も保存
    delta_path = os.path.join(save_dir, f"delta_{timestamp}.npy")
    np.save(delta_path, delta)

    print_log(f"start Cal_rho_ee\n")
    for i in range(len(delta)):  # レーザー周波数ごと
        print_log(f"Cal delta = {i}\n")
        for j in range(M):
            print_log(f"Cal particle = {j}\n")
            sum_rho = 0.0

            rho_gg = 1.0
            rho_ge = 0.0 + 0.0j
            Omega = 1
            print_log(f"call solving me system at [{i}][{j}]")
            for step in range(start_step, end_step):
                save_index = step - start_step

                try:
                    rho_ee, rho_ge, Omega = bloph_timeeq_v(
                        rho_gg,
                        rho_ge,
                        v[step - 1, j],
                        delta[i],
                        Omega
                    )

                    if not np.isfinite(rho_ge):
                        raise FloatingPointError("rho_ge became non-finite")

                    if not np.isfinite(rho_ee):
                        raise FloatingPointError("rho_ee became non-finite")

                    if abs(rho_ge) > 1e6:
                        raise FloatingPointError("rho value diverged")

                    rho_ee_all[i, j, save_index] = rho_ee.real
                    sum_rho += rho_ee.real * dt_rec

                except FloatingPointError as e:
                    print("Numerical error detected")
                    print("reason =", e)
                    print("i =", i)
                    print("j =", j)
                    print("step =", step)
                    print("delta =", delta[i])
                    print("x =", x[step - 1, j])
                    print("rho_ge =", rho_ge)
                    print("rho_ee =", rho_ee)

                    rho_ee_all[i, j, save_index:] = np.nan
                    sum_rho = np.nan
                    break
            print_log(f"finish solving me system")
            print_log(f"Cal int")
            rho_int[i, j] = sum_rho
            print_log(f"end of Cal int")

        if i % 50 == 0:
            print(f"calc {i}/{len(delta)}")

    # rho_int 保存
    rho_int_path = os.path.join(save_dir, f"rho_int_{timestamp}.npy")
    np.save(rho_int_path, rho_int)

    print("saved rho_ee:", rho_ee_path)
    print("saved rho_int:", rho_int_path)
    print("saved delta:", delta_path)

    return rho_ee_path, rho_int_path, delta_path