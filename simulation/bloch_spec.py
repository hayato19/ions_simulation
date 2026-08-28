import numpy as np
import time
import math
from simulation.params import dt, N, w, ht
from simulation.bloph import phase_shift_x, phase_shift_v, bloph_timeeq_x, bloph_timeeq_v, Gamma
import matplotlib.pyplot as plt
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# 各ワーカープロセスで共有する読み取り専用データ
_WORKER_X = None
_WORKER_V = None
_WORKER_M = None
_WORKER_START_STEP = None
_WORKER_END_STEP = None
_WORKER_DT_REC = None


def print_log(message):
    """現在時刻を付けてログを表示する"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


def _initialize_bloch_worker(x, v, M, start_step, end_step, dt_rec):
    """各ワーカープロセスの起動時に共通データを設定する。"""
    global _WORKER_X
    global _WORKER_V
    global _WORKER_M
    global _WORKER_START_STEP
    global _WORKER_END_STEP
    global _WORKER_DT_REC

    _WORKER_X = x
    _WORKER_V = v
    _WORKER_M = M
    _WORKER_START_STEP = start_step
    _WORKER_END_STEP = end_step
    _WORKER_DT_REC = dt_rec


def _calculate_one_delta(task):
    """1つの離調について、全粒子のBloch方程式を計算する。"""
    i, delta_i = task

    n_steps_save = _WORKER_END_STEP - _WORKER_START_STEP
    rho_int_row = np.zeros(_WORKER_M, dtype=np.float64)
    rho_ee_row = np.empty((_WORKER_M, n_steps_save), dtype=np.float64)
    errors = []

    for j in range(_WORKER_M):
        sum_rho = 0.0

        # 初期条件: rho_gg = 1, rho_ee = 0
        # rho_ggは保持せず、常に rho_gg = 1 - rho_ee とする。
        rho_ee = 0.0
        rho_ge = 0.0 + 0.0j
        Omega = 1.0 + 0.0j

        for step in range(_WORKER_START_STEP, _WORKER_END_STEP):
            save_index = step - _WORKER_START_STEP

            try:
                rho_ee, rho_ge, Omega = bloph_timeeq_v(
                    rho_ee,
                    rho_ge,
                    _WORKER_V[step - 1, j],
                    delta_i,
                    Omega,
                )

                if not np.isfinite(rho_ge):
                    raise FloatingPointError("rho_ge became non-finite")

                if not np.isfinite(rho_ee):
                    raise FloatingPointError("rho_ee became non-finite")

                if abs(rho_ge) > 1e6:
                    raise FloatingPointError("rho value diverged")

                rho_ee_real = float(np.real(rho_ee))
                rho_ee_row[j, save_index] = rho_ee_real
                sum_rho += rho_ee_real * _WORKER_DT_REC

            except FloatingPointError as e:
                rho_ee_row[j, save_index:] = np.nan
                sum_rho = np.nan

                errors.append(
                    {
                        "reason": str(e),
                        "i": i,
                        "j": j,
                        "step": step,
                        "delta": delta_i,
                        "x": _WORKER_X[step - 1, j],
                        "v": _WORKER_V[step - 1, j],
                        "rho_ge": rho_ge,
                        "rho_ee": rho_ee,
                    }
                )
                break

        rho_int_row[j] = sum_rho

    return i, rho_int_row, rho_ee_row, errors


def _resolve_n_workers(n_workers, n_tasks):
    """指定値または論理プロセッサ数からワーカー数を決定する。"""
    logical_cores = os.cpu_count() or 1

    if n_workers is None:
        # ノートPCでもCPUを占有しすぎない保守的な自動設定
        n_workers = max(1, logical_cores // 2)

    if not isinstance(n_workers, int):
        raise TypeError("n_workers must be int or None")

    if n_workers <= 0:
        raise ValueError("n_workers must be greater than 0")

    return min(n_workers, n_tasks), logical_cores


def calculate_spec_bloch(M, x, v, n_workers=None):
    c = 299_792_458.0
    ramda = 313e-9
    omega_0 = 2 * math.pi * c / ramda

    scale = 0.5e7 * 2 * math.pi
    detuning_num = 3000
    delta = np.linspace(-scale, scale, detuning_num)
    cal_start_ratio = 7 / 15

    mode = "each"
    dt_rec = dt * w

    start_step = int(N * cal_start_ratio)
    end_step = N
    n_steps_save = end_step - start_step

    n_workers, logical_cores = _resolve_n_workers(
        n_workers=n_workers,
        n_tasks=len(delta),
    )

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
        shape=(len(delta), M, n_steps_save),
    )

    # delta も保存
    delta_path = os.path.join(save_dir, f"delta_{timestamp}.npy")
    np.save(delta_path, delta)

    print_log("start Cal_rho_ee")
    print_log(f"logical processors = {logical_cores}")
    print_log(f"parallel workers = {n_workers}")

    tasks = [
        (i, float(delta_i))
        for i, delta_i in enumerate(delta)
    ]

    completed_count = 0

    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_initialize_bloch_worker,
            initargs=(
                    x,
                    v,
                    M,
                    start_step,
                    end_step,
                    dt_rec,
            ),
    ) as executor:
        futures = [
            executor.submit(
                _calculate_one_delta,
                task,
            )
            for task in tasks
        ]

        for future in as_completed(futures):
            (
                i,
                rho_int_row,
                rho_ee_row,
                errors,
            ) = future.result()

            # memmapへの書き込みは親プロセスだけが行う。
            rho_int[i, :] = rho_int_row
            rho_ee_all[i, :, :] = rho_ee_row

            completed_count += 1

            if errors:
                for error in errors:
                    print_log(
                        "Numerical error detected"
                    )
                    print(
                        "reason =",
                        error["reason"],
                    )
                    print(
                        "i =",
                        error["i"],
                    )
                    print(
                        "j =",
                        error["j"],
                    )
                    print(
                        "step =",
                        error["step"],
                    )
                    print(
                        "delta =",
                        error["delta"],
                    )
                    print(
                        "x =",
                        error["x"],
                    )
                    print(
                        "v =",
                        error["v"],
                    )
                    print(
                        "rho_ge =",
                        error["rho_ge"],
                    )
                    print(
                        "rho_ee =",
                        error["rho_ee"],
                    )

            if (
                    completed_count % 10 == 0
                    or completed_count == len(delta)
            ):
                print_log(
                    f"calc {completed_count}/{len(delta)}"
                )

            if completed_count % 20 == 0:
                rho_ee_all.flush()

    rho_ee_all.flush()

    # rho_int 保存
    rho_int_path = os.path.join(save_dir, f"rho_int_{timestamp}.npy")
    np.save(rho_int_path, rho_int)

    print("saved rho_ee:", rho_ee_path)
    print("saved rho_int:", rho_int_path)
    print("saved delta:", delta_path)

    return rho_ee_path, rho_int_path, delta_path, scale, cal_start_ratio, detuning_num, ramda
