"""
bloch_spec_qutip.py

添付 bloch_spec.py / bloph.py に準拠した QuTiP 版 OBE 計算コード。

特徴:
    - phase_source の選択肢は削除し、既存 bloph.py の bloph_timeeq_v と同じく速度 v のみを使う。
    - x, v, N, dt, w, start_step, end_step, delta, 保存ファイル名は添付 bloch_spec.py に準拠する。
    - 出力ファイル:
        ./data/rho_ee_time_<timestamp>.npy
        ./data/rho_int_<timestamp>.npy
        ./data/delta_<timestamp>.npy

既存コードでの使い方:
    from simulation.bloch_spec_qutip import calculate_spec_bloch_qutip

    rho_ee_path, rho_int_path, delta_path = calculate_spec_bloch_qutip(M, x, v)

または、既存の呼び出し名に合わせたい場合:
    from simulation.bloch_spec_qutip import calculate_spec_bloch
"""

from __future__ import annotations

import math
import time
import os
import traceback
from datetime import datetime

import numpy as np
from qutip import basis, mesolve

from simulation.params import dt, N, w, ht
from simulation.bloph import Gamma

# bloph.py に準拠した定数
# s_gamma = 10
# Gamma = 2 * math.pi * s_gamma * 1000.0  # 自然放出率 [rad/s]
Omega_0 = 1.0                           # 位相を含まない Rabi 周波数 [rad/s]
ramda = 313e-9                          # 波長 [m]
k = 2 * math.pi / ramda                 # 波数 [rad/m]
c = 299_792_458.0                       # 光速 [m/s]
somega_0 = k * c                        # 参考値 [rad/s]

def print_log(message):
    """現在時刻を付けてログを表示する"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)

def _build_two_level_system():
    """
    2準位系の演算子を作る。

    |g> = basis(2, 0)
    |e> = basis(2, 1)

    sigma_plus  = |e><g|
    sigma_minus = |g><e|
    Pe          = |e><e|

    崩壊演算子:
        C = sqrt(Gamma) |g><e|
    """
    g = basis(2, 0)
    e = basis(2, 1)

    sigma_plus = e * g.dag()
    sigma_minus = g * e.dag()
    Pe = e * e.dag()

    rho0 = g * g.dag()
    c_ops = [np.sqrt(Gamma) * sigma_minus]

    return sigma_plus, sigma_minus, Pe, rho0, c_ops


def _complex_interp_func(tlist: np.ndarray, values: np.ndarray):
    """
    QuTiP の時間依存係数に渡すための補間関数。

    values は複素数配列なので、実部と虚部に分けて補間する。
    """
    tlist = np.asarray(tlist, dtype=float)
    values = np.asarray(values, dtype=np.complex128)

    real_values = values.real
    imag_values = values.imag

    def coeff(t, **kwargs):
        real_part = np.interp(t, tlist, real_values)
        imag_part = np.interp(t, tlist, imag_values)
        return real_part + 1j * imag_part

    return coeff


def _make_omega_from_v(v_segment: np.ndarray) -> np.ndarray:
    """
    添付 bloph.py の phase_shift_v に対応する Ω(t) を作る。

    添付 bloph.py では、各 step で

        Omega = Omega * exp(1j * k * v * dt)

    としている。

    ここでは、同じ更新をあらかじめ配列として作り、
    QuTiP の時間依存ハミルトニアン係数として使う。

    Parameters
    ----------
    v_segment:
        shape = (n_steps_save,)
        添付 bloch_spec.py と同じく v[step - 1, j] から作った速度列。

    Returns
    -------
    omega_values:
        shape = (n_steps_save + 1,)
        各 OBE 時刻での Ω。最後の要素は補間用に最後の値を重複させる。
    """
    v_segment = np.asarray(v_segment, dtype=float)

    omega_each_step = np.empty(len(v_segment), dtype=np.complex128)

    omega = complex(Omega_0)

    for n, v_now in enumerate(v_segment):
        omega = omega * np.exp(1j * k * v_now * dt)
        omega_each_step[n] = omega

    if len(omega_each_step) == 0:
        raise ValueError("v_segment is empty.")

    # mesolve では tlist と係数配列の時刻数を合わせるため、
    # 最後の値を1つ追加して n_steps_save + 1 点にする。
    omega_values = np.empty(len(v_segment) + 1, dtype=np.complex128)
    omega_values[:-1] = omega_each_step
    omega_values[-1] = omega_each_step[-1]

    return omega_values


def _solve_one_particle_one_delta(
        delta_i: float,
        v_segment: np.ndarray,
        tlist: np.ndarray,
        system,
        solver_method: str = "adams",
        rtol: float = 1e-2,
        atol: float = 1e-8,
) -> np.ndarray:

    sigma_plus, sigma_minus, Pe, rho0, c_ops = system

    n_steps_save = len(v_segment)

    if len(tlist) != n_steps_save + 1:
        raise ValueError(
            f"tlist length mismatch: len(tlist)={len(tlist)}, "
            f"n_steps_save+1={n_steps_save + 1}"
        )

    # 変更: QuTiP に渡す時間軸が有限・単調増加かを事前確認する。
    if len(tlist) < 2:
        raise ValueError(f"tlist must contain at least 2 points: len={len(tlist)}")

    if not np.all(np.isfinite(tlist)):
        raise ValueError("tlist contains non-finite values")

    tdiff = np.diff(tlist)
    if np.any(tdiff <= 0):
        raise ValueError(
            "tlist must be strictly increasing: "
            f"min diff={np.min(tdiff)}"
        )

    if solver_method not in ("adams", "bdf"):
        raise ValueError(
            f"solver_method must be 'adams' or 'bdf', "
            f"but got {solver_method!r}"
        )

    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError(f"rtol must be positive and finite: rtol={rtol}")

    if not np.isfinite(atol) or atol <= 0:
        raise ValueError(f"atol must be positive and finite: atol={atol}")

    omega_values = _make_omega_from_v(v_segment)
    omega_coeff = _complex_interp_func(tlist, omega_values)

    def omega_conj_coeff(t, **kwargs):
        return np.conjugate(omega_coeff(t, **kwargs))

    H = [
        delta_i * Pe,
        [-0.5 * sigma_plus, omega_coeff],
        [-0.5 * sigma_minus, omega_conj_coeff],
        ]

    result = mesolve(
        H,
        rho0,
        tlist,
        c_ops=c_ops,
        e_ops=[Pe],
        options={
            "store_states": False,
            "method": solver_method,
            "nsteps": 10000,
            "rtol": rtol,
            "atol": atol,
        },
    )

    rho_ee_t = np.asarray(result.expect[0], dtype=float)

    return rho_ee_t[1:]

def calculate_spec_bloch_qutip(M: int, x: np.ndarray, v: np.ndarray):
    """
    添付 bloch_spec.py の calculate_spec_bloch(M, x, v) に準拠した QuTiP 版。

    Parameters
    ----------
    M:
        粒子数。
    x:
        位置配列。添付コードに合わせて shape = (N, M) を想定。
        QuTiP 版では計算には使わないが、エラー表示と shape 確認のため受け取る。
    v:
        速度配列。shape = (N, M) を想定。

    Returns
    -------
    rho_ee_path, rho_int_path, delta_path:
        添付 bloch_spec.py と同じ形式で保存したファイルパス。
    """
    x = np.asarray(x)
    v = np.asarray(v)

    if x.ndim != 2 or v.ndim != 2:
        raise ValueError(f"x and v must be 2D arrays. x.shape={x.shape}, v.shape={v.shape}")

    if x.shape != v.shape:
        raise ValueError(f"x.shape and v.shape must match. x.shape={x.shape}, v.shape={v.shape}")

    if x.shape[0] < N or v.shape[0] < N:
        raise ValueError(f"x and v must have at least N rows. N={N}, x.shape={x.shape}, v.shape={v.shape}")

    if x.shape[1] < M or v.shape[1] < M:
        raise ValueError(f"x and v must have at least M columns. M={M}, x.shape={x.shape}, v.shape={v.shape}")

    # 添付 bloch_spec.py に準拠
    scale = 0.5e7 * 2 * math.pi
    delta = np.linspace(-scale, scale, 300)

    dt_rec = dt * w

    start_step = N // 15
    end_step = N
    n_steps_save = end_step - start_step

    # 変更: 時間刻みと保存区間を QuTiP 呼び出し前に検証する。
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be positive and finite: dt={dt}")

    if n_steps_save <= 0:
        raise ValueError(
            f"n_steps_save must be positive: "
            f"start_step={start_step}, end_step={end_step}"
        )

    # QuTiP 積分器設定
    solver_method = "bdf"
    solver_rtol = 1e-5
    solver_atol = 1e-10

    tlist = np.arange(n_steps_save + 1, dtype=float) * dt
    system = _build_two_level_system()

    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rho_int = np.zeros((len(delta), M))

    rho_ee_path = os.path.join(save_dir, f"rho_ee_time_{timestamp}.npy")

    rho_ee_all = np.lib.format.open_memmap(
        rho_ee_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(delta), M, n_steps_save),
    )

    delta_path = os.path.join(save_dir, f"delta_{timestamp}.npy")
    np.save(delta_path, delta)

    print("=== QuTiP OBE calculation ===")
    print(f"M = {M}")
    print(f"delta points = {len(delta)}")
    print(f"start_step = {start_step}")
    print(f"end_step = {end_step}")
    print(f"n_steps_save = {n_steps_save}")
    print(f"dt for OBE solver = {dt}")
    print(f"dt_rec for spectrum integration = {dt_rec}")
    print(f"Gamma = {Gamma}")
    print(f"Omega_0 = {Omega_0}")
    print(f"solver_method = {solver_method}")  # 変更
    print(f"solver_rtol = {solver_rtol}")      # 変更
    print(f"solver_atol = {solver_atol}")      # 変更

    print_log(f"start Cal_rho_ee\n")
    for i in range(len(delta)):
        print_log(f"Cal delta = {i}\n")
        for j in range(M):
            print_log(f"Cal particle = {j}\n")
            try:
                # 添付 bloch_spec.py では bloph_timeeq_v に v[step - 1, j] を渡している。
                # したがって、QuTiP 版でも同じ速度列を使う。
                v_segment = v[start_step - 1:end_step - 1, j]

                print_log(f"call solving me system at [{i}][{j}]")
                rho_ee_t = _solve_one_particle_one_delta(
                    delta[i],
                    v_segment,
                    tlist,
                    system,
                    solver_method=solver_method,  # 変更: calculate側の設定を反映
                    rtol=solver_rtol,             # 変更: calculate側の設定を反映
                    atol=solver_atol,             # 変更: calculate側の設定を反映
                )
                print_log(f"finish solving me system")
                if len(rho_ee_t) != n_steps_save:
                    raise FloatingPointError(
                        f"rho_ee_t length mismatch: {len(rho_ee_t)} != {n_steps_save}"
                    )

                if not np.all(np.isfinite(rho_ee_t)):
                    raise FloatingPointError("rho_ee became non-finite")

                if np.nanmax(np.abs(rho_ee_t)) > 1e6:
                    raise FloatingPointError("rho value diverged")

                rho_ee_all[i, j, :] = rho_ee_t.real

                # 添付 bloch_spec.py の sum_rho += rho_ee.real * dt_rec に準拠。
                # 台形積分ではなく単純和にして出力の意味をそろえる。
                print_log(f"Cal int")
                rho_int[i, j] = np.sum(rho_ee_t.real) * dt_rec
                print_log(f"end of Cal int")

            except FloatingPointError as e:
                print("Numerical error detected")
                print("reason =", e)
                print("i =", i)
                print("j =", j)
                print("delta =", delta[i])
                print("x =", x[start_step - 1, j])
                print("v =", v[start_step - 1, j])

                rho_ee_all[i, j, :] = np.nan
                rho_int[i, j] = np.nan

            except Exception as e:
                print("Unexpected error detected")
                print("reason =", repr(e))
                print("i =", i)
                print("j =", j)
                print("delta =", delta[i])
                print("x =", x[start_step - 1, j])
                print("v =", v[start_step - 1, j])

                print("----- traceback -----")
                traceback.print_exc()
                print("---------------------")

                rho_ee_all[i, j, :] = np.nan
                rho_int[i, j] = np.nan

        if i % 50 == 0:
            print(f"calc {i}/{len(delta)}")

    rho_ee_all.flush()

    rho_int_path = os.path.join(save_dir, f"rho_int_{timestamp}.npy")
    print_log(f"Save rho_int")
    np.save(rho_int_path, rho_int)
    print_log(f"end of save rho_int")
    print("saved rho_ee:", rho_ee_path)
    print("saved rho_int:", rho_int_path)
    print("saved delta:", delta_path)

    return rho_ee_path, rho_int_path, delta_path


# 既存コードの呼び出し名に合わせたい場合に備えて alias を置く。
# これにより、from simulation.bloch_spec_qutip import calculate_spec_bloch も可能。
calculate_spec_bloch = calculate_spec_bloch_qutip
