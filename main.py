import json
import os
import random
import shutil
from datetime import datetime

import numpy as np

from simulation.params import dt, N, w, alpha, eps, ips, ht, hbar, set_particle_params
from simulation.initialize import initialize_arrays_multi
from plotting.plot_x_range import plot_x_range
from plotting.plot_full import plot_full_x, plot_full_f, plot_full_rho
from plotting.plot_fft import plot_fft_all_particles
from plotting.plot_energy import plot_energy
from simulation.spectroscopy import calculate_rho_sp
from plotting.plot_spectroscopy import plot_spectroscopy
from simulation.calculation_t import T_ratio_with_and_without_COM, kB
from plotting.plot_t import plot_t
from simulation.solver_rk4 import rk4_step_multi
from simulation.bloch_spec import calculate_spec_bloch


# Noneの場合は、実行ごとに32 bitの乱数seedを生成して保存する。
# 同じ乱数系列を再現したい場合は、整数を明示的に指定する。
RANDOM_SEED = None

# metadata.json の構造バージョン
SCHEMA_VERSION = 1

# Bloch計算側で使用するワーカー数。
# 現在のmain.pyでは calculate_spec_bloch() に明示的には渡していない。
N_WORKERS = None


def resolve_random_seed(seed):
    """実行に使用する32 bit乱数seedを決定する。"""
    if seed is None:
        return int.from_bytes(os.urandom(4), byteorder="little", signed=False)

    seed = int(seed)
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("RANDOM_SEED must be between 0 and 2**32 - 1")
    return seed


def to_json_list(value):
    """NumPy配列・スカラーをJSONへ保存可能なlistへ変換する。"""
    return np.asarray(value).tolist()


def initial_value_as_particle_list(value, M):
    """初期条件を粒子数MのlistとしてJSON保存できる形にする。"""
    array = np.asarray(value)

    if array.ndim == 0:
        return np.full(M, float(array)).tolist()

    if array.size != M:
        raise ValueError(
            f"initial condition size mismatch: expected {M}, got {array.size}"
        )

    return array.reshape(M).tolist()


def calculate_trap_frequency_hz(k_arr, m_arr):
    """
    k_i = m_i * (2*pi*f_trap)^2 から trap frequency [Hz] を求める。

    現在のJSON仕様では trap_f_Hz を単一値として保存するため、
    全粒子で同一周波数であることを確認する。
    """
    k_values = np.asarray(k_arr, dtype=float)
    m_values = np.asarray(m_arr, dtype=float)

    if k_values.shape != m_values.shape:
        raise ValueError("k_arr and m_arr must have the same shape")

    trap_f_values = np.sqrt(k_values / m_values) / (2.0 * np.pi)

    if not np.allclose(
            trap_f_values,
            trap_f_values[0],
            rtol=1e-12,
            atol=0.0,
    ):
        raise ValueError(
            "trap frequency is not identical for all particles; "
            "current metadata schema expects one trap_f_Hz value"
        )

    return float(trap_f_values[0])


def move_result_file(source_path, destination_path):
    """Bloch計算が生成したファイルを今回のrunフォルダへ移動する。"""
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"result file not found: {source_path}")

    return shutil.move(source_path, destination_path)


def save_metadata(metadata_path, metadata):
    """metadataをUTF-8のJSONとして保存する。"""
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )


def main():
    # ======================================
    # 実行識別・乱数seed設定
    # ======================================

    run_started_at = datetime.now().astimezone()
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")

    random_seed = resolve_random_seed(RANDOM_SEED)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ======================================
    # 初期条件設定
    # ======================================

    M = 2
    x0s = np.linspace(-12e-6, 12e-6, M)
    v0s = 0.0

    m_arr, k_arr, kl_arr, gamma_arr, S0_arr, delta_arr = set_particle_params(M)

    # JSONへ保存する派生パラメータ
    dt_rec = dt * w
    trap_f_Hz = calculate_trap_frequency_hz(k_arr, m_arr)

    # ======================================
    # 1回のsimulation = 1結果フォルダ
    # ======================================

    data_root = "./data"
    run_dir = os.path.join(data_root, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)

    x_path = os.path.join(run_dir, "x.npy")
    v_path = os.path.join(run_dir, "v.npy")
    metadata_path = os.path.join(run_dir, "metadata.json")

    # ======================================
    # 運動シミュレーション
    # ======================================

    t, xM, vM, f = initialize_arrays_multi(M, N, w, dt, x0s, v0s)

    xM, vM, f, heating_log, r, e = rk4_step_multi(
        m_arr, k_arr, xM, vM, f, dt, N, w,
        alpha, eps, S0_arr, kl_arr, gamma_arr, delta_arr,
        ips, ht
    )

    print(f"t_final = {t[-1]:.3e} s")

    # x, vを今回のrunフォルダへ保存
    np.save(x_path, xM)
    np.save(v_path, vM)

    # 可視化例（粒子2、時間範囲3e-5〜t_end）
    # plot_x_range(t, xM, t_start=3e-5, t_end=t[-1], particle_index=2)

    # 可視化(全粒子位置、全時間範囲)
    # plot_full_x(t, xM, save_dir="./figs")

    # 可視化(全粒子の受ける力、全時間範囲)
    # plot_full_f(t, xM, save_dir="./figs")

    # 可視化(全粒子ρ、全時間範囲)
    # plot_full_rho(t, r, save_dir="./figs")

    # 可視化(全粒子のFFT、指定周波数範囲)
    # f_lines = plot_fft_all_particles(t, xM, dt, save_dir="./figs")

    # 可視化(総エネルギー、全範囲)
    # plot_energy(t, e, save_dir="./figs")

    # ======================================
    # 温度による冷却の評価
    # ======================================

    T, T_min, n_sum = T_ratio_with_and_without_COM(
        v=vM,
        m=m_arr[0],
        Gamma=gamma_arr[0],  # [rad/s]
        s0=S0_arr[0],  # dimensionless
    )
    plot_t(t, T, T_min, M, n_sum)

    # ======================================
    # 分光信号のシミュレーション
    # ======================================

    rho_ee_source, rho_int_source, delta_source, scale_rad_s, cal_start_ratio, detuning_num, ramda_m = calculate_spec_bloch(
        M,
        xM,
        vM,
    )

    # bloch_spec.pyが./dataへ生成した結果を同一runフォルダへ集約する。
    rho_ee_path = os.path.join(run_dir, "rho_ee.npy")
    rho_int_path = os.path.join(run_dir, "rho_int.npy")
    delta_path = os.path.join(run_dir, "delta.npy")

    move_result_file(rho_ee_source, rho_ee_path)
    move_result_file(rho_int_source, rho_int_path)
    move_result_file(delta_source, delta_path)

    # ======================================
    # metadata.json 出力
    # ======================================

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": run_started_at.isoformat(timespec="seconds"),
        "simulation": {
            "M": int(M),
            "N": int(N),
            "w": int(w),
            "dt_s": float(dt),
            "dt_rec_s": float(dt_rec),
            "random_seed": int(random_seed),
        },
        "initial_conditions": {
            "x0s_m": initial_value_as_particle_list(x0s, M),
            "v0s_m_s": initial_value_as_particle_list(v0s, M),
        },
        "motion_parameters": {
            "m_arr_kg": to_json_list(m_arr),
            "trap_f_Hz": float(trap_f_Hz),
            "gamma_arr_rad_s": to_json_list(gamma_arr),
            "S0_arr": to_json_list(S0_arr),
            "delta_arr_rad_s": to_json_list(delta_arr),
            "ht": float(ht),
            "alpha_N_m2": float(alpha),
        },
        "constants": {
            "hbar_J_s": float(hbar),
        },
        "files": {
            "x": "x.npy",
            "v": "v.npy",
            "rho_ee": "rho_ee.npy",
            "rho_int": "rho_int.npy",
            "delta": "delta.npy",
        },
    }

    save_metadata(metadata_path, metadata)

    print(f"saved run directory: {run_dir}")
    print(f"saved metadata: {metadata_path}")

    # ======================================
    # LaTeX用パラメータ出力
    # ======================================

    with open("tex/sections/params.tex", "w", encoding="utf-8") as f:
        f.write(rf"\newcommand{{\Mval}}{{{M}}}" + "\n")
        f.write(rf"\newcommand{{\dtval}}{{{dt}}}" + "\n")
        f.write(rf"\newcommand{{\hbarval}}{{{hbar}}}" + "\n")
        f.write(rf"\newcommand{{\kval}}{{{k_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\kBval}}{{{kB}}}" + "\n")
        f.write(rf"\newcommand{{\klval}}{{{kl_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\mval}}{{{m_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\Sval}}{{{S0_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\alphaval}}{{{alpha}}}" + "\n")
        f.write(rf"\newcommand{{\gammaval}}{{{gamma_arr[0]}}}" + "\n")
        f.write(rf"\newcommand{{\htval}}{{{ht}}}" + "\n")
        f.write(rf"\newcommand{{\deltaval}}{{{delta_arr[0]}}}" + "\n")
        # f.write(rf"\newcommand{{\gammadval}}{{{gammad}}}" + "\n")
        # f.write(rf"\newcommand{{\Sspval}}{{{s_sp}}}" + "\n")
        # f.write(rf"\newcommand{{\fvala}}{{{f_lines[0]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvalb}}{{{f_lines[1]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvalc}}{{{f_lines[2]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvald}}{{{f_lines[3]}}}" + "\n")
        # f.write(rf"\newcommand{{\fvale}}{{{f_lines[4]}}}" + "\n")


if __name__ == "__main__":
    main()
