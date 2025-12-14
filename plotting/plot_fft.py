import numpy as np
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime

def plot_fft_all_particles(t, xM, dt, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"fft_all_particles_{timestamp}.png"
    save_hist_name = f"fft_peak_hist_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)
    save_hist_path = os.path.join(save_dir, save_hist_name)
    os.makedirs(save_dir, exist_ok=True)

    Nrec, M = xM.shape

    dt_eff = t[1] - t[0]
    freq = np.fft.fftfreq(Nrec, dt_eff)

    plt.figure(figsize=(10, 6))
    mask_pos = freq > 0

    max_amp = 0.0

    # 各粒子の上位5ピーク周波数を溜めるリスト
    peak_freq_global = []

    for j in range(M):
        X = np.fft.fft(xM[:, j])
        amp = np.abs(X) / Nrec

        f_pos = freq[mask_pos]
        amp_pos = amp[mask_pos]

        # プロット用最大振幅
        max_amp = max(max_amp, np.max(amp_pos))

        # FFT プロット
        plt.plot(f_pos, amp_pos, label=f"p{j}")

        # 振幅で降順ソート → 上位5つを抽出
        idx_sorted = np.argsort(amp_pos)[::-1]
        top5_idx = idx_sorted[:5]

        top5_freqs = f_pos[top5_idx]
        peak_freq_global.extend(top5_freqs.tolist())

    plt.xlim(0, 6e6)
    if max_amp > 0:
        plt.ylim(0, max_amp * 1.2)

    # 任意のfに直線を描画
    f_lines = [1, math.sqrt(3), math.sqrt(5.818), math.sqrt(9.332), math.sqrt(13.47)]  # 仮の周波数 [Hz]
    for f_line in f_lines:
        plt.axvline(
            f_line * 1e6,
            color="black",
            linestyle="-",
            linewidth=1.0,
            alpha=0.7
        )

    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved FFT figure:", save_path)

    # # ヒストグラム表示
    # plt.figure(figsize=(8, 5))
    # plt.hist(peak_freq_global, bins=30, color='skyblue', edgecolor='black')
    # plt.xlabel("frequency [Hz]")
    # plt.ylabel("count")
    # plt.title("Histogram of Top 5 FFT Peaks (All Particles)")
    # plt.tight_layout()
    # plt.savefig(save_hist_path, dpi=200)
    # plt.show()
    #
    # print("Saved histogram:", save_hist_path)

    # 上位5ピークの周波数値（全粒子合計）の中から大きい順に 5 つ表示
    print("\n=== Global Top 5 Peak Frequencies ===")
    peak_freq_global = np.array(peak_freq_global)
    idx_top5_global = np.argsort(peak_freq_global)[-5:]   # 大きい順TOP5
    top5_values = np.sort(peak_freq_global[idx_top5_global])

    for i, f in enumerate(top5_values[::-1], start=1):
        print(f"Top {i}: {f:.6e} Hz")

    return peak_freq_global
