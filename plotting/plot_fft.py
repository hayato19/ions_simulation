import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime
plt.rcParams["font.size"] = 14

def plot_fft_all_particles(t, xM, dt, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    Nrec, M = xM.shape

    dt_eff = t[1] - t[0]
    freq = np.fft.fftfreq(Nrec, dt_eff)
    mask_pos = freq > 0
    f_pos = freq[mask_pos]

    # -------------------------------
    # 理論的ピーク位置 [Hz]
    # -------------------------------
    f_lines = np.array([
        1,
        math.sqrt(3),
        math.sqrt(5.818),
        math.sqrt(9.332),
        math.sqrt(13.47)
    ]) * 1e6

    # ===============================
    # ① 全体 FFT 図
    # ===============================
    plt.figure(figsize=(10, 6))
    max_amp = 0.0

    amp_all = []

    for j in range(M):
        X = np.fft.fft(xM[:, j])
        amp = np.abs(X) / Nrec
        amp_pos = amp[mask_pos]
        amp_all.append(amp_pos)

        max_amp = max(max_amp, np.max(amp_pos))
        plt.plot(f_pos, amp_pos, label=f"p{j}")

    for f_line in f_lines:
        plt.axvline(
            f_line,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5
        )

    # ★ 描画範囲を 0〜4.5 MHz に変更
    plt.xlim(0, 4.5e6)

    if max_amp > 0:
        plt.ylim(0, max_amp * 1.2)

    plt.xlabel("frequency [Hz]")

    # ★ 縦軸ラベルを任意単位に変更
    plt.ylabel("amplitude [arb. units]")

    # ★ 縦軸の数値を消す
    plt.yticks([])

    plt.legend()
    plt.tight_layout()

    save_full = os.path.join(save_dir, f"fft_all_{timestamp}.png")
    plt.savefig(save_full, dpi=200)
    plt.show()

    print("Saved:", save_full)


    # ===============================
    # ② ピーク周辺の拡大図
    # ===============================
    # df_zoom = 0.2e6   # ±0.2 MHz
    #
    # amp_all = np.array(amp_all)  # (M, Nfreq)
    #
    # for i, f0 in enumerate(f_lines, start=1):
    #     plt.figure(figsize=(8, 5))
    #
    #     zoom_mask = (f_pos > f0 - df_zoom) & (f_pos < f0 + df_zoom)
    #
    #     for j in range(M):
    #         plt.plot(
    #             f_pos[zoom_mask],
    #             amp_all[j, zoom_mask],
    #             label=f"p{j}"
    #         )
    #
    #     plt.axvline(
    #         f0,
    #         color="black",
    #         linestyle="--",
    #         linewidth=1.0,
    #         label="guide"
    #     )
    #
    #     plt.xlabel("frequency [Hz]")
    #     plt.ylabel("amplitude [m]")
    #     plt.title(f"Zoom around peak {i}: f ≈ {f0/1e6:.3f} MHz")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #
    #     save_zoom = os.path.join(
    #         save_dir,
    #         f"fft_zoom_peak{i}_{timestamp}.png"
    #     )
    #     plt.savefig(save_zoom, dpi=200)
    #     plt.show()
    #
    #     print(f"Saved zoom {i}:", save_zoom)
