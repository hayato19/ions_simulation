import numpy as np
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime

plt.rcParams["font.size"] = 14


def plot_fft_all_particles(t, xM, dt=None, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    # ===============================
    # xM の後半のみを使用
    # ===============================
    L = xM.shape[0]
    start = L // 2

    t = t[start:]
    xM = xM[start:, :]

    Nrec, M = xM.shape

    # dt は引数ではなく t から計算する
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
    # FFT計算
    # ===============================
    amp_all = []

    for j in range(M):
        # 平均値を引いて DC 成分を除去
        x = xM[:, j] - np.mean(xM[:, j])

        X = np.fft.fft(x)
        amp = np.abs(X) / Nrec
        amp_pos = amp[mask_pos]

        amp_all.append(amp_pos)

    amp_all = np.array(amp_all)  # shape: (M, Nfreq)

    # ===============================
    # ① 全体FFT：粒子ごとに縦に並べる
    # ===============================
    fig, axes = plt.subplots(
        M, 1,
        figsize=(10, 2.5 * M),
        sharex=True
    )

    if M == 1:
        axes = [axes]

    max_amp = np.max(amp_all)

    for j, ax in enumerate(axes):
        ax.plot(f_pos, amp_all[j], label=f"p{j}")

        for f_line in f_lines:
            ax.axvline(
                f_line,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.5
            )

        ax.set_xlim(0, 4.5e6)

        if max_amp > 0:
            ax.set_ylim(0, max_amp * 1.2)

        ax.set_ylabel(f"p{j}")
        ax.set_yticks([])
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frequency [Hz]")

    fig.supylabel("amplitude [arb. units]")
    plt.tight_layout()

    save_full = os.path.join(
        save_dir,
        f"fft_all_stacked_{timestamp}.png"
    )
    plt.savefig(save_full, dpi=200)
    plt.show()

    print("Saved:", save_full)

    # ===============================
    # ② 拡大図：各ピークごとに粒子を縦に並べる
    # ===============================
    df_zoom = 0.2e6   # ±0.2 MHz

    for i, f0 in enumerate(f_lines, start=1):
        fig, axes = plt.subplots(
            M, 1,
            figsize=(8, 2.5 * M),
            sharex=True
        )

        if M == 1:
            axes = [axes]

        zoom_mask = (f_pos > f0 - df_zoom) & (f_pos < f0 + df_zoom)

        max_amp_zoom = np.max(amp_all[:, zoom_mask])

        for j, ax in enumerate(axes):
            ax.plot(
                f_pos[zoom_mask],
                amp_all[j, zoom_mask],
                label=f"p{j}"
            )

            ax.axvline(
                f0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="guide"
            )

            if max_amp_zoom > 0:
                ax.set_ylim(0, max_amp_zoom * 1.2)

            ax.set_ylabel(f"p{j}")
            ax.grid(True)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("frequency [Hz]")

        fig.suptitle(
            f"Zoom around peak {i}: f ≈ {f0/1e6:.3f} MHz"
        )
        fig.supylabel("amplitude [arb. units]")

        plt.tight_layout()

        save_zoom = os.path.join(
            save_dir,
            f"fft_zoom_peak{i}_stacked_{timestamp}.png"
        )
        plt.savefig(save_zoom, dpi=200)
        plt.show()

        print(f"Saved zoom {i}:", save_zoom)

    return f_lines
