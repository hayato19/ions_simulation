import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_fft_all_particles(t, xM, dt, save_dir="./figs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"fft_all_particles_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    Nrec, M = xM.shape

    # サンプリング間隔 (正しい FFT 用)
    dt_eff = t[1] - t[0]
    freq = np.fft.fftfreq(Nrec, dt_eff)

    plt.figure(figsize=(10, 6))

    # 周波数の正の成分のみ使用（0 Hz 付近除去のため）
    mask_pos = freq > 0

    max_amp = 0.0

    for j in range(M):
        X = np.fft.fft(xM[:, j])
        amp = np.abs(X) / Nrec

        # f ≅ 0 を除去した領域で最大値を取る
        max_amp = max(max_amp, np.max(amp[mask_pos]))

        # f > 0 成分のみを描画
        plt.plot(freq[mask_pos], amp[mask_pos], label=f"p{j}")

    # 周波数範囲：小さなピーク探索のため 0〜5 MHz に絞る
    plt.xlim(0, 6e6)

    # Y 軸を見えやすいように自動スケール
    if max_amp > 0:
        plt.ylim(0, max_amp * 1.2)

    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    print("Saved FFT figure:", save_path)
