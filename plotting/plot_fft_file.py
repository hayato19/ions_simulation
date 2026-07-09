import numpy as np
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime

plt.rcParams["font.size"] = 14


# ===============================
# 設定
# ===============================
x_file = "../data/x_data_20260707_153851.npy"      # xMデータ
dt = 25e-9              # サンプリング間隔 [s]
save_dir = "./figs"


# ===============================
# xMを読み込む
# ===============================
ext = os.path.splitext(x_file)[1]

if ext == ".npy":
    xM = np.load(x_file)
elif ext in [".csv", ".txt"]:
    xM = np.loadtxt(x_file, delimiter=",")
else:
    raise ValueError("Unsupported file format")

# 1粒子だけの場合にも対応
if xM.ndim == 1:
    xM = xM.reshape(-1, 1)

# 時間軸を生成
N = xM.shape[0]
t = np.arange(N) * dt

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

dt_eff = t[1] - t[0]

freq = np.fft.fftfreq(Nrec, dt_eff)
mask_pos = freq > 0
f_pos = freq[mask_pos]

# -------------------------------
# 理論ピーク
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

    for f in f_lines:
        ax.axvline(
            f,
            color="black",
            linestyle="--",
            linewidth=1,
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

save_full = os.path.join(save_dir, f"fft_all_stacked_{timestamp}.png")
plt.savefig(save_full, dpi=200)
plt.show()

print("Saved:", save_full)

# ===============================
# ② 拡大図：各ピークごとに粒子を縦に並べる
# ===============================
df_zoom = 0.2e6

for i, f0 in enumerate(f_lines, start=1):

    fig, axes = plt.subplots(
        M, 1,
        figsize=(8, 2.5 * M),
        sharex=True
    )

    if M == 1:
        axes = [axes]

    zoom = (f_pos > f0 - df_zoom) & (f_pos < f0 + df_zoom)

    max_amp_zoom = np.max(amp_all[:, zoom])

    for j, ax in enumerate(axes):
        ax.plot(
            f_pos[zoom],
            amp_all[j, zoom],
            label=f"p{j}"
        )

        ax.axvline(
            f0,
            color="black",
            linestyle="--",
            linewidth=1,
            label="guide"
        )

        if max_amp_zoom > 0:
            ax.set_ylim(0, max_amp_zoom * 1.2)

        ax.set_ylabel(f"p{j}")
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frequency [Hz]")

    fig.suptitle(f"Zoom around peak {i}: f ≈ {f0/1e6:.3f} MHz")
    fig.supylabel("amplitude [arb. units]")

    plt.tight_layout()

    save_zoom = os.path.join(
        save_dir,
        f"fft_zoom_peak{i}_stacked_{timestamp}.png"
    )

    plt.savefig(save_zoom, dpi=200)
    plt.show()

    print("Saved:", save_zoom)