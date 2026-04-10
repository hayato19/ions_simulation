import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
plt.rcParams["font.size"] = 15

def plot_x_range(t, xM, t_start, t_end, particle_index, save_dir="./figs"):
    """
    特定時間範囲、特定粒子の x(t) をプロット
    """

    M = xM.shape[1]
    if not (0 <= particle_index < M):
        print("粒子番号が範囲外です")
        return

    idx = np.where((t >= t_start) & (t <= t_end))[0]
    if len(idx) == 0:
        print("指定した時間範囲にデータがありません")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"x_p{particle_index}_{t_start:.1e}_{t_end:.1e}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)

    plt.figure(figsize=(16,5))
    plt.plot(t[idx], xM[idx, particle_index],
             label=f"particle {particle_index}", lw=1.5)
    plt.grid(True)
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("x(t)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)
