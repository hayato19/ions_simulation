import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# パラメータ
# =========================
dt = 0.02
t_max = 20
t = np.arange(0, t_max, dt)

A = 1.0
omega = 2 * np.pi * 0.5
gamma = 0.15

# =========================
# 1. 単振動
# x = A cos(wt)
# =========================
x1 = A * np.cos(omega * t)

# =========================
# 2. 減衰振動
# x = A exp(-gamma t) cos(wt)
# =========================
x2 = A * np.exp(-gamma * t) * np.cos(omega * t)

# =========================
# 3. 運動量を周期的に受ける減衰振動
# m x'' + c x' + k x = 0
# 一定時間ごとに速度 v にキックを加える
# =========================
m = 1.0
k = omega**2
c = 0.4

x3 = np.zeros_like(t)
v3 = np.zeros_like(t)

kick_interval = 4.0
kick_strength = 2.0

for i in range(1, len(t)):
    # 一定間隔で運動量を与える
    if abs(t[i] % kick_interval) < dt:
        v3[i - 1] += kick_strength

    # 運動方程式: x'' = -(c/m)x' - (k/m)x
    a = -(c / m) * v3[i - 1] - (k / m) * x3[i - 1]

    v3[i] = v3[i - 1] + a * dt
    x3[i] = x3[i - 1] + v3[i] * dt

# =========================
# 描画設定
# =========================
fig, axes = plt.subplots(3, 1, figsize=(8, 6))

titles = [
    "1D Harmonic Oscillation",
    "1D Damped Oscillation",
    "Damped Oscillation with Repeated Momentum Kicks"
]

balls = []
trails = []

for ax, title in zip(axes, titles):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(title)
    ax.set_yticks([])
    ax.grid(True)

    # 中心線
    ax.axhline(0, color="gray", linewidth=1)

    # 球
    ball, = ax.plot([], [], "o", markersize=20)

    # 軌跡
    trail, = ax.plot([], [], "-", linewidth=1)

    balls.append(ball)
    trails.append(trail)

x_data = [x1, x2, x3]

# =========================
# アニメーション更新関数
# =========================
def update(frame):
    for j in range(3):
        x = x_data[j]

        balls[j].set_data([x[frame]], [0])

        start = max(0, frame - 100)
        trails[j].set_data(x[start:frame], np.zeros(frame - start))

    return balls + trails

ani = FuncAnimation(
    fig,
    update,
    frames=len(t),
    interval=20,
    blit=True
)

plt.tight_layout()
plt.show()