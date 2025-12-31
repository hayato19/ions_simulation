# simulation/params.py

import math
from decimal import Decimal, getcontext

getcontext().prec = 50  # 計算精度

# --- params ---
dt = 1e-9        # 時間刻み（0.05ns）
N  = 500 * 3 * 10*2*3    # 記録数
w  = 50 // 3         # 記録間隔（w ステップに 1 回記録）
                     # ステップ数　= N * w

alpha = 2.3e-28       # クーロン反発の係数
eps   = 1e-7          # 発散防止のための微小数

# ---- heating parameters ----
ht  = 1000      # heating 判定間隔
ips = 0         # scattering param

# ---- operation parameters ----
cool_time = 50e-6
spec_time = 1e-6
cycle_time = cool_time + spec_time

# --- physical constants ---
hbar = Decimal("1.054e-34")     # ディラック定数
NA   = Decimal("6.02214076e23") # アボガドロ定数
