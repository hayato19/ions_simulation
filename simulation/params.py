# simulation/params.py

import math
from decimal import Decimal, getcontext

getcontext().prec = 50  # 計算精度

# --- params ---
dt = 1e-10 / 2        # 時間刻み（0.05ns）
N  = 5000             # 記録数
w  = 200              # 記録間隔（w ステップに 1 回記録）
                      # ステップ数　= N * w

alpha = 2.3e-28       # クーロン反発の係数
eps   = 1e-7          # 発散防止のための微小数

# ---- heating parameters ----
ht  = 1000      # heating 判定間隔
ips = 0         # scattering param

# --- physical constants ---
hbar = Decimal("1.054e-34")     # ディラック定数
NA   = Decimal("6.02214076e23") # アボガドロ定数
