# Checkpoint/restart test — joint capacity fade and per-cycle voltage curves.
# Verifies continuity at the cycle-50 restart boundary.
#
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_ckpt_test.py

import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

BASE     = Path('results/lpl_ckpt_test')
RESTART_CYCLE = 50   # last cycle of first run

# ──────────────────────────────────────────────────────────────────
# Load per-cycle capacity (DCC integration)
# ──────────────────────────────────────────────────────────────────

dcc_path = BASE / 'DCC'
subs = sorted([int(d) for d in os.listdir(dcc_path) if (dcc_path / d).is_dir()])

cycles, caps = [], []
for k in subs:
    p = dcc_path / str(k)
    try:
        tc = np.loadtxt(p / 'time.txt')
        ic = np.loadtxt(p / 'current.txt')
        cap = abs(np.trapezoid(ic, tc)) / 3600.0
        cycles.append(k)
        caps.append(cap)
    except Exception:
        pass

cycles = np.array(cycles)
caps   = np.array(caps)
soh    = caps / caps[0] * 100.0

mask1 = cycles <= RESTART_CYCLE
mask2 = cycles >  RESTART_CYCLE

# ──────────────────────────────────────────────────────────────────
# Load per-cycle discharge voltage curves near the restart boundary
# ──────────────────────────────────────────────────────────────────

WINDOW = [48, 49, 50, 51, 52, 53]
cmap   = plt.cm.RdYlGn
n_w    = len(WINDOW)

cycle_curves = {}
for k in WINDOW:
    p = dcc_path / str(k)
    try:
        tc = np.loadtxt(p / 'time.txt')
        vc = np.loadtxt(p / 'voltage.txt')
        ic = np.loadtxt(p / 'current.txt')
        ah = np.abs(np.cumsum(ic * np.gradient(tc))) / 3600.0
        cycle_curves[k] = (ah, vc)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────

fig, (ax_cap, ax_vlt) = plt.subplots(1, 2, figsize=(12, 5))

# — Left: capacity fade —————————————————————————————————————————
ax_cap.plot(cycles[mask1], soh[mask1], 'o-', color='#1f77b4', lw=1.6,
            ms=3, label=f'First run  (cycles 1–{RESTART_CYCLE})')
ax_cap.plot(cycles[mask2], soh[mask2], 's--', color='#d62728', lw=1.6,
            ms=3, label=f'Restart    (cycles {RESTART_CYCLE+1}–{cycles[-1]})')
ax_cap.axvline(RESTART_CYCLE + 0.5, color='#555', ls=':', lw=1.0)
ax_cap.text(RESTART_CYCLE + 1.5, soh.min() + 0.3, 'restart', fontsize=8, color='#555')
ax_cap.axhline(80, color='gray', ls=':', lw=0.8)
ax_cap.text(2, 80.4, '80%', fontsize=7, color='gray')
ax_cap.set_xlabel('Cycle number')
ax_cap.set_ylabel('Normalised discharge capacity (%)')
ax_cap.set_title('Capacity fade — checkpoint test\n1C · 25°C · SEI+LPL model')
ax_cap.set_xlim(left=0)
ax_cap.set_ylim(soh.min() - 0.5, 100.5)
ax_cap.legend(loc='lower left')
ax_cap.grid(True, color='lightgray', linewidth=0.5)

# — Right: discharge voltage curves near restart ————————————————
for i, k in enumerate(WINDOW):
    if k not in cycle_curves:
        continue
    ah, vc = cycle_curves[k]
    color = cmap(i / (n_w - 1))
    ls    = '-' if k <= RESTART_CYCLE else '--'
    lbl   = f'Cycle {k}' + ('  [restart →]' if k == RESTART_CYCLE else
                             ('  [← restart]' if k == RESTART_CYCLE + 1 else ''))
    ax_vlt.plot(ah, vc, color=color, ls=ls, lw=1.5, label=lbl)

ax_vlt.set_xlabel('Discharge capacity (Ah)')
ax_vlt.set_ylabel('Voltage (V)')
ax_vlt.set_title('Discharge voltage — cycles near restart boundary')
ax_vlt.set_ylim(2.7, 4.2)
ax_vlt.legend(loc='lower left', fontsize=8)
ax_vlt.grid(True, color='lightgray', linewidth=0.5)

fig.tight_layout(pad=1.2)
fig.savefig('results/ckpt_test.jpg', dpi=200, bbox_inches='tight')
print('Saved results/ckpt_test.jpg')
plt.show()
