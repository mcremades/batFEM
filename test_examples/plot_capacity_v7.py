# Capacity fade for lpl_v7 parametric study
# 3 C-rates × 3 temperatures, SEI+LPL model (i0_sei=1e-13, i0_lpl=0.01, U_lpl=0V)
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_capacity_v7.py

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

# ──────────────────────────────────────────────────────────────────
# Run definitions
# ──────────────────────────────────────────────────────────────────

C_RATES  = [0.5, 1.0, 2.0]
TEMPS    = [10,  25,  40]

C_COLORS = {0.5: '#1f77b4', 1.0: '#d62728', 2.0: '#2ca02c'}
T_STYLES = {10: '-', 25: '--', 40: ':'}
T_MARKERS= {10: 'o', 25: 's', 40: '^'}
C_LABELS = {0.5: '0.5C', 1.0: '1.0C', 2.0: '2.0C'}
T_LABELS = {10: '10°C', 25: '25°C', 40: '40°C'}

def run_name(C, T):
    return f'lpl_v7_{int(C*10):02d}C_{T:02d}deg'

def load_soh(base):
    path = base / 'DCC'
    if not path.exists():
        return None, None
    subs = sorted([int(d) for d in os.listdir(path) if (path / d).is_dir()])
    cycles, caps = [], []
    for s in subs:
        p = path / str(s)
        try:
            tc = np.loadtxt(p / 'time.txt')
            ic = np.loadtxt(p / 'current.txt')
            cap = abs(np.trapezoid(ic, tc)) / 3600.0
            cycles.append(s)
            caps.append(cap)
        except Exception:
            pass
    if not caps:
        return None, None
    caps = np.array(caps)
    return np.array(cycles), caps / caps[0] * 100.0

# ──────────────────────────────────────────────────────────────────
# Figure layout: 1×3 subplots, one per C-rate
# ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, C in zip(axes, C_RATES):
    for T in TEMPS:
        name = run_name(C, T)
        base = Path(f'results/{name}')
        cyc, soh = load_soh(base)
        if cyc is None:
            ax.text(0.5, 0.5, f'{T_LABELS[T]}\n(no data)', ha='center',
                    va='center', transform=ax.transAxes, fontsize=8, color='gray')
            continue
        blown = len(cyc) < 249
        lbl = f'{T_LABELS[T]}'
        if blown:
            lbl += f'  * cycle {cyc[-1]}'
        ax.plot(cyc, soh, color=C_COLORS[C], ls=T_STYLES[T], lw=1.6,
                label=lbl)
        if blown:
            ax.plot(cyc[-1], soh[-1], 'x', color=C_COLORS[C], ms=8, mew=2)

    ax.axhline(80, color='gray', ls=':', lw=0.8)
    ax.text(2, 80.8, '80%', fontsize=7, color='gray')
    ax.set_title(f'{C_LABELS[C]} charging', pad=5)
    ax.set_xlabel('Cycle number')
    ax.set_ylim(50, 102)
    ax.set_xlim(left=0)
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.legend(loc='lower left')

axes[0].set_ylabel('Normalised discharge capacity (%)')

fig.suptitle(
    'Capacity fade — SEI+LPL model  |  Ecker2015 cell\n'
    r'$i_{0,\mathrm{SEI}}=10^{-13}$ mol/m²/s,  '
    r'$i_{0,\mathrm{LPL}}=0.01$ A/m²,  $U_\mathrm{LPL}=0$ V,  $k_\mathrm{LPL}=10$',
    fontsize=10
)
fig.tight_layout(pad=1.2)
fig.savefig('results/capacity_v7.eps', format='eps', bbox_inches='tight')
fig.savefig('results/capacity_v7.jpg', dpi=200, bbox_inches='tight')
print('Saved results/capacity_v7.eps / .jpg')
plt.show()
