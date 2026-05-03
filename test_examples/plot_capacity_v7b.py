# Capacity fade for lpl_v7 parametric study — subplots per temperature
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_capacity_v7b.py

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

C_RATES  = [0.5, 1.0, 2.0]
TEMPS    = [10,  25,  40]

T_COLORS = {10: '#1f77b4', 25: '#d62728', 40: '#2ca02c'}
C_STYLES = {0.5: '-', 1.0: '--', 2.0: ':'}
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
            caps.append(abs(np.trapezoid(ic, tc)) / 3600.0)
            cycles.append(s)
        except Exception:
            pass
    if not caps:
        return None, None
    caps = np.array(caps)
    return np.array(cycles), caps / caps[0] * 100.0

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, T in zip(axes, TEMPS):
    color = T_COLORS[T]
    for C in C_RATES:
        name = run_name(C, T)
        cyc, soh = load_soh(Path(f'results/{name}'))
        if cyc is None:
            continue
        blown = len(cyc) < 249
        lbl = C_LABELS[C]
        if blown:
            lbl += f'  * cycle {cyc[-1]}'
        ax.plot(cyc, soh, color=color, ls=C_STYLES[C], lw=1.6, label=lbl)
        if blown:
            ax.plot(cyc[-1], soh[-1], 'x', color=color, ms=8, mew=2)

    ax.axhline(80, color='gray', ls=':', lw=0.8)
    ax.text(2, 80.8, '80%', fontsize=7, color='gray')
    ax.set_title(f'{T_LABELS[T]}', pad=5)
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
fig.savefig('results/capacity_v7b.eps', format='eps', bbox_inches='tight')
fig.savefig('results/capacity_v7b.jpg', dpi=200, bbox_inches='tight')
print('Saved results/capacity_v7b.eps / .jpg')
plt.show()
