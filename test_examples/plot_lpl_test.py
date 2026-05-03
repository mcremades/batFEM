# Quick diagnostic: check LPL formation/stripping within cycles
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_lpl_test.py [results/test_lpl]

import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/test_lpl')
LABEL  = BASE.name   # used in title and output filename
OUTDIR = BASE.parent

# Electrode parameters (NE_Ecker2015)
F     = 96485.0
eps_s = 0.372
R_p   = 1.37e-5
a_s   = 3.0 * eps_s / R_p   # specific interfacial area (m²/m³)
U_sei = 0.4                  # SEI OCP vs Li/Li+ (V)
U_lpl = 0.0                  # LPL OCP vs Li/Li+ (V)

# Anode OCP function (Ecker2015 graphite)
def U_anode(x):
    return (  0.716502 * np.exp(-369.028 * x)
            + 0.12193  * np.exp(-35.6478 * (x - 0.0530947))
            - 0.0189193 * np.tanh(21.1967 * (x - 0.196176))
            - 0.0169644 * np.tanh(27.1365 * (x - 0.312832))
            - 0.0199313 * np.tanh(28.5697 * (x - 0.614221))
            - 0.931153  * np.exp(36.328  * (x - 1.10743))
            + 0.140031)

t      = np.loadtxt(BASE / 'time.txt')
v      = np.loadtxt(BASE / 'voltage.txt')
i      = np.loadtxt(BASE / 'current.txt')
d      = np.loadtxt(BASE / 'delta_film_a.txt')
c_sei  = np.loadtxt(BASE / 'c_sei_a.txt')
c_lpl  = np.loadtxt(BASE / 'c_lpl_a.txt')
xs_sur = np.loadtxt(BASE / 'xs_sur_a.txt')

t_h = t / 3600.0

# SEI and LPL current densities (mA/m² of particle surface)
# Positive = formation/plating; negative = dissolution/stripping
scale = F * 1e3 / a_s
j_sei = -np.gradient(c_sei, t) * scale
j_lpl = -np.gradient(c_lpl, t) * scale

# Equilibrium ageing overpotentials at the anode surface
# eta_X = U_anode(xs_sur) - U_X  (< 0 means reaction X is thermodynamically favoured)
Ua      = U_anode(xs_sur)
eta_sei = Ua - U_sei   # negative almost always → SEI always favoured
eta_lpl = Ua - U_lpl   # = Ua; negative only when anode is below 0 V vs Li/Li+

fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

axes[0].plot(t_h, v, 'k', lw=1)
axes[0].set_ylabel('Voltage (V)')
axes[0].set_ylim(2.5, 4.3)

axes[1].plot(t_h, i, 'b', lw=1)
axes[1].set_ylabel('Current (A)')
axes[1].axhline(0, color='k', lw=0.5, ls='--')

axes[2].plot(t_h, d * 1e9, 'g', lw=1)
axes[2].set_ylabel('δ_film (nm)')

axes[3].plot(t_h, j_sei, color='#d62728', lw=1, label='SEI rate')
axes[3].plot(t_h, j_lpl, color='steelblue', lw=1, label='LPL rate')
axes[3].axhline(0, color='k', lw=0.5, ls='--')
axes[3].set_ylabel('j (mA/m²)')
axes[3].legend(fontsize=8)

axes[4].plot(t_h, eta_sei * 1e3, color='#d62728', lw=1, label=rf'$\eta_{{SEI}} = U_a - {U_sei}\,V$')
axes[4].plot(t_h, eta_lpl * 1e3, color='steelblue', lw=1, label=rf'$\eta_{{LPL}} = U_a - {U_lpl}\,V$')
axes[4].axhline(0, color='k', lw=0.8, ls='--')
axes[4].set_ylabel('η (mV)')
axes[4].set_xlabel('Time (h)')
axes[4].legend(fontsize=8)

for ax in axes:
    ax.grid(True, color='lightgray', linewidth=0.5)

fig.suptitle(f'LPL test — {LABEL}  |  k_lpl=10, i0_lpl=0.1, U_lpl=0.0 V, i0_sei=1e-12, SEI+LPL model', fontsize=11)
fig.tight_layout()
outpath = OUTDIR / f'lpl_test_{LABEL}.jpg'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
