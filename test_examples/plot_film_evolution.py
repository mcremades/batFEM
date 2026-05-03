# Film thickness evolution over the complete experiment
# Compares SEI-only vs SEI+LPL and decomposes the latter into components.
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_film_evolution.py

import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ──────────────────────────────────────────────────────────────────
# Material parameters (NE_Ecker2015)
# ──────────────────────────────────────────────────────────────────

M_SEI   = 0.162    ; RHO_SEI = 1690.0
M_LPL   = 6.94e-3  ; RHO_LPL =  534.0
EPS_S   = 0.372    ; R_P     = 1.37e-5
A_S     = 3.0 * EPS_S / R_P   # m²/m³

def to_nm(c, M, rho):
    return c * M / (rho * A_S) * 1e9   # mol/m³ → nm

# ──────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────

sei_path = Path('results/None')
lpl_path = Path('results/lpl_100_v4_cu')

t_sei   = np.loadtxt(sei_path / 'time.txt') / 3600.0          # h
d_sei   = np.loadtxt(sei_path / 'delta_film_a.txt') * 1e9     # nm

t_lpl   = np.loadtxt(lpl_path / 'time.txt') / 3600.0
d_total = np.loadtxt(lpl_path / 'delta_film_a.txt') * 1e9
c_sei   = np.loadtxt(lpl_path / 'c_sei_a.txt')
c_lpl   = np.loadtxt(lpl_path / 'c_lpl_a.txt')

d_sei_comp = to_nm(c_sei, M_SEI, RHO_SEI)   # SEI contribution
d_lpl_comp = to_nm(c_lpl, M_LPL, RHO_LPL)   # LPL contribution

# ──────────────────────────────────────────────────────────────────
# Figure — two panels
# ──────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: total film comparison ────────────────────────────────────
ax1.plot(t_sei, d_sei,   color='#1f77b4', lw=1.6,
         label=f'SEI only  (285 cycles, {t_sei[-1]:.0f} h)')
ax1.plot(t_lpl, d_total, color='#d62728', lw=1.6, ls='--',
         label=f'SEI+LPL total  (100 cycles, {t_lpl[-1]:.0f} h)')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Film thickness δ_film (nm)')
ax1.set_title('Total anode film — SEI only vs SEI+LPL')
ax1.legend()
ax1.grid(True, color='lightgray', linewidth=0.5)

# ── Right: SEI+LPL decomposed ──────────────────────────────────────
ax2.plot(t_lpl, d_total,    color='#d62728', lw=1.8,
         label=f'Total  ({d_total[-1]:.0f} nm)')
ax2.plot(t_lpl, d_sei_comp, color='#ff7f0e', lw=1.4, ls='--',
         label=f'SEI component  ({d_sei_comp[-1]:.0f} nm)')
ax2.plot(t_lpl, d_lpl_comp, color='#2ca02c', lw=1.4, ls=':',
         label=f'LPL component  ({d_lpl_comp[-1]:.0f} nm)')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Film thickness (nm)')
ax2.set_title('SEI+LPL run — component decomposition')
ax2.legend()
ax2.grid(True, color='lightgray', linewidth=0.5)

fig.suptitle('Anode film growth — Ecker2015 cell, 1C / 10 °C', fontsize=11)
fig.tight_layout(pad=1.2)
fig.savefig('results/film_evolution.eps', format='eps', bbox_inches='tight')
fig.savefig('results/film_evolution.jpg', dpi=200, bbox_inches='tight')
print('Saved results/film_evolution.eps / .jpg')

plt.show()
