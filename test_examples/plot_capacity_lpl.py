# Capacity evolution analogous to Keil & Jossen 2020 Fig. 9
# Compares SEI-only vs SEI+LPL (new k_lpl=1000 run)
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_capacity_lpl.py

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
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

CASES = {
    'SEI': {
        'path':  Path('results/None'),
        'label': 'SEI only',
        'color': '#1f77b4',
        'ls':    '-',
    },
    'SEI+LPL': {
        'path':  Path('results/lpl_100_v4'),
        'label': 'SEI + LPL  (k_lpl=10, U_lpl=0.05V, i0=1e-3)',
        'color': '#d62728',
        'ls':    '--',
    },
}

MAX_CYCLES = 100   # truncate both runs to this many cycles for a fair comparison

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def load_dcc_cycles(base, max_n=None):
    """
    Return (cycle_index, capacity_Ah, t_start_s) for every DCC sub-folder.
    cycle_index is just the sub-folder integer (≈ occurrence number).
    """
    path = base / 'DCC'
    subs = sorted([int(d) for d in os.listdir(path) if (path / d).is_dir()])
    if max_n is not None:
        subs = subs[:max_n]
    cycles, caps, t0s = [], [], []
    for s in subs:
        p = path / str(s)
        try:
            tc = np.loadtxt(p / 'time.txt')
            ic = np.loadtxt(p / 'current.txt')
            cap = abs(np.trapezoid(ic, tc)) / 3600.0
            cycles.append(s)
            caps.append(cap)
            t0s.append(tc[0])
        except Exception:
            pass
    return np.array(cycles), np.array(caps), np.array(t0s)


def capacity_loss_from_film(base, t0s):
    """
    Compute capacity loss fraction ΔQ/Q0 at each cycle start t0
    from the global delta_film timeseries.
    Uses: ΔQ_film / Q0 = a_s * (ρ/M) * F * Δδ / (Q0/V_electrode)
    Simplified: just normalise delta_film increase to the first-cycle capacity.
    Returns: (delta_sei_nm, delta_lpl_nm) arrays at t0s, or None if not available.
    """
    t_g    = np.loadtxt(base / 'time.txt')
    d_film = np.loadtxt(base / 'delta_film_a.txt') * 1e9  # → nm

    d_at_t0 = np.interp(t0s, t_g, d_film)
    return d_at_t0


# ──────────────────────────────────────────────────────────────────
# Figure – normalised discharge capacity vs cycle
# ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_soh, ax_film = axes

for key, cfg in CASES.items():
    base = cfg['path']
    if not base.exists():
        print(f'  {key}: path {base} not found, skipping.')
        continue

    cyc, caps, t0s = load_dcc_cycles(base, max_n=MAX_CYCLES)
    if len(caps) == 0:
        continue

    Q0   = caps[0]
    SoH  = caps / Q0 * 100.0

    ax_soh.plot(cyc, SoH, color=cfg['color'], ls=cfg['ls'],
                lw=1.8, label=cfg['label'])

    # Film thickness at each discharge start
    d_nm = capacity_loss_from_film(base, t0s)
    ax_film.plot(cyc, d_nm, color=cfg['color'], ls=cfg['ls'],
                 lw=1.8, label=cfg['label'])

# Decoration
ax_soh.axhline(80, color='gray', ls=':', lw=1)
ax_soh.text(1, 80.5, '80% SoH', fontsize=8, color='gray')
ax_soh.set_xlabel('Cycle number')
ax_soh.set_ylabel('Normalised discharge capacity (%)')
ax_soh.set_title('Capacity fade — 1C / 10 °C\nEcker2015 cell')
ax_soh.set_ylim(50, 102)
ax_soh.legend()
ax_soh.grid(True, color='lightgray', linewidth=0.5)

ax_film.set_xlabel('Cycle number')
ax_film.set_ylabel('Film thickness δ_film (nm, spatial avg.)')
ax_film.set_title('Anode film growth vs cycle')
ax_film.legend()
ax_film.grid(True, color='lightgray', linewidth=0.5)

fig.tight_layout()
fig.savefig('results/capacity_lpl.eps', format='eps', bbox_inches='tight')
fig.savefig('results/capacity_lpl.jpg', dpi=200, bbox_inches='tight')
print('Saved results/capacity_lpl.eps / .jpg')

plt.show()
