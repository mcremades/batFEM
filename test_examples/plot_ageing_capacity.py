# Date: 29/04/2026
# Auth: Manuel Cremades, manuel.cremades@usc.es
#
# Generates thesis figures for section 2.4.2 "Simulation of battery aging by
# prolonged cycling":
#   Figure 1 – SoH vs cycle (1C DCC) and 0.2C checkup capacity
#   Figure 2 – Film thickness and porosity vs cycle
#   Figure 3 – Discharge voltage profiles at selected cycles
#   Figure 4 – SEI vs LPL contributions (when both are available)

import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

CASES = {
    'SEI': {
        'path': Path('results/None'),
        'label': 'SEI only',
        'color': '#1f77b4',
        'ls': '-',
    },
    # Uncomment when SEI+LPL run is available:
    # 'SEI+LPL': {
    #     'path': Path('results/sei_lpl'),
    #     'label': 'SEI + LPL',
    #     'color': '#d62728',
    #     'ls': '--',
    # },
}

CYCLES_FOR_VOLTAGE = [1, 50, 100, 150, 200, 250]

# ──────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────

def load_dcc_cycles(base):
    """Return arrays (cycle_numbers, capacities_Ah, t_start_s)."""
    path = base / 'DCC'
    subs = sorted([int(d) for d in os.listdir(path) if (path / d).is_dir()])
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


def load_checkup_cycles(base, state='CU_DCC_2'):
    """Return (cycle_numbers_after_which_checkup_ran, capacities_Ah)."""
    path = base / state
    if not path.exists():
        return np.array([]), np.array([])
    subs = sorted([int(d) for d in os.listdir(path) if (path / d).is_dir()])
    cycles, caps = [], []
    for k, s in enumerate(subs):
        p = path / str(s)
        try:
            tc = np.loadtxt(p / 'time.txt')
            ic = np.loadtxt(p / 'current.txt')
            cap = abs(np.trapezoid(ic, tc)) / 3600.0
            cycles.append((k + 1) * 25)
            caps.append(cap)
        except Exception:
            pass
    return np.array(cycles), np.array(caps)


def load_ageing_metrics(base):
    """Interpolate film thickness and porosity at DCC cycle start times."""
    t_g = np.loadtxt(base / 'time.txt')
    delta = np.loadtxt(base / 'delta_film_a.txt')
    eps = np.loadtxt(base / 'eps_e_a.txt')

    # c_sei and c_lpl (may not exist for SEI-only runs)
    c_sei_path = base / 'c_sei_a.txt'
    c_lpl_path = base / 'c_lpl_a.txt'
    c_sei = np.loadtxt(c_sei_path) if c_sei_path.exists() else None
    c_lpl = np.loadtxt(c_lpl_path) if c_lpl_path.exists() else None

    return t_g, delta, eps, c_sei, c_lpl


def load_voltage_profile(base, cycle):
    """Load (time_s, voltage_V) for one DCC cycle."""
    p = base / 'DCC' / str(cycle)
    tc = np.loadtxt(p / 'time.txt')
    vc = np.loadtxt(p / 'voltage.txt')
    ic = np.loadtxt(p / 'current.txt')
    cap = abs(np.trapezoid(ic, tc)) / 3600.0
    # Normalise time to Ah discharged
    ah = np.abs(np.array([np.trapezoid(ic[:i+1], tc[:i+1]) for i in range(len(tc))]) / 3600.0)
    return ah, vc, cap


# ──────────────────────────────────────────────────────────────────
# Figure 1 – SoH vs cycle number
# ──────────────────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(6, 4))

for key, cfg in CASES.items():
    base = cfg['path']
    cyc, caps, _ = load_dcc_cycles(base)
    Q0 = caps[0]
    SoH = caps / Q0

    ax1.plot(cyc, SoH * 100, color=cfg['color'], ls=cfg['ls'],
             label=f"{cfg['label']} — 1C discharge", lw=1.5)

    # Checkup capacity
    cu_cyc, cu_caps = load_checkup_cycles(base)
    if len(cu_caps) > 0:
        ax1.plot(cu_cyc, cu_caps / cu_caps[0] * 100,
                 'o', color=cfg['color'], ms=5,
                 label=f"{cfg['label']} — 0.2C checkup")

ax1.axhline(80, color='gray', ls=':', lw=1, label='80% SoH threshold')
ax1.set_xlabel('Cycle number')
ax1.set_ylabel('State of Health (%)')
ax1.set_title('Ecker2015 (NCO / graphite, 10 °C, 1C cycling)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(60, 102)

fig1.tight_layout()
fig1.savefig('results/ageing_SoH.pdf', bbox_inches='tight')
print('Saved results/ageing_SoH.pdf')

# ──────────────────────────────────────────────────────────────────
# Figure 2 – Film thickness and porosity vs cycle
# ──────────────────────────────────────────────────────────────────

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

for key, cfg in CASES.items():
    base = cfg['path']
    cyc, caps, t0s = load_dcc_cycles(base)
    t_g, delta, eps, c_sei, c_lpl = load_ageing_metrics(base)

    delta_at_cyc = np.interp(t0s, t_g, delta) * 1e9  # convert to nm
    eps_at_cyc   = np.interp(t0s, t_g, eps)

    ax2a.plot(cyc, delta_at_cyc, color=cfg['color'], ls=cfg['ls'],
              label=cfg['label'], lw=1.5)
    ax2b.plot(cyc, eps_at_cyc,   color=cfg['color'], ls=cfg['ls'],
              label=cfg['label'], lw=1.5)

ax2a.set_ylabel('Film thickness (nm)')
ax2a.set_title('Negative electrode degradation layer (spatial average)')
ax2a.legend()
ax2a.grid(True, alpha=0.3)

ax2b.set_xlabel('Cycle number')
ax2b.set_ylabel('Electrolyte porosity (-)')
ax2b.legend()
ax2b.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig('results/ageing_film_porosity.pdf', bbox_inches='tight')
print('Saved results/ageing_film_porosity.pdf')

# ──────────────────────────────────────────────────────────────────
# Figure 3 – Voltage profiles at selected cycles
# ──────────────────────────────────────────────────────────────────

for key, cfg in CASES.items():
    base = cfg['path']
    cyc_all, caps_all, _ = load_dcc_cycles(base)
    Q0 = caps_all[0]

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.plasma
    available = sorted([int(d) for d in os.listdir(base / 'DCC')
                        if (base / 'DCC' / d).is_dir()])

    for i, cyc in enumerate(CYCLES_FOR_VOLTAGE):
        c_plot = cmap(i / max(len(CYCLES_FOR_VOLTAGE) - 1, 1))
        # Find nearest available cycle
        near = min(available, key=lambda x: abs(x - cyc))
        ah, vc, cap = load_voltage_profile(base, near)
        SoH_val = cap / Q0 * 100
        ax3.plot(ah, vc, color=c_plot, lw=1.5,
                 label=f'Cycle {near}  (SoH = {SoH_val:.1f}%)')

    ax3.set_xlabel('Discharged capacity (Ah)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title(f'1C discharge profiles — {cfg["label"]}')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    out = f'results/ageing_voltage_profiles_{key}.pdf'
    fig3.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

# ──────────────────────────────────────────────────────────────────
# Figure 4 – SEI vs LPL concentration (SEI+LPL case only)
# ──────────────────────────────────────────────────────────────────

for key, cfg in CASES.items():
    base = cfg['path']
    cyc, caps, t0s = load_dcc_cycles(base)
    t_g, delta, eps, c_sei, c_lpl = load_ageing_metrics(base)

    if c_sei is None:
        continue

    c_sei_at = np.interp(t0s, t_g, c_sei)
    delta_sei = c_sei_at / (c_sei_at[0] if c_sei_at[0] > 0 else 1)

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(cyc, c_sei_at, color='#1f77b4', lw=1.5, label='SEI concentration')

    if c_lpl is not None:
        c_lpl_at = np.interp(t0s, t_g, c_lpl)
        ax4.plot(cyc, c_lpl_at, color='#d62728', lw=1.5, label='LPL concentration')

    ax4.set_xlabel('Cycle number')
    ax4.set_ylabel('Molar concentration (mol/m³)')
    ax4.set_title(f'SEI and LPL concentration — {cfg["label"]}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    out = f'results/ageing_sei_lpl_{key}.pdf'
    fig4.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

plt.show()
