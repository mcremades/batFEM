# Date: 29/04/2026
# Auth: Manuel Cremades, manuel.cremades@usc.es
#
# Figures analogous to Keil & Jossen 2020 (JES doi:10.1149/1945-7111/aba44f)
# Figs 8–10, comparing SEI-only and SEI+LPL ageing runs of the batFEM P2D model
# (Ecker2015 cell, 1C cycling at 10 °C).
#
# Output (EPS): results/comparison_fig8_voltages.eps
#               results/comparison_fig9_capacity.eps
#               results/comparison_fig10_degradation.eps
#               results/comparison_fig10b_transient.eps
#
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_param_comparison.py

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
# Physical and material constants
# ──────────────────────────────────────────────────────────────────

F_CONST  = 96485.0   # C/mol
R_CONST  = 8.314     # J/(mol K)

# Negative electrode (NE_Ecker2015.json)
EPS_S_A  = 0.372
R_P_A    = 1.37e-5   # m  (particle radius)
A_S_A    = 3.0 * EPS_S_A / R_P_A  # specific area, ~81460 m⁻¹
C_SMAX_A = 31920.0   # mol/m³
XS1_A    = 0.848
XS0_A    = 0.003
DXS_A    = XS1_A - XS0_A
L_A      = 7.3e-5    # m  (electrode thickness per layer, cell_Ecker2015.json)

# SEI (Li₂CO₃-type)
M_SEI   = 0.162      # kg/mol
RHO_SEI = 1690.0     # kg/m³

# LPL (metallic Li)
M_LPL   = 6.94e-3    # kg/mol
RHO_LPL = 534.0      # kg/m³

# Normalised capacity loss from SEI lithium consumption:
#   ΔQ_sei / Q0 = c_sei / (eps_s * c_smax * Δxs)
# with c_sei = a_s * (rho_sei / M_sei) * delta_film   [SEI-only exact]
SEI_SCALE = A_S_A * (RHO_SEI / M_SEI)   # mol/m⁴  (converts δ → c_sei)
CAP_DENOM = EPS_S_A * C_SMAX_A * DXS_A  # ~10028 mol/m³

# ──────────────────────────────────────────────────────────────────
# Cases
# ──────────────────────────────────────────────────────────────────

CASES = {
    'SEI': {
        'path':  Path('results/None'),
        'label': 'SEI only',
        'color': '#1f77b4',
        'ls':    '-',
    },
    'SEI+LPL': {
        'path':  Path('results/sei_lpl'),
        'label': 'SEI + LPL',
        'color': '#d62728',
        'ls':    '--',
    },
    # Uncomment once paper-parameter run completes:
    # 'SEI (paper params)': {
    #     'path':  Path('results/sei_paper'),
    #     'label': 'SEI only (Keil & Jossen params)',
    #     'color': '#2ca02c',
    #     'ls':    '-.',
    # },
}

CYCLES_FOR_VOLTAGE = [1, 50, 100, 150, 200, 250]
CYCLES_TRANSIENT   = [5, 200]   # cycles shown in Fig 10b

# ──────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────

def _dcc_subdirs(base):
    p = base / 'DCC'
    return sorted([int(d) for d in os.listdir(p) if (p / d).is_dir()])


def load_dcc_cycles(base):
    """Return (cycle_numbers, capacities_Ah, t_start_s)."""
    subs = _dcc_subdirs(base)
    cycles, caps, t0s = [], [], []
    for s in subs:
        p = base / 'DCC' / str(s)
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


def load_checkup_capacity(base, state='CU_DCC_2'):
    p = base / state
    if not p.exists():
        return np.array([]), np.array([])
    subs = sorted([int(d) for d in os.listdir(p) if (p / d).is_dir()])
    cycles, caps = [], []
    for k, s in enumerate(subs):
        try:
            tc = np.loadtxt(p / str(s) / 'time.txt')
            ic = np.loadtxt(p / str(s) / 'current.txt')
            caps.append(abs(np.trapezoid(ic, tc)) / 3600.0)
            cycles.append((k + 1) * 25)
        except Exception:
            pass
    return np.array(cycles), np.array(caps)


def load_global(base):
    """Return (t, delta_film_m, eps_e, c_sei_reconstructed)."""
    t_g    = np.loadtxt(base / 'time.txt')
    delta  = np.loadtxt(base / 'delta_film_a.txt')
    eps_e  = np.loadtxt(base / 'eps_e_a.txt')
    # Attempt to load stored concentrations; reconstruct SEI if missing
    c_sei_path = base / 'c_sei_a.txt'
    c_lpl_path = base / 'c_lpl_a.txt'
    if c_sei_path.exists():
        c_sei = np.loadtxt(c_sei_path)
    else:
        c_sei = SEI_SCALE * delta   # exact for SEI-only; upper-bound for SEI+LPL
    c_lpl = np.loadtxt(c_lpl_path) if c_lpl_path.exists() else None
    return t_g, delta, eps_e, c_sei, c_lpl


def load_voltage_profile(base, cycle):
    """Return (Ah_discharged, voltage_V, capacity_Ah) for one DCC cycle."""
    p = base / 'DCC' / str(cycle)
    tc = np.loadtxt(p / 'time.txt')
    vc = np.loadtxt(p / 'voltage.txt')
    ic = np.loadtxt(p / 'current.txt')
    cap = abs(np.trapezoid(ic, tc)) / 3600.0
    ah  = np.abs([np.trapezoid(ic[:i+1], tc[:i+1]) for i in range(len(tc))]) / 3600.0
    return ah, vc, cap


def nearest_cycle(base, target):
    avail = _dcc_subdirs(base)
    return min(avail, key=lambda x: abs(x - target))


# ──────────────────────────────────────────────────────────────────
# Figure 8 – Discharge voltage profiles at selected cycles
# ──────────────────────────────────────────────────────────────────

n_cases = len(CASES)
fig8, axes8 = plt.subplots(1, n_cases, figsize=(5 * n_cases, 4), sharey=True)
if n_cases == 1:
    axes8 = [axes8]

cmap = plt.cm.plasma
for ax, (key, cfg) in zip(axes8, CASES.items()):
    base    = cfg['path']
    cyc_all, caps_all, _ = load_dcc_cycles(base)
    Q0      = caps_all[0]
    for i, cyc in enumerate(CYCLES_FOR_VOLTAGE):
        near = nearest_cycle(base, cyc)
        if near > max(_dcc_subdirs(base)):
            continue
        ah, vc, cap = load_voltage_profile(base, near)
        SoH_val = cap / Q0 * 100
        c_plot = cmap(i / max(len(CYCLES_FOR_VOLTAGE) - 1, 1))
        ax.plot(ah / Q0 * 100, vc, color=c_plot, lw=1.5,
                label=f'Cycle {near}  (SoH={SoH_val:.0f}%)')
    ax.set_xlabel('Normalised discharged capacity (%)')
    ax.set_title(cfg['label'])
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.set_xlim(0, 105)
    ax.set_ylim(2.7, 4.2)

axes8[0].set_ylabel('Voltage (V)')
fig8.suptitle('1C discharge profiles — Ecker2015 cell, 10 °C', y=1.02)
fig8.tight_layout()
fig8.savefig('results/comparison_fig8_voltages.eps', format='eps', bbox_inches='tight')
print('Saved results/comparison_fig8_voltages.eps')

# ──────────────────────────────────────────────────────────────────
# Figure 9 – Capacity loss and SoH
# ──────────────────────────────────────────────────────────────────

fig9, (ax9a, ax9b) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

for key, cfg in CASES.items():
    base = cfg['path']
    cyc, caps, t0s = load_dcc_cycles(base)
    Q0   = caps[0]
    SoH  = caps / Q0 * 100

    # 1C discharge SoH
    ax9a.plot(cyc, SoH, color=cfg['color'], ls=cfg['ls'],
              label=cfg['label'], lw=1.5)

    # 0.2C checkup (1C recharge) capacity
    cu_cyc, cu_caps = load_checkup_capacity(base, 'CU_DCC')
    if len(cu_caps) > 0:
        ax9b.plot(cu_cyc, cu_caps / cu_caps[0] * 100,
                  's', color=cfg['color'], ms=5, ls=cfg['ls'],
                  label=f'{cfg["label"]} — 0.2C checkup')

    # SEI lithium loss (from delta_film reconstruction)
    t_g, delta, eps_e, c_sei, _ = load_global(base)
    delta_at_t0 = np.interp(t0s, t_g, delta)
    c_sei_at_t0 = SEI_SCALE * delta_at_t0
    dQ_sei_frac = c_sei_at_t0 / CAP_DENOM * 100  # %
    label_sei = f'{cfg["label"]} — SEI Li loss (δ→c_sei)'
    ax9b.plot(cyc, dQ_sei_frac, color=cfg['color'], ls=':', lw=1.2,
              label=label_sei)

ax9a.axhline(80, color='gray', ls=':', lw=1, label='80% SoH threshold')
ax9a.set_ylabel('State of Health (%)')
ax9a.set_title('Capacity fade — 1C cycling (Ecker2015, 10 °C)')
ax9a.legend(fontsize=8)
ax9a.grid(True, alpha=0.3)
ax9a.set_ylim(70, 103)

ax9b.set_xlabel('Cycle number')
ax9b.set_ylabel('Capacity loss / SEI Li loss (%)')
ax9b.set_title('Capacity checkup (0.2C) and SEI contribution')
ax9b.legend(fontsize=7)
ax9b.grid(True, alpha=0.3)
ax9b.text(0.02, 0.97,
    'Note: SEI Li loss from δ→c_sei; δ includes LPL in SEI+LPL case',
    transform=ax9b.transAxes, va='top', fontsize=7, color='gray')

fig9.tight_layout()
fig9.savefig('results/comparison_fig9_capacity.eps', format='eps', bbox_inches='tight')
print('Saved results/comparison_fig9_capacity.eps')

# ──────────────────────────────────────────────────────────────────
# Figure 10 – Degradation state variables vs cycle
# ──────────────────────────────────────────────────────────────────

fig10, axes10 = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
ax10_film, ax10_por, ax10_csei, ax10_loss = (axes10[0, 0], axes10[0, 1],
                                              axes10[1, 0], axes10[1, 1])

for key, cfg in CASES.items():
    base = cfg['path']
    cyc, caps, t0s = load_dcc_cycles(base)
    Q0   = caps[0]
    t_g, delta, eps_e, c_sei, c_lpl = load_global(base)

    delta_at  = np.interp(t0s, t_g, delta)   * 1e9  # nm
    eps_at    = np.interp(t0s, t_g, eps_e)
    c_sei_at  = SEI_SCALE * np.interp(t0s, t_g, delta)  # mol/m³
    loss_sei  = c_sei_at / CAP_DENOM * 100  # %

    ax10_film.plot(cyc, delta_at, color=cfg['color'], ls=cfg['ls'],
                   label=cfg['label'], lw=1.5)
    ax10_por.plot(cyc, eps_at,    color=cfg['color'], ls=cfg['ls'],
                  label=cfg['label'], lw=1.5)
    ax10_csei.plot(cyc, c_sei_at, color=cfg['color'], ls=cfg['ls'],
                   label=cfg['label'], lw=1.5)
    ax10_loss.plot(cyc, (Q0 - caps) / Q0 * 100,
                   color=cfg['color'], ls=cfg['ls'],
                   label=f'{cfg["label"]} total', lw=1.5)
    ax10_loss.plot(cyc, loss_sei, color=cfg['color'], ls=':', lw=1.2,
                   label=f'{cfg["label"]} SEI Li loss')

ax10_film.set_ylabel('Film thickness (nm)')
ax10_film.set_title('SEI/LPL film thickness (spatial avg)')
ax10_film.legend(fontsize=8)
ax10_film.grid(True, alpha=0.3)

ax10_por.set_ylabel('Electrolyte porosity (−)')
ax10_por.set_title('Negative electrode porosity')
ax10_por.legend(fontsize=8)
ax10_por.grid(True, alpha=0.3)

ax10_csei.set_ylabel(r'$c_\mathrm{SEI}$ (mol/m³)')
ax10_csei.set_title(r'SEI concentration (reconstructed from $\delta$)')
ax10_csei.legend(fontsize=8)
ax10_csei.grid(True, alpha=0.3)
ax10_csei.text(0.02, 0.97,
    'c_SEI = a_s·(ρ/M)·δ (exact for SEI-only)',
    transform=ax10_csei.transAxes, va='top', fontsize=7, color='gray')

ax10_loss.set_ylabel('Capacity loss (%)')
ax10_loss.set_title('Total vs SEI Li loss')
ax10_loss.legend(fontsize=8)
ax10_loss.grid(True, alpha=0.3)

for ax in axes10[1, :]:
    ax.set_xlabel('Cycle number')

fig10.suptitle('Degradation metrics — Ecker2015 cell, 1C at 10 °C', y=1.01)
fig10.tight_layout()
fig10.savefig('results/comparison_fig10_degradation.eps', format='eps', bbox_inches='tight')
print('Saved results/comparison_fig10_degradation.eps')

# ──────────────────────────────────────────────────────────────────
# Figure 10b – Transient DCC profiles at selected cycles
# ──────────────────────────────────────────────────────────────────

n_trans = len(CYCLES_TRANSIENT)
fig10b, axes10b = plt.subplots(n_trans, n_cases, figsize=(5.5 * n_cases, 3.5 * n_trans),
                                sharex='col', sharey='row')
if n_cases == 1:
    axes10b = axes10b[:, np.newaxis]

for col, (key, cfg) in enumerate(CASES.items()):
    base    = cfg['path']
    cyc_all, caps_all, _ = load_dcc_cycles(base)
    Q0      = caps_all[0]
    t_g     = np.loadtxt(base / 'time.txt')
    delta_g = np.loadtxt(base / 'delta_film_a.txt')

    for row, cyc_target in enumerate(CYCLES_TRANSIENT):
        ax = axes10b[row, col]
        near = nearest_cycle(base, cyc_target)
        p    = base / 'DCC' / str(near)
        tc   = np.loadtxt(p / 'time.txt')
        vc   = np.loadtxt(p / 'voltage.txt')
        ic   = np.loadtxt(p / 'current.txt')
        cap  = abs(np.trapezoid(ic, tc)) / 3600.0
        ah   = np.abs([np.trapezoid(ic[:i+1], tc[:i+1]) for i in range(len(tc))]) / 3600.0

        ax2 = ax.twinx()
        # Film growth rate proxy: d(delta)/dt interpolated at cycle times
        delta_cyc = np.interp(tc, t_g, delta_g)
        dt_arr    = np.gradient(tc)
        dt_arr    = np.where(np.abs(dt_arr) < 1e-10, 1e-10, dt_arr)
        ddelta_dt = np.gradient(delta_cyc, tc)  # m/s
        # Smooth with 20-point moving average
        kernel = np.ones(20) / 20.0
        ddelta_sm = np.convolve(ddelta_dt * 1e12, kernel, mode='same')  # pm/s

        ax.plot(ah / Q0 * 100, vc, color=cfg['color'], lw=1.5,
                label=f'Voltage (Cycle {near})')
        ax2.plot(ah / Q0 * 100, ddelta_sm, color='#7f7f7f', lw=1.0, ls=':',
                 label=r'$d\delta/dt$ (pm/s)')
        ax.set_ylabel('Voltage (V)')
        ax2.set_ylabel(r'$d\delta/dt$ (pm/s)', color='#7f7f7f')
        ax2.tick_params(axis='y', labelcolor='#7f7f7f')
        SoH_val = cap / Q0 * 100
        ax.set_title(f'{cfg["label"]} — Cycle {near}  (SoH={SoH_val:.1f}%)')
        ax.grid(True, color='lightgray', linewidth=0.5)
        if row == n_trans - 1:
            ax.set_xlabel('Normalised discharged capacity (%)')
        # combined legend
        lines_v, labels_v = ax.get_legend_handles_labels()
        lines_r, labels_r = ax2.get_legend_handles_labels()
        ax.legend(lines_v + lines_r, labels_v + labels_r, fontsize=7)

fig10b.suptitle('Transient discharge profiles and film growth rate', y=1.01)
fig10b.tight_layout()
fig10b.savefig('results/comparison_fig10b_transient.eps', format='eps', bbox_inches='tight')
print('Saved results/comparison_fig10b_transient.eps')

plt.show()
