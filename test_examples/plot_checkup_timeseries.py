# Date: 30/04/2026
# Auth: Manuel Cremades, manuel.cremades@usc.es
#
# Two figures analogous to Keil & Jossen 2020 (JES) Fig. 10:
#
#   Figure A  –  Checkup CCCV cycle (0.2C at 25°C)
#   Figure B  –  First 1C cycling CCCV cycle after each checkup
#
# Both show 4 stacked panels:
#   1 – Voltage (V)
#   2 – Current (C-rate)
#   3 – SEI formation rate (nm/h proxy from d(δ_film)/dt, SEI-only run)
#   4 – LPL formation rate (nm/h proxy = difference between SEI+LPL and SEI-only rates)
#
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_checkup_timeseries.py

import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

CASE_SEI     = Path('results/None')       # SEI-only (δ_film = SEI reference)
CASE_SEI_LPL = Path('results/lpl_100_v4_cu')   # SEI + LPL (k_lpl=10, U_lpl=0.05V, i0_lpl=1e-3)

# Checkup indices to overlay (1-based). None → all available.
CHECKUPS_TO_PLOT = None   # ← set to e.g. [1, 2] to show only 2 checkups

Q_NOMINAL = 7.5   # Ah, for C-rate normalisation

# Smoothing half-window (in global timestep units, applied independently per window)
SMOOTH_WIN_CHECKUP = 80

# ──────────────────────────────────────────────────────────────────
# Physical constants & material parameters (NE_Ecker2015.json)
# ──────────────────────────────────────────────────────────────────

F_CONST = 96485.0
EPS_S   = 0.372 ; R_P = 1.37e-5
A_S     = 3.0 * EPS_S / R_P   # specific interfacial area (m²/m³)

# ──────────────────────────────────────────────────────────────────
# Helpers – data loading
# ──────────────────────────────────────────────────────────────────

def _subdirs(path):
    return sorted([int(d) for d in os.listdir(path) if (path / d).is_dir()])


def stitch_states(base, states, k):
    """
    Concatenate time, current, voltage arrays for states[0]/k … states[-1]/k.
    Inserts NaN breaks at state boundaries so lines don't connect across gaps.
    Returns (t_rel_h, current_A, voltage_V, t_abs_start, t_abs_end).
    """
    ts, ics, vcs = [], [], []
    for s in states:
        p = base / s / str(k)
        if not p.exists():
            continue
        tc = np.loadtxt(p / 'time.txt')
        ic = np.loadtxt(p / 'current.txt')
        vc = np.loadtxt(p / 'voltage.txt')
        if ts:   # insert a NaN break to avoid connecting state boundaries
            nan = np.array([np.nan])
            ts.append(np.array([tc[0]]))  # keep time continuous
            ics.append(nan); vcs.append(nan)
        ts.append(tc); ics.append(ic); vcs.append(vc)

    if not ts:
        return None, None, None, None, None

    t  = np.concatenate(ts)
    ic = np.concatenate(ics)
    vc = np.concatenate(vcs)
    t0 = t[0]
    t1 = t[~np.isnan(t)][-1]
    return (t - t0) / 3600.0, ic, vc, t0, t1


def concentration_rate(t_g, c_g, t_start, t_end, smooth_win, clip_neg=False):
    """
    Extract dc/dt from global concentration array in [t_start, t_end] and
    convert to reaction current density at the particle surface.

    j [mA/m²] = +dc/dt × F / a_s × 1e3
      positive = formation/plating  (c increasing)
      negative = dissolution/stripping (c decreasing)
    """
    mask = (t_g >= t_start - 5.0) & (t_g <= t_end + 5.0)
    tg = t_g[mask]; cg = c_g[mask]
    if len(tg) < 5:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    rate = np.gradient(cg, tg) * F_CONST * 1e3 / A_S   # mA/m²
    win = min(smooth_win, max(3, len(rate) // 8))
    rate_sm = uniform_filter1d(rate, size=win)
    if clip_neg:
        rate_sm = np.maximum(rate_sm, 0.0)
    t_rel = (tg - t_start) / 3600.0
    return t_rel, rate_sm


# ──────────────────────────────────────────────────────────────────
# Load global timeseries once
# ──────────────────────────────────────────────────────────────────

t_g_lpl   = np.loadtxt(CASE_SEI_LPL / 'time.txt')
c_sei_g   = np.loadtxt(CASE_SEI_LPL / 'c_sei_a.txt')
c_lpl_g   = np.loadtxt(CASE_SEI_LPL / 'c_lpl_a.txt')

# ──────────────────────────────────────────────────────────────────
# Select checkups
# ──────────────────────────────────────────────────────────────────

avail = _subdirs(CASE_SEI_LPL / 'CU_CCC')
ks    = [k for k in (CHECKUPS_TO_PLOT or avail) if k in avail]
N     = len(ks)
cmap  = plt.cm.plasma
colors = [cmap(i / max(N - 1, 1)) for i in range(N)]

# ──────────────────────────────────────────────────────────────────
# Helper – build one 4-panel figure
# ──────────────────────────────────────────────────────────────────

def build_figure(title, ylabel_extra=''):
    fig, axmat = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    ax_v, ax_i = axmat[0]
    ax_sei, ax_lpl = axmat[1]
    ax_v.set_title(title, pad=6)
    ax_v.set_ylabel('Voltage (V)')
    ax_i.set_ylabel('Current (C-rate)')
    ax_sei.set_ylabel('SEI rate (mA/m², +formation)' + ylabel_extra)
    ax_lpl.set_ylabel('LPL rate (mA/m², +plating/−stripping)' + ylabel_extra)
    ax_sei.set_xlabel('Time within cycle (h)')
    ax_lpl.set_xlabel('Time within cycle (h)')
    axes = (ax_v, ax_i, ax_sei, ax_lpl)
    for ax in axes:
        ax.grid(True, color='lightgray', linewidth=0.5)
    return fig, axes


def populate_figure(axes, ks, colors, states, smooth_win,
                    charge_discharge_label=True):
    ax_v, ax_i, ax_sei, ax_lpl = axes

    t_cross_list = []

    for k, color in zip(ks, colors):
        label = f'CU {k}  (after cycle {k * 25})'

        # ── Voltage and current ──────────────────────────────────────
        t_rel, ic, vc, t0, t1 = stitch_states(CASE_SEI_LPL, states, k)
        if t_rel is None:
            continue
        ax_v.plot(t_rel, vc, color=color, lw=1.4, label=label)
        ic_crate = ic / Q_NOMINAL
        ax_i.plot(t_rel, ic_crate, color=color, lw=1.4)

        # Locate charge/discharge crossover (first sign change in current)
        ic_nn = ic[~np.isnan(ic)]
        t_nn  = t_rel[~np.isnan(ic)]
        sc    = np.where(np.diff(np.sign(ic_nn)))[0]
        if len(sc):
            t_cross_list.append(t_nn[sc[0]])

        # ── SEI formation rate ───────────────────────────────────────
        tr_sei, r_sei = concentration_rate(t_g_lpl, c_sei_g, t0, t1,
                                           smooth_win, clip_neg=True)
        ax_sei.plot(tr_sei, r_sei, color=color, lw=1.4)

        # ── LPL reaction rate (+ = plating, − = stripping) ───────────
        tr_lpl, r_lpl = concentration_rate(t_g_lpl, c_lpl_g, t0, t1,
                                           smooth_win, clip_neg=False)
        ax_lpl.plot(tr_lpl, r_lpl, color=color, lw=1.4)

    # ── Decoration ──────────────────────────────────────────────────
    ax_lpl.axhline(0, color='k', lw=0.6, ls='--')
    ax_v.legend(loc='lower right', fontsize=8)
    ax_v.set_ylim(2.6, 4.3)

    if charge_discharge_label and t_cross_list:
        t_cx = float(np.median(t_cross_list))
        t_max = ax_v.get_xlim()[1] if ax_v.get_xlim()[1] > 0 else 11.0
        for ax in axes:
            ax.axvline(t_cx, color='#555', ls=':', lw=0.8)
        ax_v.text(t_cx * 0.45, 2.68, 'Charge', ha='center', fontsize=8, color='#555')
        ax_v.text((t_max + t_cx) * 0.5, 2.68, 'Discharge', ha='center',
                  fontsize=8, color='#555')

    return axes


# ──────────────────────────────────────────────────────────────────
# Figure A – Checkup CCCV cycle (0.2C at 25 °C)
# ──────────────────────────────────────────────────────────────────

CU_STATES   = ['CU_CCC',   'CU_CCV',   'CU_DCC',   'CU_DC0']
CU_STATES_2 = ['CU_DCC_2', 'CU_DC0_2']

figA, axesA = build_figure(
    'Checkup protocol 1 — CCCV + discharge\n'
    'Ecker2015 cell, SEI+LPL model'
)
populate_figure(axesA, ks, colors, CU_STATES, SMOOTH_WIN_CHECKUP)

figA.tight_layout(pad=1.0)
figA.savefig('results/checkup_timeseries.eps', format='eps', bbox_inches='tight')
figA.savefig('results/checkup_timeseries.jpg', format='jpeg', dpi=200, bbox_inches='tight')
print('Saved results/checkup_timeseries.eps / .jpg')

# ──────────────────────────────────────────────────────────────────
# Figure B – Checkup protocol 2 (CU_*_2 states)
# ──────────────────────────────────────────────────────────────────

figB, axesB = build_figure(
    'Checkup protocol 2 — CCCV + discharge\n'
    'Ecker2015 cell, SEI+LPL model'
)
populate_figure(axesB, ks, colors, CU_STATES_2, SMOOTH_WIN_CHECKUP)

figB.tight_layout(pad=1.0)
figB.savefig('results/checkup_timeseries_2.eps', format='eps', bbox_inches='tight')
figB.savefig('results/checkup_timeseries_2.jpg', format='jpeg', dpi=200, bbox_inches='tight')
print('Saved results/checkup_timeseries_2.eps / .jpg')

plt.show()
