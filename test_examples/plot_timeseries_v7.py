# Per-scenario cycle timeseries plots for lpl_v7 parametric study.
# Produces two 4-panel figures per scenario (saved inside the scenario folder):
#   checkup_timeseries.jpg  — full CCCV charge + discharge  (CCC + CCV + DCC + DC0)
#   checkup_timeseries_2.jpg — discharge only               (DCC + DC0)
#
# Cycles are sampled uniformly (up to N_SAMPLE) to show ageing evolution.
#
# Run from batFEM root:
#   conda run -n fenics-env python3 test_examples/plot_timeseries_v7.py

import sys; sys.path.insert(0, '..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

C_RATES = [0.5, 1.0, 2.0]
TEMPS   = [10,  25,  40]

def run_name(C, T):
    return f'lpl_v7_{int(C*10):02d}C_{T:02d}deg'

Q_NOMINAL  = 7.5    # Ah, for C-rate normalisation
N_SAMPLE   = 10     # max number of cycles to overlay
SMOOTH_WIN = 60     # smoothing window for rate plots

# Physical constants & NE_Ecker2015 material parameters
F_CONST = 96485.0
EPS_S   = 0.372; R_P = 1.37e-5
A_S     = 3.0 * EPS_S / R_P

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _subdirs(path):
    try:
        return sorted([int(d) for d in os.listdir(path)
                       if (path / d).is_dir()])
    except FileNotFoundError:
        return []


def stitch_states(base, states, k):
    """
    Concatenate time/current/voltage for the given cycle index k across states.
    Returns (t_rel_h, current_A, voltage_V, t_abs_start, t_abs_end) or
            (None, None, None, None, None) if no data found.
    """
    ts, ics, vcs = [], [], []
    for s in states:
        p = base / s / str(k)
        if not p.exists():
            continue
        try:
            tc = np.loadtxt(p / 'time.txt')
            ic = np.loadtxt(p / 'current.txt')
            vc = np.loadtxt(p / 'voltage.txt')
        except Exception:
            continue
        if ts:
            nan = np.array([np.nan])
            ts.append(np.array([tc[0]]))
            ics.append(nan); vcs.append(nan)
        ts.append(tc); ics.append(ic); vcs.append(vc)

    if not ts:
        return None, None, None, None, None

    t  = np.concatenate(ts)
    ic = np.concatenate(ics)
    vc = np.concatenate(vcs)
    valid = t[~np.isnan(t)]
    t0, t1 = valid[0], valid[-1]
    return (t - t0) / 3600.0, ic, vc, t0, t1


def concentration_rate(t_g, c_g, t_start, t_end, smooth_win, clip_neg=False):
    """
    dc/dt → surface current density [mA/m²].
    +ve = formation/plating, -ve = dissolution/stripping.
    """
    mask = (t_g >= t_start - 5.0) & (t_g <= t_end + 5.0)
    tg = t_g[mask]; cg = c_g[mask]
    if len(tg) < 5:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    rate = np.gradient(cg, tg) * F_CONST * 1e3 / A_S
    win  = min(smooth_win, max(3, len(rate) // 8))
    rate_sm = uniform_filter1d(rate, size=win)
    if clip_neg:
        rate_sm = np.maximum(rate_sm, 0.0)
    return (tg - t_start) / 3600.0, rate_sm


# ──────────────────────────────────────────────────────────────────
# Figure builder
# ──────────────────────────────────────────────────────────────────

def build_figure(title):
    fig, axmat = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    ax_v, ax_i = axmat[0]
    ax_sei, ax_lpl = axmat[1]
    ax_v.set_title(title, pad=6)
    ax_v.set_ylabel('Voltage (V)')
    ax_i.set_ylabel('Current (C-rate)')
    ax_sei.set_ylabel('SEI rate (mA/m², +formation)')
    ax_lpl.set_ylabel('LPL rate (mA/m², +plating/−stripping)')
    ax_sei.set_xlabel('Time within cycle (h)')
    ax_lpl.set_xlabel('Time within cycle (h)')
    for ax in axmat.flat:
        ax.grid(True, color='lightgray', linewidth=0.5)
    return fig, (ax_v, ax_i, ax_sei, ax_lpl)


def populate_figure(axes, base, t_g, c_sei_g, c_lpl_g, ks, colors, states,
                    smooth_win):
    ax_v, ax_i, ax_sei, ax_lpl = axes
    t_cross_list = []

    for k, color in zip(ks, colors):
        label = f'Cycle {k}'
        t_rel, ic, vc, t0, t1 = stitch_states(base, states, k)
        if t_rel is None:
            continue
        ax_v.plot(t_rel, vc, color=color, lw=1.4, label=label)
        ax_i.plot(t_rel, ic / Q_NOMINAL, color=color, lw=1.4)

        ic_nn = ic[~np.isnan(ic)]
        t_nn  = t_rel[~np.isnan(ic)]
        sc    = np.where(np.diff(np.sign(ic_nn)))[0]
        if len(sc):
            t_cross_list.append(t_nn[sc[0]])

        tr_sei, r_sei = concentration_rate(t_g, c_sei_g, t0, t1,
                                           smooth_win, clip_neg=True)
        ax_sei.plot(tr_sei, r_sei, color=color, lw=1.4)

        tr_lpl, r_lpl = concentration_rate(t_g, c_lpl_g, t0, t1,
                                           smooth_win, clip_neg=False)
        ax_lpl.plot(tr_lpl, r_lpl, color=color, lw=1.4)

    ax_lpl.axhline(0, color='k', lw=0.6, ls='--')
    ax_v.legend(loc='lower right', fontsize=8)
    ax_v.set_ylim(2.6, 4.3)

    if t_cross_list:
        t_cx  = float(np.median(t_cross_list))
        xlim  = ax_v.get_xlim()
        t_max = xlim[1] if xlim[1] > 0 else 11.0
        for ax in axes:
            ax.axvline(t_cx, color='#555', ls=':', lw=0.8)
        ax_v.text(t_cx * 0.45, 2.68, 'Charge',
                  ha='center', fontsize=8, color='#555')
        ax_v.text((t_max + t_cx) * 0.5, 2.68, 'Discharge',
                  ha='center', fontsize=8, color='#555')


# ──────────────────────────────────────────────────────────────────
# Main loop — one scenario at a time
# ──────────────────────────────────────────────────────────────────

RESULTS = Path('results')

FULL_STATES = ['CCC', 'CCV', 'DCC', 'DC0']
DISCH_STATES = ['DCC', 'DC0']

for C in C_RATES:
    for T in TEMPS:
        name = run_name(C, T)
        base = RESULTS / name

        if not base.exists():
            print(f'{name}: folder missing, skip')
            continue

        dcc_path = base / 'DCC'
        avail = _subdirs(dcc_path)
        if not avail:
            print(f'{name}: no DCC cycles, skip')
            continue

        # Sample up to N_SAMPLE cycles uniformly
        if len(avail) <= N_SAMPLE:
            ks = avail
        else:
            idx = np.round(np.linspace(0, len(avail) - 1, N_SAMPLE)).astype(int)
            ks = [avail[i] for i in idx]

        n_ks = len(ks)
        cmap   = plt.cm.plasma
        colors = [cmap(i / max(n_ks - 1, 1)) for i in range(n_ks)]

        # Load global timeseries
        try:
            t_g     = np.loadtxt(base / 'time.txt')
            c_sei_g = np.loadtxt(base / 'c_sei_a.txt')
            c_lpl_g = np.loadtxt(base / 'c_lpl_a.txt')
        except Exception as e:
            print(f'{name}: cannot load global timeseries ({e}), skip')
            continue

        title_base = (f'SEI+LPL model — {name}\n'
                      r'$i_{0,\mathrm{SEI}}=10^{-13}$ mol/m²/s,  '
                      r'$i_{0,\mathrm{LPL}}=0.01$ A/m²,  $U_\mathrm{LPL}=0$ V')

        # Figure A – full CCCV cycle
        figA, axesA = build_figure(title_base + '\nFull charge+discharge cycle')
        populate_figure(axesA, base, t_g, c_sei_g, c_lpl_g,
                        ks, colors, FULL_STATES, SMOOTH_WIN)
        figA.tight_layout(pad=1.0)
        outA = base / 'checkup_timeseries.jpg'
        figA.savefig(outA, format='jpeg', dpi=200, bbox_inches='tight')
        plt.close(figA)

        # Figure B – discharge only
        figB, axesB = build_figure(title_base + '\nDischarge only')
        populate_figure(axesB, base, t_g, c_sei_g, c_lpl_g,
                        ks, colors, DISCH_STATES, SMOOTH_WIN)
        figB.tight_layout(pad=1.0)
        outB = base / 'checkup_timeseries_2.jpg'
        figB.savefig(outB, format='jpeg', dpi=200, bbox_inches='tight')
        plt.close(figB)

        print(f'{name}: saved checkup_timeseries.jpg and checkup_timeseries_2.jpg  '
              f'({len(ks)} cycles: {ks[0]}..{ks[-1]})')

print('Done.')
