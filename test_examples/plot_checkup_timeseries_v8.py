# Checkup timeseries figure for lpl_v8 runs.
# Analogous to plot_checkup_timeseries.py but reads from a single lpl_v8_* result folder.
#
# Usage:
#   conda run -n fenics-env python3 test_examples/plot_checkup_timeseries_v8.py [--case lpl_v8_10C_25deg]
#
# Produces two figures per available checkup:
#   Figure A – first CCCV half of checkup (CU_CCC + CU_CCV + CU_DCC + CU_DC0)
#   Figure B – second CCCV half (CU_CCC_2 + CU_CCV_2 + CU_DCC_2)
#
# Each figure has 4 panels: Voltage, Current (C-rate), SEI rate, LPL rate.

import sys; sys.path.insert(0, '..')
import argparse
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

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--case', default='lpl_v8_10C_25deg',
                    help='Result folder name under results/')
args = parser.parse_args()

CASE = Path('results') / args.case
Q_NOMINAL = 7.5    # Ah for C-rate normalisation
SMOOTH_WIN = 60    # smoothing window for derivative plots

F_CONST = 96485.0
EPS_S   = 0.372
R_P     = 1.37e-5
A_S     = 3.0 * EPS_S / R_P   # specific interfacial area (m²/m³)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subdirs(path):
    """Return sorted integer subfolder indices that exist under path."""
    if not path.exists():
        return []
    return sorted([int(d) for d in os.listdir(path)
                   if (path / d).is_dir() and d.isdigit()])


def stitch_states(base, states, k):
    """
    Concatenate time/current/voltage for states[0]/k … states[-1]/k.
    Returns (t_rel_h, current_A, voltage_V, t_abs_start, t_abs_end).
    Inserts NaN breaks at state boundaries.
    """
    ts, ics, vcs = [], [], []
    for s in states:
        p = base / s / str(k)
        if not p.exists():
            continue
        tf = p / 'time.txt'; cf = p / 'current.txt'; vf = p / 'voltage.txt'
        if not (tf.exists() and cf.exists() and vf.exists()):
            continue
        tc = np.loadtxt(tf)
        ic = np.loadtxt(cf)
        vc = np.loadtxt(vf)
        if len(np.atleast_1d(tc)) == 0:
            continue
        if ts:
            ts.append(np.array([tc[0] if np.ndim(tc) > 0 else tc]))
            ics.append(np.array([np.nan]))
            vcs.append(np.array([np.nan]))
        ts.append(np.atleast_1d(tc))
        ics.append(np.atleast_1d(ic))
        vcs.append(np.atleast_1d(vc))

    if not ts:
        return None, None, None, None, None

    t  = np.concatenate(ts)
    ic = np.concatenate(ics)
    vc = np.concatenate(vcs)
    t_valid = t[~np.isnan(t)]
    if len(t_valid) == 0:
        return None, None, None, None, None
    t0, t1 = t_valid[0], t_valid[-1]
    return (t - t0) / 3600.0, ic, vc, t0, t1


def concentration_rate(t_g, c_g, t0, t1, smooth_win):
    """
    Compute dc/dt over [t0, t1] from global timeseries.
    Returns (t_rel_h, rate_mA_m2).
    """
    mask = (t_g >= t0 - 5.0) & (t_g <= t1 + 5.0)
    tg = t_g[mask]; cg = c_g[mask]
    if len(tg) < 5:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    rate = np.gradient(cg, tg) * F_CONST * 1e3 / A_S
    win = min(smooth_win, max(3, len(rate) // 8))
    rate_sm = uniform_filter1d(rate, size=win)
    return (tg - t0) / 3600.0, rate_sm


# ─────────────────────────────────────────────────────────────────────────────
# Load global ageing timeseries
# ─────────────────────────────────────────────────────────────────────────────

t_g      = np.loadtxt(CASE / 'time.txt')
c_sei_g  = np.loadtxt(CASE / 'c_sei_a.txt')
c_lpl_g  = np.loadtxt(CASE / 'c_lpl_a.txt')
n = min(len(t_g), len(c_sei_g), len(c_lpl_g))
t_g, c_sei_g, c_lpl_g = t_g[:n], c_sei_g[:n], c_lpl_g[:n]

# ─────────────────────────────────────────────────────────────────────────────
# Discover available checkups
# ─────────────────────────────────────────────────────────────────────────────

avail = _subdirs(CASE / 'CU_CCC')
if not avail:
    print('No checkup data found yet. Run the simulation further.')
    sys.exit(0)

N = len(avail)
cmap   = plt.cm.plasma
colors = [cmap(i / max(N - 1, 1)) for i in range(N)]

# Approximate cycle count for each checkup:
#   checkup 1 = initial (cycle 0); subsequent ones ~every 51 cycles
def checkup_label(k):
    if k == 1:
        return f'CU {k}  (initial, cycle 0)'
    return f'CU {k}  (~cycle {(k-1)*51})'

# ─────────────────────────────────────────────────────────────────────────────
# Build figures
# ─────────────────────────────────────────────────────────────────────────────

CU_STATES_A = ['CU_CCC',   'CU_CCV',   'CU_DCC',   'CU_DC0']
CU_STATES_B = ['CU_CCC_2', 'CU_CCV_2', 'CU_DCC_2']

def build_fig(title):
    fig, axmat = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    ax_v, ax_i   = axmat[0]
    ax_sei, ax_lpl = axmat[1]
    ax_v.set_title(title, pad=6)
    ax_v.set_ylabel('Voltage (V)')
    ax_i.set_ylabel('Current (C-rate)')
    ax_sei.set_ylabel('SEI rate (mA/m², + = formation)')
    ax_lpl.set_ylabel('LPL rate (mA/m², + = plating, − = stripping)')
    ax_sei.set_xlabel('Time within cycle (h)')
    ax_lpl.set_xlabel('Time within cycle (h)')
    for ax in (ax_v, ax_i, ax_sei, ax_lpl):
        ax.grid(True, color='lightgray', linewidth=0.5)
    return fig, (ax_v, ax_i, ax_sei, ax_lpl)


def populate(axes, states, avail, colors):
    ax_v, ax_i, ax_sei, ax_lpl = axes
    t_cross_list = []
    for k, color in zip(avail, colors):
        t_rel, ic, vc, t0, t1 = stitch_states(CASE, states, k)
        if t_rel is None:
            continue
        lbl = checkup_label(k)
        ax_v.plot(t_rel, vc, color=color, lw=1.4, label=lbl)
        ax_i.plot(t_rel, ic / Q_NOMINAL, color=color, lw=1.4)

        ic_nn = ic[~np.isnan(ic)] if ic is not None else np.array([])
        t_nn  = t_rel[~np.isnan(ic)] if ic is not None else np.array([])
        sc    = np.where(np.diff(np.sign(ic_nn)))[0] if len(ic_nn) > 1 else []
        if len(sc):
            t_cross_list.append(t_nn[sc[0]])

        tr_sei, r_sei = concentration_rate(t_g, c_sei_g, t0, t1, SMOOTH_WIN)
        ax_sei.plot(tr_sei, np.maximum(r_sei, 0.0), color=color, lw=1.4)

        tr_lpl, r_lpl = concentration_rate(t_g, c_lpl_g, t0, t1, SMOOTH_WIN)
        ax_lpl.plot(tr_lpl, r_lpl, color=color, lw=1.4)

    ax_lpl.axhline(0, color='k', lw=0.6, ls='--')
    ax_v.legend(loc='lower right', fontsize=8)
    ax_v.set_ylim(2.6, 4.3)

    if t_cross_list:
        t_cx = float(np.median(t_cross_list))
        xlim = ax_v.get_xlim()
        t_max = xlim[1] if xlim[1] > 0 else 11.0
        for ax in axes:
            ax.axvline(t_cx, color='#555', ls=':', lw=0.8)
        ax_v.text(t_cx * 0.45, 2.68, 'Charge', ha='center', fontsize=8, color='#555')
        ax_v.text((t_max + t_cx) * 0.5, 2.68, 'Discharge', ha='center', fontsize=8, color='#555')


case_label = args.case.replace('lpl_v8_', '').replace('_', ' ')

figA, axesA = build_fig(
    f'Checkup protocol 1 — C/5 CCCV + C/5 discharge\n'
    f'{args.case}  ({N} checkup(s) available)'
)
populate(axesA, CU_STATES_A, avail, colors)
figA.tight_layout(pad=1.0)

figB, axesB = build_fig(
    f'Checkup protocol 2 — C/5 CCCV + 1C discharge (SOH reference)\n'
    f'{args.case}  ({N} checkup(s) available)'
)
populate(axesB, CU_STATES_B, avail, colors)
figB.tight_layout(pad=1.0)

out_a = str(CASE / 'checkup_timeseries.jpg')
out_b = str(CASE / 'checkup_timeseries_2.jpg')
figA.savefig(out_a, format='jpeg', dpi=200, bbox_inches='tight')
figB.savefig(out_b, format='jpeg', dpi=200, bbox_inches='tight')
print(f'Saved {out_a}')
print(f'Saved {out_b}')

plt.show()
