# Capacity fade comparison for lpl_v8_* runs.
# Plots 1C SOH discharge capacity (from CU_DCC_2/k) vs cycle number for every
# completed lpl_v8_* result folder under results/.

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

RESULTS = Path('results')
CYCLES_PER_CHECKUP = 51       # CU 1 = cycle 0, CU k = ~(k-1)*51
Q_NOMINAL = 7.5               # Ah


def discharge_capacity(folder):
    """Return capacity in Ah from a CU_DCC_2/k folder, or None on failure."""
    try:
        t = np.loadtxt(folder / 'time.txt')
        i = np.loadtxt(folder / 'current.txt')
        if t.size < 2:
            return None
        return -np.trapezoid(i, t) / 3600.0
    except Exception:
        return None


def load_case(case_dir):
    """Return (cycles, caps_Ah) for a case based on CU_DCC_2 indices."""
    cu = case_dir / 'CU_DCC_2'
    if not cu.exists():
        return np.array([]), np.array([])
    indices = sorted([int(d.name) for d in cu.iterdir()
                      if d.is_dir() and d.name.isdigit()])
    cycles, caps = [], []
    for k in indices:
        c = discharge_capacity(cu / str(k))
        if c is None or c <= 0:
            continue
        cycles.append(0 if k == 1 else (k - 1) * CYCLES_PER_CHECKUP)
        caps.append(c)
    return np.array(cycles), np.array(caps)


cases = sorted([d for d in RESULTS.iterdir()
                if d.is_dir() and d.name.startswith('lpl_v8_')
                and (d / 'CU_DCC_2').exists()])

# Filter out smoke/test/reference/log dirs
cases = [c for c in cases if not any(s in c.name for s in
         ['smoke', 'cu_test', 'reference', 'logs'])]

if not cases:
    print('No lpl_v8_* result folders with CU_DCC_2 found.')
    sys.exit(0)

# Color by C-rate, marker by temperature
crate_colors = {'05': '#1f77b4', '10': '#d62728', '20': '#2ca02c'}
temp_markers = {'10': 's', '25': 'o', '40': '^'}

fig, (ax_abs, ax_soh) = plt.subplots(1, 2, figsize=(13, 5))

for case in cases:
    label = case.name.replace('lpl_v8_', '')
    parts = label.split('_')
    crate = parts[0].replace('C', '')
    temp = parts[1].replace('deg', '')
    color = crate_colors.get(crate, 'k')
    marker = temp_markers.get(temp, 'o')

    cycles, caps = load_case(case)
    if len(cycles) < 2:
        continue

    ls = '-'
    ax_abs.plot(cycles, caps, color=color, ls=ls, marker=marker, ms=5,
                lw=1.4, label=label.replace('_', ' '))
    ax_soh.plot(cycles, caps / caps[0] * 100, color=color, ls=ls,
                marker=marker, ms=5, lw=1.4,
                label=label.replace('_', ' '))

ax_abs.axhline(Q_NOMINAL, color='gray', ls=':', lw=0.8, label='Q_nominal = 7.5 Ah')
ax_abs.set_xlabel('Cycle number')
ax_abs.set_ylabel('1C discharge capacity (Ah)')
ax_abs.set_title('Absolute capacity (CU_DCC_2)')
ax_abs.grid(True, color='lightgray', linewidth=0.5)
ax_abs.legend(loc='best', fontsize=8)

ax_soh.axhline(80, color='gray', ls=':', lw=0.8, label='80% SoH')
ax_soh.set_xlabel('Cycle number')
ax_soh.set_ylabel('SoH (% of CU 1 capacity)')
ax_soh.set_title('Relative SoH (normalised to initial checkup)')
ax_soh.grid(True, color='lightgray', linewidth=0.5)
ax_soh.legend(loc='best', fontsize=8)

fig.suptitle('lpl_v8 capacity fade — 1C discharge from CU_DCC_2', y=1.0)
fig.tight_layout(pad=1.0)

out = RESULTS / 'lpl_v8_capacity_fade.jpg'
fig.savefig(out, format='jpeg', dpi=200, bbox_inches='tight')
print(f'Saved {out}')

# Also print the numerical table
print()
print(f'{"Case":<14} {"Cycle":>6} {"Q (Ah)":>9} {"SoH":>7}')
print('-' * 40)
for case in cases:
    label = case.name.replace('lpl_v8_', '')
    cycles, caps = load_case(case)
    if len(caps) == 0:
        continue
    q0 = caps[0]
    for c, q in zip(cycles, caps):
        print(f'{label:<14} {c:>6d} {q:>9.4f} {q/q0*100:>6.2f}%')
    print()

plt.show()
