import sys; sys.path.insert(0,'..');
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
import meshio
import json
import batFEM.class_battery

# ==========================================================
# CONFIGURATION
# ==========================================================

battery_name = "Prada_CCCV"
base_path = Path("results")
tol="1em2"
cases = {
    "SDIRK1": {"folder": f"{battery_name}_fixed", "style": "-", "color": "blue"},
    "SDIRK3CP":  {"folder": f"{battery_name}_SDIRK3CP_{tol}",  "style": "None",  "marker":"*", "color": "red"},
    "ESDIRK4":  {"folder": f"{battery_name}_ESDIRK4_{tol}",  "style": "None", "marker":"v", "color": "orange"},
    "ROW3P":   {"folder": f"{battery_name}_ROW3P_{tol}",   "style": "None", "marker":"^", "color": "green"},
    "ROW4PW2": {"folder": f"{battery_name}_ROW4PW2_{tol}", "style": "-",  "marker":"", "color": "purple"}
}

reference_case = "SDIRK1"


# ==========================================================
# UTILITIES
# ==========================================================

def load_time_series(case_name):
    folder = base_path / cases[case_name]["folder"]
    return {
        "time": np.loadtxt(folder / "time.txt"),
        "voltage": np.loadtxt(folder / "voltage.txt"),
        "xs_avg_a": np.loadtxt(folder / "xs_avg_a.txt"),
        "xs_sur_a": np.loadtxt(folder / "xs_sur_a.txt"),
    }

def rmse(a, b):
    return np.linalg.norm(a - b) / np.sqrt(len(a))

# ==========================================================
# LOAD ALL CASES
# ==========================================================

data_cases = {name: load_time_series(name) for name in cases}
ref = data_cases[reference_case]
times = ref["time"]

print(times[-1])
plt.figure()

for name, data in data_cases.items():
        if name == reference_case:
            plt.plot(data["time"], 
                     data["voltage"],
                     '--',
                     color=cases[name]["color"],
                     label=name)
        else:
            plt.plot(data["time"], 
                     data["voltage"],
                     '--',
                     color=cases[name]["color"],
                     label=name)

plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title(battery_name+', rtol =  1$\\times$ 10^-'+tol[-1])

plt.figure()

for name, data in data_cases.items():
        if name == reference_case:
            plt.plot(data["time"][1:],np.diff(data["time"]), 
                     color=cases[name]["color"],
                     ls='--',
                     marker='o',
                     label=name,ms=2)
        else:
            plt.plot(data["time"][1:],np.diff(data["time"]),
                     color=cases[name]["color"],
                     ls='--',
                     marker='o',
                     label=name,ms=2)
plt.xlabel('Time [s]')
plt.ylabel('Timestep [s]')
plt.title(battery_name+', rtol =  1$\\times$ 10^-'+tol[-1])

plt.figure()

for name, data in data_cases.items():
        if name == reference_case:
            pass
        else:
            v_interp = np.interp(data["time"], ref["time"], ref["voltage"])

            plt.plot(data["time"], 
                     1e3*abs(data["voltage"]-v_interp),
                     '--',
                     color=cases[name]["color"],
                     label=name)
            RMSE=np.sqrt(((1e3*abs(data["voltage"]-v_interp))**2).mean())
            MAE=max(1e3*abs(data["voltage"]-v_interp))
            print(name)
            print(RMSE,MAE)
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Error [mV]')

plt.show()