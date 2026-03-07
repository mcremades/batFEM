import sys; sys.path.insert(0,'..')
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import json
import meshio
import batFEM.class_battery

# ==========================================================
# CONFIG
# ==========================================================

battery_name = "Marquis2019"
test = "UDDS"
res_batfem = f"{battery_name}_{test}"
results_dir = Path("results") / res_batfem

input_profile = np.load(f"test_examples/{battery_name}/{test}_input_profile_p2d.npz")
output_profile = np.load(f"test_examples/{battery_name}/{test}_output_profile_p2d.npz")

time_pybamm = input_profile["time"]
target_times = np.linspace(0, 2400, 13)

# ==========================================================
# BATTERY PARAMETERS
# ==========================================================

with open(f"json_battery/cells/cell_{battery_name}.json") as f:
    json_file = json.load(f)

json_battery = batFEM.class_battery.parse_json(json_file)

cs_max_a = json_battery['negativeElectrode']['composition'][0]['maximumConcentration']['value']
cs_max_c = json_battery['positiveElectrode']['composition'][0]['maximumConcentration']['value']

ce_ref = json_battery['electrolyte']['initialConcentration']['value']

V_a=json_battery['negativeElectrode']['thickness']['value']*json_battery['negativeElectrode']['area']['value']
V_c=json_battery['positiveElectrode']['thickness']['value']*json_battery['positiveElectrode']['area']['value']

Q_bat = json_battery['properties']['nominalCapacity']['value']

nx_n = output_profile['cs_sur_n'].shape[0]
nx_p = output_profile['cs_sur_p'].shape[0]
nx   = output_profile['ce'].shape[0]

# ==========================================================
# IO UTILITIES
# ==========================================================

def read_pvd(pvd_path):
    root = ET.parse(pvd_path).getroot()
    return [{"time": float(ds.attrib["timestep"]),
             "file": Path(pvd_path).parent / ds.attrib["file"]}
            for ds in root.iter("DataSet")]

def select_time(pvd_data, t_target):
    times = np.array([d["time"] for d in pvd_data])
    return pvd_data[np.argmin(np.abs(times - t_target))]

def load_1d_vtu(vtu_path):
    mesh = meshio.read(vtu_path)
    x = mesh.points[:, 0]
    u = next(iter(mesh.point_data.values()))
    idx = np.argsort(x)
    return x[idx], u[idx]

def load_field(results_dir, field, t_target, factors=(1,1,1), domains=("a","s","c")):
    xs, us = [], []

    for i, d in enumerate(domains):
        pvd = results_dir / field / f"{field}_{d}.pvd"

        if not pvd.exists():
            xs.append(np.linspace(i, i+1, 21))
            us.append(np.zeros(21))
            continue

        entry = select_time(read_pvd(pvd), t_target)
        x, u = load_1d_vtu(entry["file"])
        xs.append(x + i)
        us.append(u * factors[i])

    return xs, us

# ==========================================================
# GENERIC SPATIAL COMPARISON
# ==========================================================

def plot_spatial_comparison(field, ylabel,
                            pybamm_keys=None,
                            factors=(1,1,1),
                            mode="split",
                            combine=None):

    plt.figure()
    cmap = plt.get_cmap("tab10")

    for i_time, t in enumerate(target_times):
        color = cmap(i_time % cmap.N)

        # -------- FEM ----------
        xs, us = load_field(results_dir, field, t, factors)

        for xd, ud in zip(xs, us):
            plt.plot(xd, ud, '-', color=color,
                     label=f"t={t:.0f}s" if xd is xs[0] else None)

        # -------- PyBaMM ----------
        idx = np.argmin(np.abs(time_pybamm - t))

        if mode == "full":
            # single domain [0,3]
            x_dom = np.linspace(0, 3,
                                    output_profile[pybamm_keys].shape[0])
            data = output_profile[pybamm_keys][:, idx]

            data *= factors[0]  # assume same factor for all domains
            plt.plot(x_dom, data, '--', color=color)

        elif mode == "split":
            for i_dom, key in enumerate(pybamm_keys):
                if key is None:
                    continue

                x_dom = np.linspace(i_dom, i_dom+1,
                                    output_profile[key].shape[0])

                data = output_profile[key][:, idx]

                if combine is not None:
                    data = combine(i_dom, data, idx)

                data *= factors[i_dom]

                plt.plot(x_dom, data, '--', color=color)

    plt.xlabel("Cell thickness [-]")
    plt.ylabel(ylabel)
    plt.title(f"{field} — {battery_name}")
    plt.grid(True)
    plt.legend(fontsize="small")

# ==========================================================
# FIELD DEFINITIONS
# ==========================================================

plot_spatial_comparison(
    field="c_e",
    ylabel=r"$c_e/c_{e,ref}$ [-]",
    pybamm_keys="ce",
    factors=(1/ce_ref,1/ce_ref,1/ce_ref),
    mode="full",
)

plot_spatial_comparison(
    field="j_tot",
    ylabel=r"$V\cdot j_{tot}$ [A]",
    pybamm_keys=["ji_n", None, "ji_p"],
    factors=(V_a, 1, V_c)
)

plot_spatial_comparison(
    field="c_s_sur",
    ylabel=r"$c_{s,sur}/c_{s,max}$ [-]",
    pybamm_keys=["cs_sur_n", None, "cs_sur_p"],
    factors=(1/cs_max_a, 1, 1/cs_max_c)
)

plot_spatial_comparison(
    field="c_s_avg",
    ylabel=r"$c_{s,avg}/c_{s,max}$ [-]",
    pybamm_keys=["cs_avg_n", None, "cs_avg_p"],
    factors=(1/cs_max_a, 1, 1/cs_max_c)
)

plot_spatial_comparison(
    field="phi_e",
    ylabel=r"$\phi_e$ [V]",
    pybamm_keys=["phie_n", "phie_s", "phie_p"]
)

plot_spatial_comparison(
    field="phi_s",
    ylabel=r"$\phi_s$ [V]",
    pybamm_keys=["phise_n", None, "phise_p"],
    combine=lambda i_dom, data, idx:
        data + output_profile[f"phie_{['n','s','p'][i_dom]}"][:, idx]
        if i_dom in (0,2) else data
)

# ==========================================================
# GLOBAL COMPARISON
# ==========================================================

time_batfem = np.loadtxt(results_dir / "time.txt")
voltage_batfem = np.loadtxt(results_dir / "voltage.txt")
current_batfem = np.loadtxt(results_dir / "current.txt")

plt.figure()
plt.plot(time_batfem, voltage_batfem, label="BatFEM")
plt.plot(time_pybamm, input_profile["voltage"], '--', label="PyBaMM")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.title(f"Voltage — {battery_name}")
plt.grid(True)

plt.figure()
plt.plot(time_pybamm,
         1e3*np.abs(input_profile["voltage"] - voltage_batfem),
         '--')
plt.xlabel("Time [s]")
plt.ylabel("Error [mV]")
plt.title(f"Voltage Error — {battery_name}")
plt.grid(True)

plt.figure()
plt.plot(time_batfem, current_batfem/Q_bat, label="BatFEM")
plt.plot(time_pybamm, -input_profile["current"]/Q_bat, '--', label="PyBaMM")
plt.xlabel("Time [s]")
plt.ylabel("C-rate [1/h]")
plt.legend()
plt.title(f"C-rate — {battery_name}")
plt.grid(True)

plt.figure()
plt.plot(time_pybamm, -input_profile["current"]/Q_bat, )
plt.xlabel("Time [s]")
plt.ylabel("C-rate [1/h]")
plt.title(f"UDDS Profile")
plt.grid(True)

plt.show()