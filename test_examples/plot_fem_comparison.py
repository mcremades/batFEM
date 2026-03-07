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

battery_name = "Prada2013"
base_path = Path("results")

cases = {
    "50P2": {"folder": f"{battery_name}_FEM50P2_5C", "style": "None", "marker":".", "color": "black"},
    "20P2":  {"folder": f"{battery_name}_FEM20P2_5C",  "style": "None",  "marker":"*", "color": "orange"},
    "10P2":  {"folder": f"{battery_name}_FEM10P2_5C",  "style": "None", "marker":"v", "color": "red"},
    "5P2":   {"folder": f"{battery_name}_FEM5P2_5C",   "style": "None", "marker":"^", "color": "blue"},
    "50P1": {"folder": f"{battery_name}_FEM50P1_5C", "style": "-",  "marker":"", "color": "black"},
    "20P1":  {"folder": f"{battery_name}_FEM20P1_5C",  "style": ":",  "marker":"", "color": "orange"},
    "10P1":  {"folder": f"{battery_name}_FEM10P1_5C",  "style": "-.", "marker":"", "color": "red"},
    "5P1":   {"folder": f"{battery_name}_FEM5P1_5C",   "style": "--", "marker":"", "color": "blue"},
}

cases = {
    "50P2": {"folder": f"{battery_name}_FEM50P2_5C", "style": "-", "marker":"", "color": "black"},
    "50P1": {"folder": f"{battery_name}_FEM50P1_5C", "style": "-",  "marker":"", "color": "green"},
    "20P1":  {"folder": f"{battery_name}_FEM20P1_5C",  "style": "--",  "marker":"", "color": "orange"},
    "10P1":  {"folder": f"{battery_name}_FEM10P1_5C",  "style": "-.", "marker":"", "color": "red"},
    "5P1":   {"folder": f"{battery_name}_FEM5P1_5C",   "style": ":", "marker":"", "color": "blue"},
}

reference_case = "50P2"
target_times = np.linspace(0, 2600, 14)

json_path = "json_battery/cells/cell_"+battery_name+".json"

with open(json_path) as data_file:
    json_file = json.load(data_file)
json_battery = batFEM.class_battery.parse_json(json_file); 

cs_max_a=json_battery['negativeElectrode']['composition'][0]['maximumConcentration']['value']
cs_max_c=json_battery['positiveElectrode']['composition'][0]['maximumConcentration']['value']

ce_ref = json_battery['electrolyte']['initialConcentration']['value']
De= json_battery['electrolyte']['diffusionConstant']
L_a=json_battery['negativeElectrode']['thickness']['value']
L_s=json_battery['separator']['thickness']['value']
L_c=json_battery['positiveElectrode']['thickness']['value']

epse_a=json_battery['negativeElectrode']['porosity']['value']
epse_s=json_battery['separator']['porosity']['value']
epse_c=json_battery['positiveElectrode']['porosity']['value']

exp = lambda x: np.exp(x)
c_e = ce_ref
T = 298.15

if De["type"] == "function":
    De_eff_a=epse_a**1.5*eval(De["value"])
    De_eff_s=epse_s**1.5*eval(De["value"])
    De_eff_c=epse_c**1.5*eval(De["value"])
else:
    De_eff_a=epse_a**1.5*De["value"]
    De_eff_s=epse_s**1.5*De["value"]
    De_eff_c=epse_c**1.5*De["value"]



for C_rate in [1,2,3,4,5]:
    t_c=3600/C_rate
    print(f"C-rate: {C_rate}C")
    
    print('Da_a:',L_a**2/(t_c*De_eff_a))
    print('Da_s:',L_s**2/(t_c*De_eff_s))
    print('Da_c:',L_c**2/(t_c*De_eff_c))

V_a=L_a*json_battery['negativeElectrode']['area']['value']
V_c=L_c*json_battery['positiveElectrode']['area']['value']

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

# ==========================================================
# VOLTAGE PLOT
# ==========================================================

def plot_voltage():
    plt.figure()

    for name, data in data_cases.items():
        if name == reference_case:
            plt.plot(times, data["voltage"],
                     ls=cases[name]["style"],
                     marker=cases[name]["marker"],
                     color=cases[name]["color"],
                     label=f"Nx={name[:-2]}, {name[-2:]}")
        else:
            v_interp = np.interp(times, data["time"], data["voltage"])
            plt.plot(times, v_interp,
                     ls=cases[name]["style"],
                     marker=cases[name]["marker"],
                     color=cases[name]["color"],
                     label=f"Nx={name[:-2]}, {name[-2:]}")

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(battery_name)
    plt.title(f"Voltage — {battery_name}")
    plt.grid(True)
    plt.legend()


def plot_voltage_error():
    plt.figure()

    for name, data in data_cases.items():
        if name == reference_case:
            continue

        v_interp = np.interp(times, data["time"], data["voltage"])
        error = 1000*np.abs(ref["voltage"] - v_interp)

        plt.plot(times, error,
                 ls=cases[name]["style"],
                 marker=cases[name]["marker"],
                 color=cases[name]["color"],
                 label=f"Nx={name[:-2]}, {name[-2:]}")

        print(f"RMSE Voltage {name}: {rmse(ref['voltage'], v_interp):.6e}")

    plt.xlabel("Time [s]")
    plt.ylabel("Error [mV]")
    plt.title(f"Voltage Error — {battery_name}")
    plt.grid(True)
    plt.legend()

# ==========================================================
# SPATIAL FIELD COMPARISON
# ==========================================================

def load_field(case_name, field, t_target, factors=[1,1,1]):
    folder = base_path / cases[case_name]["folder"]
    return load_field_at_time(folder, field=field, t_target=t_target,factors=factors)

def load_field_at_time(results_dir, field, t_target, domains=("a","s","c"),factors=[1,1,1]):
    results_dir = Path(results_dir)
    field_dir = results_dir / field

    xs, us = [], []

    i=0
    for d in domains:
        pvd = field_dir / f"{field}_{d}.pvd"

        # domain does not exist (e.g. separator for phi_s)
        if not pvd.exists():
            xs.append(np.linspace(1.1, 1.9, 9))
            us.append(np.zeros(9))
            i+=1
            continue

        pvd_data = read_pvd(pvd)
        entry = select_time(pvd_data, t_target)

        x, u = load_1d_vtu(entry["file"])
        xs.append(x+i)
        us.append(u*factors[i])
        i+=1
        
    if not xs:
        raise RuntimeError(f"No data found for field '{field}'")

    #x = np.concatenate(xs)
    #u = np.concatenate(us)
    
    #idx = np.argsort(x)
    #return x[idx], u[idx]
    return xs, us

def plot_spatial_field(field, ylabel, factors=[1,1,1]):
    plt.figure()
    cmap = plt.get_cmap("tab10")

    for i_time, t in enumerate(target_times):
        color = cmap(i_time % cmap.N)

        xs_ref, us_ref = load_field(reference_case, field, t, factors)

        # Plot reference domain by domain
        for xd, ud in zip(xs_ref, us_ref):
            plt.plot(xd, ud, color=color,
                     label=f"{t:.0f} s" if xd is xs_ref[0] else None)

        for name in cases:
            if name == reference_case:
                continue

            xs, us = load_field(name, field, t, factors)

            # Interpolate domain-by-domain
            for xd_ref, xd, ud in zip(xs_ref, xs, us):
                u_interp = np.interp(xd_ref, xd, ud)
                plt.plot(xd, ud,
                         ls=cases[name]["style"],
                         marker=cases[name]["marker"],
                         color=color)

    plt.xlabel("Cell thickness [-]")
    plt.ylabel(ylabel)
    plt.title(f"{field} — {battery_name}")
    plt.grid(True)
    plt.legend()

def read_pvd(pvd_path):
    pvd_path = Path(pvd_path)
    tree = ET.parse(pvd_path)
    root = tree.getroot()

    data = []
    for ds in root.iter("DataSet"):
        data.append({
            "time": float(ds.attrib["timestep"]),
            "file": pvd_path.parent / ds.attrib["file"]
        })

    return data

def plot_spatial_error(field, ylabel, factors=[1,1,1]):
    plt.figure()
    cmap = plt.get_cmap("tab10")

    error_accumulator = {name: [] for name in cases if name != reference_case}

    for i_time, t in enumerate(target_times):
        color = cmap(i_time % cmap.N)

        xs_ref, us_ref = load_field(reference_case, field, t, factors)

        for name in cases:
            if name == reference_case:
                continue

            xs, us = load_field(name, field, t, factors)

            for xd_ref, ud_ref, xd, ud in zip(xs_ref, us_ref, xs, us):
                #u_interp = np.interp(xd_ref, xd, ud)
                #error = ud_ref - u_interp
                u_interp = np.interp(xd, xd_ref, ud_ref)
                error = ud - u_interp

                error_accumulator[name].append(error)

                plt.plot(xd, np.abs(error),
                         ls=cases[name]["style"],
                         marker=cases[name]["marker"],
                         color=color)

    # Total RMSE
    for name, error_list in error_accumulator.items():
        error_all = np.concatenate(error_list)
        total_rmse = np.linalg.norm(error_all) / np.sqrt(len(error_all))
        print(f"TOTAL RMSE {field} {name}: {total_rmse:.6e}")

    plt.xlabel("Cell thickness [-]")
    plt.ylabel(ylabel)
    plt.title(f"{field} Error — {battery_name}")
    plt.grid(True)

def select_time(pvd_data, t_target):
    times = np.array([d["time"] for d in pvd_data])
    idx = np.argmin(np.abs(times - t_target))
    return pvd_data[idx]

def load_1d_vtu(vtu_path):
    mesh = meshio.read(vtu_path)

    x = mesh.points[:, 0]

    if len(mesh.point_data) != 1:
        raise ValueError(
            f"Expected 1 field in {vtu_path}, found {list(mesh.point_data.keys())}"
        )

    u = next(iter(mesh.point_data.values()))

    idx = np.argsort(x)
    return x[idx], u[idx]

# ==========================================================
# RUN PLOTS
# ==========================================================

plot_voltage()
plot_voltage_error()

plot_spatial_field("c_e", r"$c_e/c_{e,ref}$ [-]",factors=[1/ce_ref,1/ce_ref,1/ce_ref])
plot_spatial_error("c_e", "Error [-]",factors=[1/ce_ref,1/ce_ref,1/ce_ref])

plot_spatial_field("c_s_sur", r"$c_{s,sur}/c_{s,max}$ [-]",factors=[1/cs_max_a,1,1/cs_max_c])
plot_spatial_error("c_s_sur", "Error [-]",factors=[1/cs_max_a,1,1/cs_max_c])

plot_spatial_field("c_s_avg", r"$c_{s,avg}/c_{s,max}$ [-]",factors=[1/cs_max_a,1,1/cs_max_c])
plot_spatial_error("c_s_avg", "Error [-]",factors=[1/cs_max_a,1,1/cs_max_c])

plot_spatial_field("phi_e", r"$\phi_e$ [V]")
plot_spatial_error("phi_e", "Error [V]")

plot_spatial_field("phi_s", r"$\phi_s$ [V]")
plot_spatial_error("phi_s", "Error [V]")


plot_spatial_field("j_tot", r"$V\cdot j_{tot}$ [A]",factors=[V_a,1,V_c])
plot_spatial_error("j_tot", "Error [A]",factors=[V_a,1,V_c])

plt.show()