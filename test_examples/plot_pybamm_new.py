import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

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

def select_time(pvd_data, t_target):
    times = np.array([d["time"] for d in pvd_data])
    idx = np.argmin(np.abs(times - t_target))
    return pvd_data[idx]

import meshio

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


def load_field_at_time(results_dir, field, t_target, domains=("a", "s", "c")):
    results_dir = Path(results_dir)
    field_dir = results_dir / field

    xs, us = [], []

    i=0
    for d in domains:
        pvd = field_dir / f"{field}_{d}.pvd"

        # domain does not exist (e.g. separator for phi_s)
        if not pvd.exists():
            xs.append(np.linspace(1, 2, 21))
            us.append(np.zeros(21))
            i+=1
            continue

        pvd_data = read_pvd(pvd)
        entry = select_time(pvd_data, t_target)

        x, u = load_1d_vtu(entry["file"])
        xs.append(x+i)
        us.append(u)
        i+=1
        
    if not xs:
        raise RuntimeError(f"No data found for field '{field}'")

    x = np.concatenate(xs)
    u = np.concatenate(us)
    
    idx = np.argsort(x)
    return x[idx], u[idx]


input_profile_pybamm = np.load("test_examples/Chen2020/CCCV_input_profile_p2d.npz")
output_profile_pybamm = np.load("test_examples/Chen2020/CCCV_output_profile_p2d.npz")

time_pybamm = input_profile_pybamm['time']
target_times = [0,1800,3600,5400,7200,9000,10800]
target_times = [0,600,1200,1800,2400]
nx_n=len(output_profile_pybamm['cs_sur_n'][:, 0])
nx_p=len(output_profile_pybamm['cs_sur_p'][:, 0])
nx=len(output_profile_pybamm['ce'][:, 0])

print(output_profile_pybamm['x'])
#print(output_profile['x_p'])
#print(output_profile['cs_sur_p'])
#print(output_profile['cs_sur_n'])

plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)

for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    # FEM line with label
    x, ce = load_field_at_time("results/Chen2020/", field="c_e", t_target=target_time)
    plt.plot(x, ce, '-', color=color, label=f"t={target_time/3600:.1f}h")

    # PyBaMM dashed lines without label
    idx = np.argmin(np.abs(time_pybamm - target_time))
    plt.plot(np.linspace(0, 3, nx), output_profile_pybamm['ce'][:, idx], '--', color=color)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$c_{e}$ [mol/m$^3$]")
plt.title("Electrolyte concentration")
plt.grid(True)
plt.legend(loc='best', fontsize='small', ncol=1)

plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)

for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    # FEM line with label
    x, cs_sur = load_field_at_time("results/Chen2020/", field="c_s_sur", t_target=target_time)
    plt.plot(x, cs_sur, '-', color=color, label=f"t={target_time/3600:.1f}h")

    # PyBaMM dashed lines without label
    idx = np.argmin(np.abs(time_pybamm - target_time))
    plt.plot(0 + np.linspace(0, 1, nx_n), output_profile_pybamm['cs_sur_n'][:, idx], '--', color=color)
    plt.plot(3 - np.linspace(0, 1, nx_p), output_profile_pybamm['cs_sur_p'][:, idx], '--', color=color)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$c_{s, sur}$ [mol/m$^3$]")
plt.title("Electrode surface concentration")
plt.grid(True)
plt.legend(loc='best', fontsize='small', ncol=1)

plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)

for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    # FEM line with label
    x, cs_sur = load_field_at_time("results/Chen2020/", field="c_s_avg", t_target=target_time)
    plt.plot(x, cs_sur, '-', color=color, label=f"t={target_time/3600:.1f}h")

    # PyBaMM dashed lines without label
    idx = np.argmin(np.abs(time_pybamm - target_time))
    plt.plot(0 + np.linspace(0, 1, nx_n), output_profile_pybamm['cs_avg_n'][:, idx], '--', color=color)
    plt.plot(3 - np.linspace(0, 1, nx_p), output_profile_pybamm['cs_avg_p'][:, idx], '--', color=color)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$c_{s, avg}$ [mol/m$^3$]")
plt.title("Electrode average concentration")
plt.grid(True)
plt.legend(loc='best', fontsize='small', ncol=1)

plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)

for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    # FEM line with label
    x, phi_e = load_field_at_time("results/Chen2020/", field="phi_e", t_target=target_time)
    x, phi_s = load_field_at_time("results/Chen2020/", field="phi_s", t_target=target_time)
    phi_se=phi_s - phi_e
    plt.plot(x, phi_se, '-', color=color, label=f"t={target_time/3600:.1f}h")

    # PyBaMM dashed lines without label
    idx = np.argmin(np.abs(time_pybamm - target_time))
    plt.plot(0 + np.linspace(0, 1, nx_n), output_profile_pybamm['phise_n'][:, idx], '--', color=color)
    plt.plot(3 - np.linspace(0, 1, nx_p), output_profile_pybamm['phise_p'][:, idx], '--', color=color)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$\phi_{s} - \phi_{e}$ [V]")
plt.title("Electrolyte potential difference")
plt.grid(True)
plt.legend(loc='best', fontsize='small', ncol=1)



time_batfem = np.loadtxt("results/Chen2020/time.txt")
voltage_batfem = np.loadtxt("results/Chen2020/voltage.txt") 
current_batfem = np.loadtxt("results/Chen2020/current.txt")

plt.figure()
plt.plot(time_pybamm, input_profile_pybamm['voltage'], 'b--',label='PyBaMM')
plt.plot(time_batfem, voltage_batfem, 'b-',label='batFEM')
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()


plt.figure()
plt.plot(time_pybamm, -input_profile_pybamm['current'], 'b--',label='PyBaMM')
plt.plot(time_batfem, current_batfem, 'b-',label='batFEM')
plt.xlabel("Time [s]")
plt.ylabel("Current [A]")
plt.legend()
plt.show()