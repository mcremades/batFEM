import sys; sys.path.insert(0,'..');
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

import batFEM.class_battery

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

    x = np.concatenate(xs)
    u = np.concatenate(us)
    
    idx = np.argsort(x)
    return x[idx], u[idx]

#parser = argparse.ArgumentParser()

#parser.add_argument('--plot_paths', nargs='+', help="input path with results to compare")

#for _, value in parser.parse_args()._get_kwargs():
#    if value is not None:
#        print(value)

import matplotlib.pyplot as plt

import json 

battery_name="Marquis2019"

json_path = "json_battery/cells/cell_"+battery_name+".json"

with open(json_path) as data_file:
    json_file = json.load(data_file)
json_battery = batFEM.class_battery.parse_json(json_file); 



target_times=np.linspace(0,2600,14)

cs_max_a=json_battery['negativeElectrode']['composition'][0]['maximumConcentration']['value']
cs_max_c=json_battery['positiveElectrode']['composition'][0]['maximumConcentration']['value']

V_a=json_battery['negativeElectrode']['thickness']['value']*json_battery['negativeElectrode']['area']['value']
V_c=json_battery['positiveElectrode']['thickness']['value']*json_battery['positiveElectrode']['area']['value']


times=np.loadtxt("results/"+battery_name+"_FEM100P1_5C/time.txt")
voltage=np.loadtxt("results/"+battery_name+"_FEM100P1_5C/voltage.txt")
xs_avg_a=np.loadtxt("results/"+battery_name+"_FEM100P1_5C/xs_avg_a.txt")
xs_avg_c=np.loadtxt("results/"+battery_name+"_FEM100P1_5C/xs_avg_c.txt")
xs_sur_a=np.loadtxt("results/"+battery_name+"_FEM100P1_5C/xs_sur_a.txt")
xs_sur_c=np.loadtxt("results/"+battery_name+"_FEM100P1_5C/xs_sur_c.txt")

times_5P1=np.loadtxt("results/"+battery_name+"_FEM5P1_5C/time.txt")
voltage_5P1=np.interp(times,times_5P1,np.loadtxt("results/"+battery_name+"_FEM5P1_5C/voltage.txt"))
xs_avg_a_5P1=np.interp(times,times_5P1,np.loadtxt("results/"+battery_name+"_FEM5P1_5C/xs_avg_a.txt"))
xs_avg_c_5P1=np.interp(times,times_5P1,np.loadtxt("results/"+battery_name+"_FEM5P1_5C/xs_avg_c.txt"))
xs_sur_a_5P1=np.interp(times,times_5P1,np.loadtxt("results/"+battery_name+"_FEM5P1_5C/xs_sur_a.txt"))
xs_sur_c_5P1=np.interp(times,times_5P1,np.loadtxt("results/"+battery_name+"_FEM5P1_5C/xs_sur_c.txt"))

times_10P1=np.loadtxt("results/"+battery_name+"_FEM10P1_5C/time.txt")
voltage_10P1=np.interp(times,times_10P1,np.loadtxt("results/"+battery_name+"_FEM10P1_5C/voltage.txt"))
xs_avg_a_10P1=np.interp(times,times_10P1,np.loadtxt("results/"+battery_name+"_FEM10P1_5C/xs_avg_a.txt"))
xs_avg_c_10P1=np.interp(times,times_10P1,np.loadtxt("results/"+battery_name+"_FEM10P1_5C/xs_avg_c.txt"))
xs_sur_a_10P1=np.interp(times,times_10P1,np.loadtxt("results/"+battery_name+"_FEM10P1_5C/xs_sur_a.txt"))
xs_sur_c_10P1=np.interp(times,times_10P1,np.loadtxt("results/"+battery_name+"_FEM10P1_5C/xs_sur_c.txt"))

times_50P1=np.loadtxt("results/"+battery_name+"_FEM50P1_5C/time.txt")
voltage_50P1=np.interp(times,times_50P1,np.loadtxt("results/"+battery_name+"_FEM50P1_5C/voltage.txt"))
xs_avg_a_50P1=np.interp(times,times_50P1,np.loadtxt("results/"+battery_name+"_FEM50P1_5C/xs_avg_a.txt"))
xs_avg_c_50P1=np.interp(times,times_50P1,np.loadtxt("results/"+battery_name+"_FEM50P1_5C/xs_avg_c.txt"))
xs_sur_a_50P1=np.interp(times,times_50P1,np.loadtxt("results/"+battery_name+"_FEM50P1_5C/xs_sur_a.txt"))
xs_sur_c_50P1=np.interp(times,times_50P1,np.loadtxt("results/"+battery_name+"_FEM50P1_5C/xs_sur_c.txt"))

times_5=np.loadtxt("results/"+battery_name+"_FEM20P2_5C/time.txt")
voltage_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_FEM20P2_5C/voltage.txt"))
xs_avg_a_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_FEM20P2_5C/xs_avg_a.txt"))
xs_avg_c_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_FEM20P2_5C/xs_avg_c.txt"))
xs_sur_a_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_FEM20P2_5C/xs_sur_a.txt"))
xs_sur_c_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_FEM20P2_5C/xs_sur_c.txt"))


plt.figure()
plt.plot(times,voltage,color='black',label=r'$N_x=100, P1$')
plt.plot(times,voltage_5P1,'--',color='blue',label=r'$N_x=5, P1$')
plt.plot(times,voltage_10P1,'--',color='red',label=r'$N_x=10, P1$')
plt.plot(times,voltage_50P1,'--',color='orange',label=r'$N_x=50, P1$')
#plt.plot(times,voltage_5,'--',color='green',label=r'$N_x=20, P2$')
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

plt.figure()
plt.plot(times,abs(voltage-voltage_5P1),'--',color='blue',label=r'$N_x=5, P1$')
plt.plot(times,abs(voltage-voltage_10P1),'--',color='red',label=r'$N_x=10, P1$')
plt.plot(times,abs(voltage-voltage_50P1),'--',color='orange',label=r'$N_x=50, P1$')
#plt.plot(times,abs(voltage-voltage_5),'--',color='green',label=r'$N_x=20, P2$')
plt.xlabel("Time [s]")
plt.ylabel("Error [V]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

print('RMSE Voltage 5P1',np.linalg.norm(1000*(voltage - voltage_5P1)) / np.sqrt(len(voltage)))
print('RMSE Voltage 10P1',np.linalg.norm(1000*(voltage - voltage_10P1)) / np.sqrt(len(voltage)))
print('RMSE Voltage 50P1',np.linalg.norm(1000*(voltage - voltage_50P1)) / np.sqrt(len(voltage)))
#print('RMSE Voltage 20P2',np.linalg.norm(1000*(voltage - voltage_5)) / np.sqrt(len(voltage)))
plt.figure()
plt.plot(times,xs_sur_a,color='black',label=r'$N_x=100, P1$')
plt.plot(times,xs_avg_a,color='black')
plt.plot(times,xs_sur_a_5P1,'--',color='blue',label=r'$N_x=5, P1$')
plt.plot(times,xs_avg_a_5P1,'--',color='blue')
plt.plot(times,xs_sur_a_10P1,'--',color='red',label=r'$N_x=10, P1$')
plt.plot(times,xs_avg_a_10P1,'--',color='red')
plt.plot(times,xs_sur_a_50P1,'--',color='orange',label=r'$N_x=50, P1$')
plt.plot(times,xs_avg_a_50P1,'--',color='orange')
plt.xlabel("Time [s]")
plt.ylabel(r"$x_{s,sur}^-$ [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)



plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x, ce = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, color=color,label=str(target_time)+" s")
    x, ce = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '--',color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM20P1_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '-.', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM10P2_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, ':', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '.', color=color)
    

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$x_{s,sur}$ [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)


plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x, ce = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="c_e", t_target=target_time)
    plt.plot(x, ce, color=color,label=str(target_time)+" s")
    x, ce = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="c_e", t_target=target_time)
    plt.plot(x, ce, '--',color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM20P1_5C/", field="c_e", t_target=target_time)
    plt.plot(x, ce, '-.', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM10P2_5C/", field="c_e", t_target=target_time)
    plt.plot(x, ce, ':', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="c_e", t_target=target_time)
    plt.plot(x, ce, '.', color=color)
    

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$c_{e}$ [mol/m$^3$]")
plt.title(battery_name)
plt.legend()
plt.grid(1)



plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x_100, ce_100P1 = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="c_e", t_target=target_time)
    x, ce_1 = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="c_e", t_target=target_time)
    ce_10P1 = np.interp(x_100,x,ce_1)
    x, ce_2 = load_field_at_time("results/"+battery_name+"_FEM20P1_5C/", field="c_e", t_target=target_time)
    ce_20P1 = np.interp(x_100,x,ce_2)
    x, ce_3 = load_field_at_time("results/"+battery_name+"_FEM10P2_5C/", field="c_e", t_target=target_time)
    ce_10P2 = np.interp(x_100,x,ce_3)
    x, ce_4 = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="c_e", t_target=target_time)
    ce_20P2 = np.interp(x_100,x,ce_4)

    #plt.plot(x_100, abs(ce_100P1-ce_20P2), '.', color=color,label=str(target_time)+" s")
    #plt.plot(x_100, abs(ce_100P1-ce_10P2), ':', color=color)
    plt.plot(x_100, abs(ce_100P1-ce_20P1), '-.', color=color)
    plt.plot(x_100, abs(ce_100P1-ce_10P1), '--',color=color)

RMSE_ce_10P1=np.linalg.norm((ce_100P1 - ce_10P1)) / np.sqrt(len(ce_100P1))
RMSE_ce_20P1=np.linalg.norm((ce_100P1 - ce_20P1)) / np.sqrt(len(ce_100P1))
#RMSE_ce_10P2=np.linalg.norm((ce_100P1 - ce_10P2)) / np.sqrt(len(ce_100P1))
#RMSE_ce_20P2=np.linalg.norm((ce_100P1 - ce_20P2)) / np.sqrt(len(ce_100P1))

print('RMSE_ce_10P1',RMSE_ce_10P1)
print('RMSE_ce_20P1',RMSE_ce_20P1)
#print('RMSE_ce_10P2',RMSE_ce_10P2)
#print('RMSE_ce_20P2',RMSE_ce_20P2)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"Error [mol/m$^3$]")
plt.title(battery_name)
plt.legend()
plt.grid(1)


plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x, ce = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="phi_e", t_target=target_time)
    plt.plot(x, ce, color=color,label=str(target_time)+" s")
    x, ce = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="phi_e", t_target=target_time)
    plt.plot(x, ce, '--',color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM20P1_5C/", field="phi_e", t_target=target_time)
    plt.plot(x, ce, '-.', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM10P2_5C/", field="phi_e", t_target=target_time)
    plt.plot(x, ce, ':', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="phi_e", t_target=target_time)
    plt.plot(x, ce, '.', color=color)
    

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$\phi_{e}$ [V]")
plt.title(battery_name)
plt.legend()
plt.grid(1)



plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x_100, ce_100P1 = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="phi_e", t_target=target_time)
    x, ce_1 = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="phi_e", t_target=target_time)
    ce_10P1 = np.interp(x_100,x,ce_1)
    x, ce_2 = load_field_at_time("results/"+battery_name+"_FEM20P1_5C/", field="phi_e", t_target=target_time)
    ce_20P1 = np.interp(x_100,x,ce_2)
    x, ce_3 = load_field_at_time("results/"+battery_name+"_FEM10P2_5C/", field="phi_e", t_target=target_time)
    ce_10P2 = np.interp(x_100,x,ce_3)
    x, ce_4 = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="phi_e", t_target=target_time)
    ce_20P2 = np.interp(x_100,x,ce_4)

    #plt.plot(x_100, abs(ce_100P1-ce_20P2), '.', color=color,label=str(target_time)+" s")
    #plt.plot(x_100, abs(ce_100P1-ce_10P2), ':', color=color)
    plt.plot(x_100, abs(ce_100P1-ce_20P1), '-.', color=color)
    plt.plot(x_100, abs(ce_100P1-ce_10P1), '--',color=color)

RMSE_ce_10P1=np.linalg.norm((ce_100P1 - ce_10P1)) / np.sqrt(len(ce_100P1))
RMSE_ce_20P1=np.linalg.norm((ce_100P1 - ce_20P1)) / np.sqrt(len(ce_100P1))
#RMSE_ce_10P2=np.linalg.norm((ce_100P1 - ce_10P2)) / np.sqrt(len(ce_100P1))
#RMSE_ce_20P2=np.linalg.norm((ce_100P1 - ce_20P2)) / np.sqrt(len(ce_100P1))

print('RMSE_phie_10P1',RMSE_ce_10P1)
print('RMSE_phie_20P1',RMSE_ce_20P1)
#print('RMSE_phie_10P2',RMSE_ce_10P2)
#print('RMSE_phie_20P2',RMSE_ce_20P2)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"Error [V]")
plt.title(battery_name)
plt.legend()
plt.grid(1)


plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)


    x_100, jtot_100 = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    mask_left  = x_100 <= 1.0
    mask_right = x_100 >= 2.0

    plt.plot(x_100[mask_left], jtot_100[mask_left], color=color,label=str(target_time)+" s")
    plt.plot(x_100[mask_right], jtot_100[mask_right], color=color)
    
    
    
    x, jtot_5 = load_field_at_time("results/"+battery_name+"_FEM5P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    jtot_5=np.interp(x_100,x,jtot_5)
    plt.plot(x_100[mask_left], jtot_5[mask_left], '--',color=color)
    plt.plot(x_100[mask_right], jtot_5[mask_right], '--',color=color)
    x, jtot_10 = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    jtot_10=np.interp(x_100,x,jtot_10)
    plt.plot(x_100[mask_left], jtot_10[mask_left], '-.', color=color)
    plt.plot(x_100[mask_right], jtot_10[mask_right], '-.', color=color)
    x, jtot_50 = load_field_at_time("results/"+battery_name+"_FEM50P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    jtot_50=np.interp(x_100,x,jtot_50)
    plt.plot(x_100[mask_left], jtot_50[mask_left], ':', color=color)
    plt.plot(x_100[mask_right], jtot_50[mask_right], ':', color=color)

    #x, ce = load_field_at_time("results/"+battery_name+"_FEM10P2_5C/", field="j_tot", t_target=target_time)
    #plt.plot(x, ce, ':', color=color)
    #x, ce = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="j_tot", t_target=target_time)
    #plt.plot(x, ce, '.', color=color)
    

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"V \cdot $j_{tot}$ [A]")
plt.title(battery_name)
plt.legend()
plt.grid(1)



plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x_100, ce_100P1 = load_field_at_time("results/"+battery_name+"_FEM100P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    mask_left  = x_100 <= 1.0
    mask_right = x_100 >= 2.0

    x, ce_5P1 = load_field_at_time("results/"+battery_name+"_FEM5P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    ce_5P1 = np.interp(x_100,x,ce_5P1)
    x, ce_10P1 = load_field_at_time("results/"+battery_name+"_FEM10P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    ce_10P1 = np.interp(x_100,x,ce_10P1)
    x, ce_50P1 = load_field_at_time("results/"+battery_name+"_FEM50P1_5C/", field="j_tot", t_target=target_time,factors=[V_a,1,V_c])
    ce_50P1 = np.interp(x_100,x,ce_50P1)
    #x, ce_4 = load_field_at_time("results/"+battery_name+"_FEM20P2_5C/", field="j_tot", t_target=target_time)
    #ce_20P2 = np.interp(x_100,x,ce_4)

    #plt.plot(x_100, abs(ce_100P1-ce_20P2), '.', color=color,label=str(target_time)+" s")
    #plt.plot(x_100, abs(ce_100P1-ce_10P2), ':', color=color)
    #plt.plot(x_100, abs(ce_100P1-ce_20P1), '-.', color=color)
    plt.plot(x_100[mask_left], abs(ce_100P1[mask_left]-ce_50P1[mask_left]), ':',color=color)
    plt.plot(x_100[mask_right], abs(ce_100P1[mask_right]-ce_50P1[mask_right]), ':',color=color)
    
    plt.plot(x_100[mask_left], abs(ce_100P1[mask_left]-ce_10P1[mask_left]), '-.',color=color)
    plt.plot(x_100[mask_right], abs(ce_100P1[mask_right]-ce_10P1[mask_right]), '-.',color=color)
    plt.plot(x_100[mask_left], abs(ce_100P1[mask_left]-ce_5P1[mask_left]), '--',color=color)
    plt.plot(x_100[mask_right], abs(ce_100P1[mask_right]-ce_5P1[mask_right]), '--',color=color)

RMSE_ce_5P1=np.linalg.norm((ce_100P1 - ce_5P1)) / np.sqrt(len(ce_100P1))
RMSE_ce_10P1=np.linalg.norm((ce_100P1 - ce_10P1)) / np.sqrt(len(ce_100P1))
RMSE_ce_50P1=np.linalg.norm((ce_100P1 - ce_50P1)) / np.sqrt(len(ce_100P1))
#RMSE_ce_20P2=np.linalg.norm((ce_100P1 - ce_20P2)) / np.sqrt(len(ce_100P1))

print('RMSE_jtot_5P1',RMSE_ce_5P1)
print('RMSE_jtot_10P1',RMSE_ce_10P1)
print('RMSE_jtot_50P1',RMSE_ce_50P1)
#print('RMSE_jtot_20P2',RMSE_ce_20P2)

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"Error [A]")
plt.title(battery_name)
plt.legend()
plt.grid(1)



plt.show()