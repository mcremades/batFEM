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


def load_field_at_time(results_dir, field, t_target, domains=("a", "s", "c"),factors=[1,1,1]):
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

battery_name="Chen2020"

json_path = "json_battery/cells/cell_"+battery_name+".json"

with open(json_path) as data_file:
    json_file = json.load(data_file)
json_battery = batFEM.class_battery.parse_json(json_file); 



target_times=np.linspace(0,2600,14)

cs_max_a=json_battery['negativeElectrode']['composition'][0]['maximumConcentration']['value']
cs_max_c=json_battery['positiveElectrode']['composition'][0]['maximumConcentration']['value']
Rp_a=json_battery['negativeElectrode']['composition'][0]['particleRadius']['value']
Rp_c=json_battery['positiveElectrode']['composition'][0]['particleRadius']['value']
Ds_a=json_battery['negativeElectrode']['composition'][0]['diffusionConstant']
Ds_c=json_battery['positiveElectrode']['composition'][0]['diffusionConstant']
epss_a=json_battery['negativeElectrode']['composition'][0]['volumeFraction']['value']
epss_c=json_battery['positiveElectrode']['composition'][0]['volumeFraction']['value']
exp=lambda x: np.exp(x)
x=0.5
for C_rate in [1,2,3,4,5]:
    t_c=3600/C_rate
    print(f"C-rate: {C_rate}C")
    if Ds_a["type"] == "function":
        print('Da_a:',epss_a*Rp_a**2/(t_c*eval(Ds_a["value"])))
    else:
        print('Da_a:',epss_a*Rp_a**2/(t_c*Ds_a["value"]))
    if Ds_c["type"] == "function":
        print('Da_c:',epss_c*Rp_c**2/(t_c*eval(Ds_c["value"])))
    else:
        print('Da_c:',epss_c*Rp_c**2/(t_c*Ds_c["value"]))

Q_bat = json_battery['properties']['nominalCapacity']['value']

times=np.loadtxt("results/"+battery_name+"_SGM10_5C/time.txt")
voltage=np.loadtxt("results/"+battery_name+"_SGM10_5C/voltage.txt")
current=np.loadtxt("results/"+battery_name+"_SGM10_5C/current.txt")
xs_avg_a=np.loadtxt("results/"+battery_name+"_SGM10_5C/xs_avg_a.txt")
xs_avg_c=np.loadtxt("results/"+battery_name+"_SGM10_5C/xs_avg_c.txt")
xs_sur_a=np.loadtxt("results/"+battery_name+"_SGM10_5C/xs_sur_a.txt")
xs_sur_c=np.loadtxt("results/"+battery_name+"_SGM10_5C/xs_sur_c.txt")

times_2=np.loadtxt("results/"+battery_name+"_SGM2_5C/time.txt")
voltage_2=np.interp(times,times_2,np.loadtxt("results/"+battery_name+"_SGM2_5C/voltage.txt"))
xs_avg_a_2=np.interp(times,times_2,np.loadtxt("results/"+battery_name+"_SGM2_5C/xs_avg_a.txt"))
xs_avg_c_2=np.interp(times,times_2,np.loadtxt("results/"+battery_name+"_SGM2_5C/xs_avg_c.txt"))
xs_sur_a_2=np.interp(times,times_2,np.loadtxt("results/"+battery_name+"_SGM2_5C/xs_sur_a.txt"))
xs_sur_c_2=np.interp(times,times_2,np.loadtxt("results/"+battery_name+"_SGM2_5C/xs_sur_c.txt"))

times_3=np.loadtxt("results/"+battery_name+"_SGM3_5C/time.txt")
voltage_3=np.interp(times,times_3,np.loadtxt("results/"+battery_name+"_SGM3_5C/voltage.txt"))
xs_avg_a_3=np.interp(times,times_3,np.loadtxt("results/"+battery_name+"_SGM3_5C/xs_avg_a.txt"))
xs_avg_c_3=np.interp(times,times_3,np.loadtxt("results/"+battery_name+"_SGM3_5C/xs_avg_c.txt"))
xs_sur_a_3=np.interp(times,times_3,np.loadtxt("results/"+battery_name+"_SGM3_5C/xs_sur_a.txt"))
xs_sur_c_3=np.interp(times,times_3,np.loadtxt("results/"+battery_name+"_SGM3_5C/xs_sur_c.txt"))

times_4=np.loadtxt("results/"+battery_name+"_SGM4_5C/time.txt")
voltage_4=np.interp(times,times_4,np.loadtxt("results/"+battery_name+"_SGM4_5C/voltage.txt"))
xs_avg_a_4=np.interp(times,times_4,np.loadtxt("results/"+battery_name+"_SGM4_5C/xs_avg_a.txt"))
xs_avg_c_4=np.interp(times,times_4,np.loadtxt("results/"+battery_name+"_SGM4_5C/xs_avg_c.txt"))
xs_sur_a_4=np.interp(times,times_4,np.loadtxt("results/"+battery_name+"_SGM4_5C/xs_sur_a.txt"))
xs_sur_c_4=np.interp(times,times_4,np.loadtxt("results/"+battery_name+"_SGM4_5C/xs_sur_c.txt"))

times_5=np.loadtxt("results/"+battery_name+"_SGM5_5C/time.txt")
voltage_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_SGM5_5C/voltage.txt"))
xs_avg_a_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_SGM5_5C/xs_avg_a.txt"))
xs_avg_c_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_SGM5_5C/xs_avg_c.txt"))
xs_sur_a_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_SGM5_5C/xs_sur_a.txt"))
xs_sur_c_5=np.interp(times,times_5,np.loadtxt("results/"+battery_name+"_SGM5_5C/xs_sur_c.txt"))


plt.figure()
plt.plot(times,current/Q_bat)
plt.xlabel("Time [s]")
plt.ylabel("C-rate [1/h]")
plt.title(f"5C Charge/Discharge Profile")
plt.legend()
plt.grid(1)

plt.figure()
plt.plot(times,voltage,color='black',label=r'$N_s=10$')
plt.plot(times,voltage_2,'--',color='blue',label=r'$N_s=2$')
plt.plot(times,voltage_3,'--',color='red',label=r'$N_s=3$')
plt.plot(times,voltage_4,'--',color='orange',label=r'$N_s=4$')
plt.plot(times,voltage_5,'--',color='green',label=r'$N_s=5$')
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

plt.figure()
plt.plot(times,abs(voltage-voltage_2),'--',color='blue',label=r'$N_s=2$')
plt.plot(times,abs(voltage-voltage_3),'--',color='red',label=r'$N_s=3$')
plt.plot(times,abs(voltage-voltage_4),'--',color='orange',label=r'$N_s=4$')
plt.plot(times,abs(voltage-voltage_5),'--',color='green',label=r'$N_s=5$')
plt.xlabel("Time [s]")
plt.ylabel("Error [V]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

print('RMSE Voltage SGM2',np.linalg.norm(1000*(voltage - voltage_2)) / np.sqrt(len(voltage)))
print('RMSE Voltage SGM3',np.linalg.norm(1000*(voltage - voltage_3)) / np.sqrt(len(voltage)))
print('RMSE Voltage SGM4',np.linalg.norm(1000*(voltage - voltage_4)) / np.sqrt(len(voltage)))
print('RMSE Voltage SGM5',np.linalg.norm(1000*(voltage - voltage_5)) / np.sqrt(len(voltage)))
plt.figure()
plt.plot(times,xs_sur_a,color='black',label=r'$N_s=10$')
plt.plot(times,xs_avg_a,color='black')
plt.plot(times,xs_sur_a_2,'--',color='blue',label=r'$N_s=2$')
plt.plot(times,xs_avg_a_2,'--',color='blue')
plt.plot(times,xs_sur_a_3,'--',color='red',label=r'$N_s=3$')
plt.plot(times,xs_avg_a_3,'--',color='red')
plt.plot(times,xs_sur_a_4,'--',color='orange',label=r'$N_s=4$')
plt.plot(times,xs_avg_a_4,'--',color='orange')
plt.plot(times,xs_sur_a_5,'--',color='green',label=r'$N_s=5$')
plt.plot(times,xs_avg_a_5,'--',color='green')
plt.xlabel("Time [s]")
plt.ylabel(r"$x_{s,sur}^-$ [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

plt.figure()
plt.plot(times,abs(xs_sur_a-xs_sur_a_2),'--',color='blue',label=r'$N_s=2$')
plt.plot(times,abs(xs_sur_a-xs_sur_a_3),'--',color='red',label=r'$N_s=3$')
plt.plot(times,abs(xs_sur_a-xs_sur_a_4),'--',color='orange',label=r'$N_s=4$')
plt.plot(times,abs(xs_sur_a-xs_sur_a_5),'--',color='green',label=r'$N_s=5$')
plt.xlabel("Time [s]")
plt.ylabel("Error [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)
print('RMSE xs_sur_a SGM2',np.linalg.norm((xs_sur_a - xs_sur_a_2)) / np.sqrt(len(xs_sur_a)))
print('RMSE xs_sur_a SGM3',np.linalg.norm((xs_sur_a - xs_sur_a_3)) / np.sqrt(len(xs_sur_a)))
print('RMSE xs_sur_a SGM4',np.linalg.norm((xs_sur_a - xs_sur_a_4)) / np.sqrt(len(xs_sur_a)))
print('RMSE xs_sur_a SGM5',np.linalg.norm((xs_sur_a - xs_sur_a_5)) / np.sqrt(len(xs_sur_a)))

plt.figure()
plt.plot(times,xs_sur_c,color='black',label=r'$N_s=10$')
plt.plot(times,xs_avg_c,color='black')
plt.plot(times,xs_sur_c_2,'--',color='blue',label=r'$N_s=2$')
plt.plot(times,xs_avg_c_2,'--',color='blue')
plt.plot(times,xs_sur_c_3,'--',color='red',label=r'$N_s=3$')
plt.plot(times,xs_avg_c_3,'--',color='red')
plt.plot(times,xs_sur_c_4,'--',color='orange',label=r'$N_s=4$')
plt.plot(times,xs_avg_c_4,'--',color='orange')
plt.plot(times,xs_sur_c_5,'--',color='green',label=r'$N_s=5$')
plt.plot(times,xs_avg_c_5,'--',color='green')
plt.xlabel("Time [s]")
plt.ylabel(r"$x_{s,sur}^+$ [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

plt.figure()
plt.plot(times,abs(xs_sur_c-xs_sur_c_2),'--',color='blue',label=r'$N_s=2$')
plt.plot(times,abs(xs_sur_c-xs_sur_c_3),'--',color='red',label=r'$N_s=3$')
plt.plot(times,abs(xs_sur_c-xs_sur_c_4),'--',color='orange',label=r'$N_s=4$')
plt.plot(times,abs(xs_sur_c-xs_sur_c_5),'--',color='green',label=r'$N_s=5$')
plt.xlabel("Time [s]")
plt.ylabel("Error [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)
RMSE_xs_sur_c_2=np.linalg.norm((xs_sur_c - xs_sur_c_2)) / np.sqrt(len(xs_sur_c))
RMSE_xs_sur_c_3=np.linalg.norm((xs_sur_c - xs_sur_c_3)) / np.sqrt(len(xs_sur_c))
RMSE_xs_sur_c_4=np.linalg.norm((xs_sur_c - xs_sur_c_4)) / np.sqrt(len(xs_sur_c))
RMSE_xs_sur_c_5=np.linalg.norm((xs_sur_c - xs_sur_c_5)) / np.sqrt(len(xs_sur_c))
print('RMSE xs_sur_c SGM2',RMSE_xs_sur_c_2)
print('RMSE xs_sur_c SGM3',RMSE_xs_sur_c_3)
print('RMSE xs_sur_c SGM4',RMSE_xs_sur_c_4)
print('RMSE xs_sur_c SGM5',RMSE_xs_sur_c_5)

plt.figure()
plt.plot([2,3,4,5],[RMSE_xs_sur_c_2,RMSE_xs_sur_c_3,RMSE_xs_sur_c_4,RMSE_xs_sur_c_5])


plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x, ce = load_field_at_time("results/"+battery_name+"_SGM10_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, color=color,label=str(target_time)+" s")
    x, ce = load_field_at_time("results/"+battery_name+"_SGM2_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '--',color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_SGM3_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '-.', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_SGM4_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, ':', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_SGM5_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
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

    x, ce = load_field_at_time("results/"+battery_name+"_SGM10_5C/", field="c_s_avg", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, color=color,label=str(target_time)+" s")
    x, ce = load_field_at_time("results/"+battery_name+"_SGM2_5C/", field="c_s_avg", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '--',color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_SGM3_5C/", field="c_s_avg", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '-.', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_SGM4_5C/", field="c_s_avg", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, ':', color=color)
    x, ce = load_field_at_time("results/"+battery_name+"_SGM5_5C/", field="c_s_avg", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, ce, '.', color=color)
    

plt.xlabel("Cell thickness [-]")
plt.ylabel(r"$x_{s,avg}$ [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)


plt.figure()
cmap = plt.get_cmap('tab10')
n_times = len(target_times)


for i, target_time in enumerate(target_times):
    color = cmap(i % cmap.N)

    x, cs_2 = load_field_at_time("results/"+battery_name+"_SGM2_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    x, cs_3 = load_field_at_time("results/"+battery_name+"_SGM3_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    x, cs_4 = load_field_at_time("results/"+battery_name+"_SGM4_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    x, cs_5 = load_field_at_time("results/"+battery_name+"_SGM5_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    x, cs_10 = load_field_at_time("results/"+battery_name+"_SGM10_5C/", field="c_s_sur", t_target=target_time,factors=[1/cs_max_a,1,1/cs_max_c])
    plt.plot(x, abs(cs_10-cs_5), '.', color=color,label=str(target_time)+" s")
    plt.plot(x, abs(cs_10-cs_4), ':', color=color)
    plt.plot(x, abs(cs_10-cs_3), '-.', color=color)
    plt.plot(x, abs(cs_10-cs_2), '--',color=color)



plt.xlabel("Cell thickness [-]")
plt.ylabel(r"Error [-]")
plt.title(battery_name)
plt.legend()
plt.grid(1)

plt.show()