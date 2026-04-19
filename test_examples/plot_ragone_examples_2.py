# Date: 15/02/2026
# Cleaned & simplified version (with proper energy units on x-axis)

import sys; sys.path.insert(0,'..')
from fatDAE.base.basic_import import *

import fatDAE.class_solvers
import fatDAE.class_problem
import fatDAE.dolfin_interface.class_problem
import fatDAE.dolfin_interface.class_control

import batFEM.class_battery
import batFEM.class_machine

from batFEM.PE_PE import class_PE_PE_P2D, class_PE_PE_SPM

import os, json, numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

cell = 'Prada2013'
folder_1C = f'results/dopt3/{cell}/1C/'
folder_2C = f'results/dopt3/{cell}/2C/'

# ==========================================================
# HELPERS
# ==========================================================

def energy_label(mode):
    return {
        "volumetric": "Energy (Wh/L)",
        "gravimetric": "Energy (Wh/kg)"
    }[mode]


def load_batteries(folder):
    gopt, vopt = [], []
    for f in os.listdir(folder):
        if f.endswith(".json"):
            with open(os.path.join(folder, f)) as file:
                if f[12] == 'g':
                    gopt.append(json.load(file))
                else:
                    vopt.append(json.load(file))
    return gopt, vopt


def extract_opt_vectors(data):
    x0, lb, ub, paths = [], [], [], []

    def walk(obj, path):
        if isinstance(obj, dict):
            if obj.get("opt", 0) == 1:
                x0.append(obj["value"])
                lb.append(obj["lb"])
                ub.append(obj["ub"])
                paths.append(path.copy())
            for k, v in obj.items():
                walk(v, path + [k])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, path + [i])

    walk(data, [])
    return np.array(x0), np.array(lb), np.array(ub), paths


def compute_energy(problem, mode="volumetric"):
    dt = np.diff(problem.t_list, prepend=0)
    power = np.array(problem.i_list) * np.array(problem.v_list)
    energy = -np.cumsum(power * dt) / 3600

    if mode == "volumetric":
        return energy / problem.volume
    elif mode == "gravimetric":
        return energy / problem.weight


def safe_plot(problem, mode, attr, **kwargs):
    try:
        x = compute_energy(problem, mode)
        y = getattr(problem, attr)
        plt.plot(x, y, **kwargs)
    except:
        pass


def plot_electrolyte(problem_ref, problem_maxpow, problem_maxene, mode, title):

    plt.figure()

    x_ref = compute_energy(problem_ref, mode)
    plt.plot(x_ref, problem_ref.ce_avg_a_list, color='b', label='Neg. electrode')
    plt.plot(x_ref, problem_ref.ce_avg_s_list, color='g', label='Separator')
    plt.plot(x_ref, problem_ref.ce_avg_c_list, color='r', label='Pos. electrode')

    try:
        x = compute_energy(problem_maxpow, mode)
        plt.plot(x, problem_maxpow.ce_avg_a_list, '--s', color='b', markevery=10)
        plt.plot(x, problem_maxpow.ce_avg_s_list, '--s', color='g', markevery=10)
        plt.plot(x, problem_maxpow.ce_avg_c_list, '--s', color='r', markevery=10)
    except:
        print(f"{title} max power failed")

    try:
        x = compute_energy(problem_maxene, mode)
        plt.plot(x, problem_maxene.ce_avg_a_list, '--^', color='b', markevery=10)
        plt.plot(x, problem_maxene.ce_avg_s_list, '--^', color='g', markevery=10)
        plt.plot(x, problem_maxene.ce_avg_c_list, '--^', color='r', markevery=10)
    except:
        print(f"{title} max energy failed")

    plt.xlabel(energy_label(mode))
    plt.ylabel("Electrolyte concentration (mol/m$^3$)")
    plt.title(title)
    plt.grid(True)
    plt.legend()


def plot_surface(problem_ref, problem_maxpow, problem_maxene, mode, title):

    plt.figure()

    x_ref = compute_energy(problem_ref, mode)
    plt.plot(x_ref, problem_ref.xs_sur_a_list, color='b', label='Neg. electrode')
    plt.plot(x_ref, problem_ref.xs_sur_c_list, color='r', label='Pos. electrode')

    try:
        x = compute_energy(problem_maxpow, mode)
        plt.plot(x, problem_maxpow.xs_sur_a_list, '--s', color='b', markevery=10)
        plt.plot(x, problem_maxpow.xs_sur_c_list, '--s', color='r', markevery=10)
    except:
        print(f"{title} max power failed")

    try:
        x = compute_energy(problem_maxene, mode)
        plt.plot(x, problem_maxene.xs_sur_a_list, '--^', color='b', markevery=10)
        plt.plot(x, problem_maxene.xs_sur_c_list, '--^', color='r', markevery=10)
    except:
        print(f"{title} max energy failed")

    plt.xlabel(energy_label(mode))
    plt.ylabel("Surface concentration (-)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

def print_parameters(label, problem, battery_json):
    try:
        x0, lb, ub, _ = extract_opt_vectors(battery_json)
        xnorm = (x0 - lb) / (ub - lb)

        print(f"\n--- {label} ---")
        print("Raw parameters:")
        print(x0)
        print("Normalized [0-1]:")
        print(xnorm)

    except Exception as e:
        print(f"{label} -> failed ({e})")

# ==========================================================
# LOAD DATA
# ==========================================================

battery_gopt_1C, battery_vopt_1C = load_batteries(folder_1C)
battery_gopt_2C, battery_vopt_2C = load_batteries(folder_2C)

with open(f'json_battery/cells/cell_{cell}_dopt_adj.json') as f:
    battery_json = json.load(f)

with open('json_testplan/DCC_1C.json') as f:
    testplan_1C_json = json.load(f)

with open('json_testplan/DCC_2C.json') as f:
    testplan_2C_json = json.load(f)

with open('json_options/p2d_adjoint.json') as f:
    options_json = json.load(f)

# ==========================================================
# SOLVER
# ==========================================================

embedded_1 = options_json['time discretization']['embedded'] == 0
embedded_2 = options_json['time discretization']['embedded'] == 1

with open(f"../fatDAE/json_butcher/{options_json['time discretization']['type']}/{options_json['time discretization']['name']}.json") as f:
    butcher_json = json.load(f)

solver = fatDAE.class_solvers.build(
    butcher_json,
    embedded_1,
    embedded_2,
    a_tol=options_json['timestepping properties']['abs tolerance'],
    r_tol=options_json['timestepping properties']['rel tolerance'],
    h_max=options_json['timestepping properties']['max step size'],
    h_min=options_json['timestepping properties']['min step size']
)

# ==========================================================
# SIMULATION
# ==========================================================

def sim_battery(battery_json, testplan_json):

    battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/')

    battery_model = batFEM.class_battery.Cell(
        battery_json,
        testplan_json['initial values']['exterior temperature'],
        testplan_json['initial values']['SOC'],
        formulation='vf',
        compute_stoichiometries=False
    )

    testplan = batFEM.class_machine.build_machine(testplan_json, print_level=-1)

    model_type = options_json['general properties']['model']
    if model_type == 'P2D':
        problem = class_PE_PE_P2D.RK_PE_PE_P2D(battery_model, 0., options_json['general properties']['max simulation time'], options_json)
    else:
        problem = class_PE_PE_SPM.RK_PE_PE_SPM(battery_model, 0., options_json['general properties']['max simulation time'], options_json)

    problem.build_pvd(testplan)
    problem.setup_machine(testplan)

    try:
        problem.solve(
            solver,
            state_machine=testplan,
            h=options_json['timestepping properties']['initial step size'],
            adp=(options_json['time discretization']['mode'] == 'adaptive'),
            print_level=-1
        )
    except:
        return None

    problem.weight = battery_model.weight
    problem.volume = battery_model.volume * 1000

    return problem



# ==========================================================
# RUN REFERENCE
# ==========================================================

problem_1C = sim_battery(battery_json, testplan_1C_json)
problem_2C = sim_battery(battery_json, testplan_2C_json)

# ==========================================================
# FIND OPTIMA
# ==========================================================

def find_best(batteries, testplan):
    results = []
    for b in batteries:
        p = sim_battery(b, testplan)
        if p:
            ene_g = compute_energy(p, "gravimetric")[-1]
            pow_g = np.max(np.abs(np.array(p.i_list)*np.array(p.v_list))/p.weight)
            ene_v = compute_energy(p, "volumetric")[-1]
            pow_v = np.max(np.abs(np.array(p.i_list)*np.array(p.v_list))/p.volume)

            results.append((p, b, ene_g, pow_g, ene_v, pow_v))
    return results


res_g_1C = find_best(battery_gopt_1C, testplan_1C_json)
res_v_1C = find_best(battery_vopt_1C, testplan_1C_json)
res_g_2C = find_best(battery_gopt_2C, testplan_2C_json)
res_v_2C = find_best(battery_vopt_2C, testplan_2C_json)

# pick best
problem_maxgpow_1C, json_maxgpow_1C = max(res_g_1C, key=lambda x: x[3])[0:2]
problem_maxgene_1C, json_maxgene_1C = max(res_g_1C, key=lambda x: x[2])[0:2]
problem_maxvpow_1C, json_maxvpow_1C = max(res_v_1C, key=lambda x: x[5])[0:2]
problem_maxvene_1C, json_maxvene_1C = max(res_v_1C, key=lambda x: x[4])[0:2]

problem_maxgpow_2C, json_maxgpow_2C = max(res_g_2C, key=lambda x: x[3])[0:2]
problem_maxgene_2C, json_maxgene_2C = max(res_g_2C, key=lambda x: x[2])[0:2]
problem_maxvpow_2C, json_maxvpow_2C = max(res_v_2C, key=lambda x: x[5])[0:2]
problem_maxvene_2C, json_maxvene_2C = max(res_v_2C, key=lambda x: x[4])[0:2]

# ==========================================================
# PLOTTING
# ==========================================================

def plot_block(mode, title, p_ref1, p_ref2, p_maxpow1, p_maxene1, p_maxpow2, p_maxene2, attr, ylabel):

    plt.figure()

    safe_plot(p_ref1, mode, attr, color='b', label='1C Ref.')
    safe_plot(p_ref2, mode, attr, color='r', label='2C Ref.')

    safe_plot(p_maxpow1, mode, attr, linestyle='--', marker='s', color='b', label='1C Max Pow', markevery=10)
    safe_plot(p_maxene1, mode, attr, linestyle='--', marker='^', color='b', label='1C Max Ene', markevery=10)

    safe_plot(p_maxpow2, mode, attr, linestyle='--', marker='s', color='r', label='2C Max Pow', markevery=10)
    safe_plot(p_maxene2, mode, attr, linestyle='--', marker='^', color='r', label='2C Max Ene', markevery=10)

    plt.xlabel(energy_label(mode))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()


# Voltage
plot_block("volumetric", "Voltage (Volumetric)", problem_1C, problem_2C,
           problem_maxvpow_1C, problem_maxvene_1C,
           problem_maxvpow_2C, problem_maxvene_2C,
           'v_list', "Voltage (V)")

plot_block("gravimetric", "Voltage (Gravimetric)", problem_1C, problem_2C,
           problem_maxgpow_1C, problem_maxgene_1C,
           problem_maxgpow_2C, problem_maxgene_2C,
           'v_list', "Voltage (V)")

# Temperature
plot_block("volumetric", "Temperature (Volumetric)", problem_1C, problem_2C,
           problem_maxvpow_1C, problem_maxvene_1C,
           problem_maxvpow_2C, problem_maxvene_2C,
           'k_list', "Temperature (K)")

plot_block("volumetric", "Temperature (Gravimetric)", problem_1C, problem_2C,
           problem_maxgpow_1C, problem_maxgene_1C,
           problem_maxgpow_2C, problem_maxgene_2C,
           'k_list', "Temperature (K)")

# Concentrations

plot_electrolyte(problem_1C, problem_maxvpow_1C, problem_maxvene_1C,
                 mode="volumetric",
                 title="1C Volumetric")

plot_surface(problem_1C, problem_maxvpow_1C, problem_maxvene_1C,
             mode="volumetric",
             title="1C Volumetric")

plot_electrolyte(problem_2C, problem_maxvpow_2C, problem_maxvene_2C,
                 mode="volumetric",
                 title="2C Volumetric")

plot_surface(problem_2C, problem_maxvpow_2C, problem_maxvene_2C,
             mode="volumetric",
             title="2C Volumetric")

plot_electrolyte(problem_1C, problem_maxgpow_1C, problem_maxgene_1C,
                 mode="gravimetric",
                 title="1C Gravimetric")

plot_surface(problem_1C, problem_maxgpow_1C, problem_maxgene_1C,
             mode="gravimetric",
             title="1C Gravimetric")

plot_electrolyte(problem_2C, problem_maxgpow_2C, problem_maxgene_2C,
                 mode="gravimetric",
                 title="2C Gravimetric")

plot_surface(problem_2C, problem_maxgpow_2C, problem_maxgene_2C,
             mode="gravimetric",
             title="2C Gravimetric")


print("\n================ OPTIMAL PARAMETERS ================\n")

# 1C
print_parameters("1C - Gravimetric Max Power", problem_maxgpow_1C, json_maxgpow_1C)
print_parameters("1C - Gravimetric Max Energy", problem_maxgene_1C, json_maxgene_1C)
print_parameters("1C - Volumetric Max Power", problem_maxvpow_1C, json_maxvpow_1C)
print_parameters("1C - Volumetric Max Energy", problem_maxvene_1C, json_maxvene_1C)

# 2C
print_parameters("2C - Gravimetric Max Power", problem_maxgpow_2C, json_maxgpow_2C)
print_parameters("2C - Gravimetric Max Energy", problem_maxgene_2C, json_maxgene_2C)
print_parameters("2C - Volumetric Max Power", problem_maxvpow_2C, json_maxvpow_2C)
print_parameters("2C - Volumetric Max Energy", problem_maxvene_2C, json_maxvene_2C)

plt.show()