#!/usr/bin/env python3
# Date: 18/01/2026
# Author: Manuel Cremades

import sys, os, shutil, argparse, json, copy
import numpy as np
import matplotlib.pyplot as plt

# Add fatDAE and batFEM to path
sys.path.insert(0,'..')

from fatDAE.base.basic_import import *
import fatDAE.class_solvers
import fatDAE.class_problem
import fatDAE.dolfin_interface.class_problem
import fatDAE.dolfin_interface.class_control
import batFEM.class_battery
import batFEM.class_machine
from batFEM.PE_PE import class_PE_PE_P2D

# ---- Parser ----
parser = argparse.ArgumentParser(description="batFEM gradient validation")
parser.add_argument("battery_json")
parser.add_argument("testplan_json")
parser.add_argument("options_json")
parser.add_argument("-output")
args = parser.parse_args()

save_path = 'results/' + str(args.output) + '/'
try:
    os.stat(save_path)
    shutil.rmtree(save_path)
except:
    pass
os.mkdir(save_path)

with open(args.battery_json) as f:
    battery_json = json.load(f)
with open(args.testplan_json) as f:
    testplan_json = json.load(f)
with open(args.options_json) as f:
    options_json = json.load(f)

# ---- Determine embedded RK ----
if options_json['time discretization']['embedded'] == 0:
    embedded_1, embedded_2 = True, False
elif options_json['time discretization']['embedded'] == 1:
    embedded_1, embedded_2 = False, True
else:
    raise ValueError("Embedded should be 0 or 1")

# ---- Load Butcher table ----
with open('../fatDAE/json_butcher/'+options_json['time discretization']['type']+'/' +
          options_json['time discretization']['name']+'.json') as f:
    butcher_json = json.load(f)



# ---- Extract optimization parameters ----
def extract_opt_params(data):
    params = []
    def visit(node, path):
        if isinstance(node, dict):
            if node.get("opt",0) == 1 and "value" in node:
                params.append({
                    "path": path,
                    "value": node["value"],
                    "lb": node.get("lb"),
                    "ub": node.get("ub"),
                    "unit": node.get("unit")
                })
            for k,v in node.items():
                visit(v, path+[k])
        elif isinstance(node, list):
            for i,v in enumerate(node):
                visit(v, path+[i])
    visit(data, [])
    return params

opt_params = extract_opt_params(battery_json)

# ---- Helper functions ----
def set_param(data, path, new_value):
    node = data
    for k in path[:-1]:
        node = node[k]
    node[path[-1]]["value"] = new_value

def get_param(data, path):
    node = data
    for k in path:
        node = node[k]
    return node["value"]

# ---- Define cost computation wrapper ----
def compute_cost_ene(battery_json_local):
    battery_local = batFEM.class_battery.Cell(
        battery_json_local,
        testplan_json['initial values']['exterior temperature'],
        testplan_json['initial values']['SOC']
    )
    testplan_local = batFEM.class_machine.build_machine(testplan_json)
    vproblem.build_pvd(testplan_local)
    vproblem.setup_machine(testplan_local)

    if options_json['time discretization']['mode'] == 'adaptive':
        cst = vproblem.solve(
            solver,
            state_machine=testplan_local,
            h=options_json['timestepping properties']['initial step size'],
            adp=True, adj=False, print_level=0
        )
    else:
        cst = vproblem.solve(
            solver,
            state_machine=testplan_local,
            h=options_json['timestepping properties']['initial step size'],
            adp=False, adj=False, print_level=0
        )
    return cst

# ---- Initialize problems ----
battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/')

battery = batFEM.class_battery.Cell(
    battery_json,
    testplan_json['initial values']['exterior temperature'],
    testplan_json['initial values']['SOC']
)
testplan = batFEM.class_machine.build_machine(testplan_json)

vproblem = class_PE_PE_P2D.RK_PE_PE_P2D_VolumetricEnergy(
    battery, opt_params, 0.,
    options_json['general properties']['max simulation time'],
    options_json, save_path=save_path
)
gproblem = class_PE_PE_P2D.RK_PE_PE_P2D_GravimetricEnergy(
    battery, opt_params, 0.,
    options_json['general properties']['max simulation time'],
    options_json, save_path=save_path
)

# ---- Solve adjoint gradients ----
# ---- Build solver ----
solver = fatDAE.class_solvers.build(
    butcher_json,
    embedded_1, embedded_2,
    a_tol=options_json['timestepping properties']['abs tolerance'],
    r_tol=options_json['timestepping properties']['rel tolerance'],
    h_max=options_json['timestepping properties']['max step size'],
    h_min=options_json['timestepping properties']['min step size']
)
vproblem.build_pvd(testplan)
vproblem.setup_machine(testplan)
if options_json['time discretization']['mode'] == 'adaptive':
    cst_vene, grad_vene = vproblem.solve(
        solver, state_machine=testplan,
        h=options_json['timestepping properties']['initial step size'],
        adp=True, adj=True, print_level=2
    )
else:
    cst_vene, grad_vene = vproblem.solve(
        solver, state_machine=testplan,
        h=options_json['timestepping properties']['initial step size'],
        adp=False, adj=True, print_level=2
    )

# ---- Build solver ----
solver = fatDAE.class_solvers.build(
    butcher_json,
    embedded_1, embedded_2,
    a_tol=options_json['timestepping properties']['abs tolerance'],
    r_tol=options_json['timestepping properties']['rel tolerance'],
    h_max=options_json['timestepping properties']['max step size'],
    h_min=options_json['timestepping properties']['min step size']
)
testplan = batFEM.class_machine.build_machine(testplan_json)
gproblem.build_pvd(testplan)
gproblem.setup_machine(testplan)
if options_json['time discretization']['mode'] == 'adaptive':
    cst_gene, grad_gene = gproblem.solve(
        solver, state_machine=testplan,
        h=options_json['timestepping properties']['initial step size'],
        adp=True, adj=True, print_level=2
    )
else:
    cst_gene, grad_gene = gproblem.solve(
        solver, state_machine=testplan,
        h=options_json['timestepping properties']['initial step size'],
        adp=False, adj=True, print_level=2
    )

print("VENE adjoint:", grad_vene)
print("GENE adjoint:", grad_gene)

# ---- Finite difference validation ----
eps = 1e-6
fd_grads = []

for param in opt_params:
    path = param["path"]
    val = param["value"]

    battery_plus = copy.deepcopy(battery_json)
    battery_minus = copy.deepcopy(battery_json)

    set_param(battery_plus, path, val + eps)
    set_param(battery_minus, path, val - eps)

    J_plus = compute_cost_ene(battery_plus)
    J_minus = compute_cost_ene(battery_minus)

    fd_grad = (J_plus - J_minus)/(2*eps)
    fd_grads.append(fd_grad)

fd_grads = np.array(fd_grads)

# ---- Plot adjoint vs FD gradients ----
idx = np.arange(len(fd_grads))
plt.figure()
plt.plot(idx, grad_vene, "o-", label="VENE adjoint")
plt.plot(idx, fd_grads, "s--", label="VENE finite difference")
plt.xlabel("Control parameter")
plt.ylabel("dJ/dp")
plt.legend()
plt.show()