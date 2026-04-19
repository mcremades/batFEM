# Date: 15/02/2026
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'..'); from fatDAE.base.basic_import import *

import fatDAE.class_solvers
import fatDAE.class_problem

import fatDAE.dolfin_interface.class_problem
import fatDAE.dolfin_interface.class_control

import batFEM.class_battery
import batFEM.class_machine

from batFEM.PE_PE import class_PE_PE_P2D, class_PE_PE_SPM
from batFEM.ME_PE import class_ME_PE_P2D


with open('json_battery/cells/cell_Marquis2019_dopt.json') as data_file:
    battery_json = json.load(data_file)
with open('json_testplan/DCC.json') as data_file:
    testplan_json = json.load(data_file)
with open('json_options/recommended_p2d_dopt.json') as data_file:
    options_json = json.load(data_file)

# Solver definition

if options_json['time discretization']['embedded'] == 0:
    embedded_1 = True
    embedded_2 = False
else:
    if options_json['time discretization']['embedded'] == 1:
        embedded_1 = False
        embedded_2 = True
    else:
        raise NameError('Embedded should be 0 or 1...')

with open('../fatDAE/json_butcher/'+options_json['time discretization']['type']+'/'+options_json['time discretization']['name']+'.json') as data_file:
    butcher_json = json.load(data_file)

solver = fatDAE.class_solvers.build(butcher_json, \
                                    embedded_1, \
                                    embedded_2, \
                                    a_tol=options_json['timestepping properties']['abs tolerance'], \
                                    r_tol=options_json['timestepping properties']['rel tolerance'], \
                                    h_max=options_json['timestepping properties']['max step size'], \
                                    h_min=options_json['timestepping properties']['min step size'])
    
def extract_opt_vectors(data):
    """
    Returns:
        x0, lb, ub, paths
    where paths are index/key routes into the JSON.
    """
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
    return x0, lb, ub, paths

def apply_opt_vector(data, paths, x_new):
    """
    Updates JSON by navigating explicit paths.
    """

    if len(paths) != len(x_new):
        raise ValueError("Length mismatch between paths and x")

    for path, val in zip(paths, x_new):

        obj = data
        for key in path:
            obj = obj[key]

        obj["value"] = float(val)

    return data

x0,lb,ub,refs = extract_opt_vectors(battery_json)

print("Initial values:", x0)
print("Lower bounds:", lb)
print("Upper bounds:", ub)

battery_json_0 = copy.deepcopy(battery_json)

def cost(x):

    battery_json = apply_opt_vector(battery_json_0, refs, x)
    
    battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/')
    
    battery_model = batFEM.class_battery.Cell(battery_json, testplan_json['initial values']['exterior temperature'], \
                                                            testplan_json['initial values']['SOC'],formulation='wf',compute_stoichiometries=True); 
    
    testplan = batFEM.class_machine.build_machine(testplan_json,print_level=-1)

    try:
        if options_json['general properties']['model'] == 'P2D':
            problem = class_PE_PE_P2D.RK_PE_PE_P2D(battery_model, 0., options_json['general properties']['max simulation time'], options_json, save_path=None)
        elif options_json['general properties']['model'] == 'SPM':
            problem = class_PE_PE_SPM.RK_PE_PE_SPM(battery_model, 0., options_json['general properties']['max simulation time'], options_json, save_path=None)
        elif options_json['general properties']['model'] == 'SPME':
            problem = class_PE_PE_SPM.RK_PE_PE_SPME(battery_model, 0., options_json['general properties']['max simulation time'], options_json, save_path=None)
        else:
            raise NameError('Model not implemented yet')

        # Solve problem

        problem.build_pvd(testplan); problem.setup_machine(testplan)

        if options_json['time discretization']['mode'] == 'adaptive':
            problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=True, print_level=-1)
        else:
            problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=False, print_level=-1)
        
        ene = abs(problem.get_ene())
        pow = abs(problem.get_pow())

        if problem.t_list[-1] < 600:
            print("Simulation did not complete. Returning zero cost.")
            ene=0; pow=0
    except:
        ene=0; pow=0
    
    

    ene_weight = ene / battery_model.weight
    ene_volume = ene / (battery_model.volume * 1000)
    pow_weight = pow / battery_model.weight
    pow_volume = pow / (battery_model.volume * 1000)

    print('Energy [Wh/kg]:', ene_weight, 'Power [W/kg]:', pow_weight)
    print('Energy [Wh/m3]:', ene_volume, 'Power [W/m3]:', pow_volume)

    return ene_weight, pow_weight, ene_volume, pow_volume

def cost_gene(u):
    gene, gpow, vene, vpow = cost(u)
    return -gene

def cost_gpow(u):
    gene, gpow, vene, vpow = cost(u)
    return -gpow

def cost_vene(u):
    gene, gpow, vene, vpow = cost(u)
    return -vene

def cost_vpow(u):
    gene, gpow, vene, vpow = cost(u)
    return -vpow


import scipy.optimize
import matplotlib.pyplot as plt

gene_og, gpow_og, vpow, vene = cost(x0)

min_gpow_list=range(240,300,10)
min_gene_list=range(240,300,10)
min_vpow_list=range(400,700,50)
min_vene_list=range(400,700,50)

opt_gene_mpow_list = []
opt_gpow_mpow_list = []
opt_gene_mene_list = []
opt_gpow_mene_list = []

opt_vene_mpow_list = []
opt_vpow_mpow_list = []
opt_vene_mene_list = []
opt_vpow_mene_list = []



for pow_min in min_gpow_list:
    with open('results/Marquis2019_dopt/battery_min_gpow_'+str(pow_min)+'.json') as data_file:
        battery_json = json.load(data_file)
    x0,lb,ub,refs = extract_opt_vectors(battery_json)
    opt_gene, opt_gpow, opt_vene, opt_vpow = cost(x0)
    opt_gene_mpow_list.append(opt_gene)
    opt_gpow_mpow_list.append(opt_gpow)

for ene_min in min_gene_list:
    with open('results/Marquis2019_dopt/battery_min_gene_'+str(ene_min)+'.json') as data_file:
        battery_json = json.load(data_file)
    x0,lb,ub,refs = extract_opt_vectors(battery_json)
    opt_gene, opt_gpow, opt_vene, opt_vpow = cost(x0)
    opt_gene_mene_list.append(opt_gene)
    opt_gpow_mene_list.append(opt_gpow)

plt.figure()
plt.plot(opt_gpow_mpow_list, opt_gene_mpow_list, 'bx')
plt.plot(opt_gpow_mene_list, opt_gene_mene_list, 'bx')
plt.plot(gpow_og, gene_og, 'bo')
plt.xlabel('Power [W/kg]')
plt.ylabel('Energy [Wh/kg]')
plt.title('Ragone Plot')
plt.grid(True)

# Volumetric

for pow_min in min_vpow_list:
    with open('results/Marquis2019_dopt/battery_min_vpow_'+str(pow_min)+'.json') as data_file:
        battery_json = json.load(data_file)
    x0,lb,ub,refs = extract_opt_vectors(battery_json)
    opt_gene, opt_gpow, opt_vene, opt_vpow = cost(x0)
    opt_vene_mpow_list.append(opt_vene)
    opt_vpow_mpow_list.append(opt_vpow)

for ene_min in min_vene_list:
    with open('results/Marquis2019_dopt/battery_min_vene_'+str(ene_min)+'.json') as data_file:
        battery_json = json.load(data_file)
    x0,lb,ub,refs = extract_opt_vectors(battery_json)
    opt_gene, opt_gpow, opt_vene, opt_vpow = cost(x0)
    opt_vene_mene_list.append(opt_vene)
    opt_vpow_mene_list.append(opt_vpow)

plt.figure()
plt.plot(opt_vpow_mpow_list, opt_vene_mpow_list, 'bx')
plt.plot(opt_vpow_mene_list, opt_vene_mene_list, 'bx')
plt.plot(gpow_og, gene_og, 'bo')
plt.xlabel('Power [W/L]')
plt.ylabel('Energy [Wh/L]')
plt.title('Ragone Plot')
plt.grid(True)

plt.show()