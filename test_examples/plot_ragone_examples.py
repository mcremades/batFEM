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

import os
import json
import numpy as np

cell='Prada2013'

 
folder_1C = 'results/dopt3/'+cell+'/1C/'
folder_2C = 'results/dopt3/'+cell+'/2C/'

battery_gopt_1C = []
battery_vopt_1C = []
battery_gopt_2C = []
battery_vopt_2C = []

for filename in os.listdir(folder_1C):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_1C, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            if filename[12]=='g':
                battery_gopt_1C.append(json.load(f))
            else:
                battery_vopt_1C.append(json.load(f))

for filename in os.listdir(folder_2C):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_2C, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            if filename[12]=='g':
                battery_gopt_2C.append(json.load(f))
            else:
                battery_vopt_2C.append(json.load(f))


with open('json_battery/cells/cell_'+cell+'_dopt.json') as data_file:
    battery_json = json.load(data_file)
with open('json_testplan/DCC_1C.json') as data_file:
    testplan_1C_json = json.load(data_file)
with open('json_testplan/DCC_2C.json') as data_file:
    testplan_2C_json = json.load(data_file)
with open('json_options/recommended_p2d.json') as data_file:
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
    return np.array(x0), np.array(lb), np.array(ub), paths

x0_ref,lb_ref,ub_ref,refs = extract_opt_vectors(battery_json)

def sim_battery(battery_json, testplan_json, options_json):

    battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/')

    
    battery_model = batFEM.class_battery.Cell(battery_json, testplan_json['initial values']['exterior temperature'], \
                                                            testplan_json['initial values']['SOC'],formulation='wf',compute_stoichiometries=True); 
    
    testplan = batFEM.class_machine.build_machine(testplan_json,print_level=-1)

    
    if options_json['general properties']['model'] == 'P2D':
        problem = class_PE_PE_P2D.RK_PE_PE_P2D(battery_model, 0., options_json['general properties']['max simulation time'], options_json, save_path='results')
    elif options_json['general properties']['model'] == 'SPM':
        problem = class_PE_PE_SPM.RK_PE_PE_SPM(battery_model, 0., options_json['general properties']['max simulation time'], options_json, save_path='results')
    elif options_json['general properties']['model'] == 'SPME':
        problem = class_PE_PE_SPM.RK_PE_PE_SPME(battery_model, 0., options_json['general properties']['max simulation time'], options_json, save_path='results')
    else:
        raise NameError('Model not implemented yet')

    # Solve problem

    problem.build_pvd(testplan); problem.setup_machine(testplan)

    try:
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

    return ene_weight, pow_weight, ene_volume, pow_volume, problem

ene_weight_1C, pow_weight_1C, ene_volume_1C, pow_volume_1C, problem_1C = sim_battery(battery_json, testplan_1C_json, options_json)
ene_weight_2C, pow_weight_2C, ene_volume_2C, pow_volume_2C, problem_2C = sim_battery(battery_json, testplan_2C_json, options_json)


gene_list_1C=[]
gpow_list_1C=[]
vene_list_1C=[]
vpow_list_1C=[]
gx0_list_1C=[]
vx0_list_1C=[]

for i in range(len(battery_gopt_1C)):
    x0,lb,ub,refs = extract_opt_vectors(battery_gopt_1C[i])
    gene, gpow, vene, vpow, problem = sim_battery(battery_gopt_1C[i], testplan_1C_json, options_json)
    if gene > 0:
        gene_list_1C.append(gene)
        gpow_list_1C.append(gpow)
        gx0_list_1C.append((x0-lb)/(ub-lb))
for i in range(len(battery_vopt_1C)):
    x0,lb,ub,refs = extract_opt_vectors(battery_vopt_1C[i])
    gene, gpow, vene, vpow, problem = sim_battery(battery_vopt_1C[i], testplan_1C_json, options_json)
    if gene > 0:
        vene_list_1C.append(vene)
        vpow_list_1C.append(vpow)
        vx0_list_1C.append((x0-lb)/(ub-lb))

gene_list_2C=[]
gpow_list_2C=[]
vene_list_2C=[]
vpow_list_2C=[]
gx0_list_2C=[]
vx0_list_2C=[]

for i in range(len(battery_gopt_2C)):
    x0,lb,ub,refs = extract_opt_vectors(battery_gopt_2C[i])
    gene, gpow, vene, vpow, problem = sim_battery(battery_gopt_2C[i], testplan_2C_json, options_json)
    if gene > 0:
        gene_list_2C.append(gene)
        gpow_list_2C.append(gpow)
        gx0_list_2C.append((x0-lb)/(ub-lb))
for i in range(len(battery_vopt_2C)):
    x0,lb,ub,refs = extract_opt_vectors(battery_vopt_2C[i])
    gene, gpow, vene, vpow, problem = sim_battery(battery_vopt_2C[i], testplan_2C_json, options_json)
    if gene > 0:
        vene_list_2C.append(vene)
        vpow_list_2C.append(vpow)
        vx0_list_2C.append((x0-lb)/(ub-lb))

# --- 1C ---
idx_maxgpow_1C = np.argmax(gpow_list_1C)
idx_maxgene_1C = np.argmax(gene_list_1C)
idx_maxvpow_1C = np.argmax(vpow_list_1C)
idx_maxvene_1C = np.argmax(vene_list_1C)

# --- 2C ---
idx_maxgpow_2C = np.argmax(gpow_list_2C)
idx_maxgene_2C = np.argmax(gene_list_2C)
idx_maxvpow_2C = np.argmax(vpow_list_2C)
idx_maxvene_2C = np.argmax(vene_list_2C)

# 1C
gpmax_1C, ge_at_pmax_1C = gpow_list_1C[idx_maxgpow_1C], gene_list_1C[idx_maxgpow_1C]
gp_at_emax_1C, gemax_1C = gpow_list_1C[idx_maxgene_1C], gene_list_1C[idx_maxgene_1C]
vpmax_1C, ve_at_pmax_1C = vpow_list_1C[idx_maxvpow_1C], vene_list_1C[idx_maxvpow_1C]
vp_at_emax_1C, vemax_1C = vpow_list_1C[idx_maxvene_1C], vene_list_1C[idx_maxvene_1C]

# 2C
gpmax_2C, ge_at_pmax_2C = gpow_list_2C[idx_maxgpow_2C], gene_list_2C[idx_maxgpow_2C]
gp_at_emax_2C, gemax_2C = gpow_list_2C[idx_maxgene_2C], gene_list_2C[idx_maxgene_2C]
vpmax_2C, ve_at_pmax_2C = vpow_list_2C[idx_maxvpow_2C], vene_list_2C[idx_maxvpow_2C]
vp_at_emax_2C, vemax_2C = vpow_list_2C[idx_maxvene_2C], vene_list_2C[idx_maxvene_2C]
     
import matplotlib.pyplot as plt

plt.figure()
plt.plot(gpow_list_1C, gene_list_1C, 'bo',
         label='1C optimums', zorder=1, markerfacecolor='none')
plt.plot(gpow_list_2C, gene_list_2C, 'ro',
         label='2C optimums', zorder=1, markerfacecolor='none')
plt.plot(pow_weight_1C, ene_weight_1C, 'b*',
         markersize=12, label='1C original', zorder=3)
plt.plot(pow_weight_2C, ene_weight_2C, 'r*',
         markersize=12, label='2C original', zorder=3)
plt.plot(gpmax_1C, ge_at_pmax_1C, 'bs', markersize=9,
         label='1C max power', zorder=4)
plt.plot(gp_at_emax_1C, gemax_1C, 'b^', markersize=9,
         label='1C max energy', zorder=4)
plt.plot(gpmax_2C, ge_at_pmax_2C, 'rs', markersize=9,
         label='2C max power', zorder=4)
plt.plot(gp_at_emax_2C, gemax_2C, 'r^', markersize=9,
         label='2C max energy', zorder=4)
plt.xlabel('Power [W/kg]')
plt.ylabel('Energy [Wh/kg]')
plt.title('Ragone Plot')
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(vpow_list_1C, vene_list_1C, 'bo',
         label='1C optimums', zorder=1, markerfacecolor='none')
plt.plot(vpow_list_2C, vene_list_2C, 'ro',
         label='2C optimums', zorder=1, markerfacecolor='none')
plt.plot(pow_volume_1C, ene_volume_1C, 'b*',
         markersize=12, label='1C original', zorder=3)
plt.plot(pow_volume_2C, ene_volume_2C, 'r*',
         markersize=12, label='2C original', zorder=3)
plt.plot(vpmax_1C, ve_at_pmax_1C, 'bs', markersize=9,
         label='1C max power', zorder=4)
plt.plot(vp_at_emax_1C, vemax_1C, 'b^', markersize=9,
         label='1C max energy', zorder=4)
plt.plot(vpmax_2C, ve_at_pmax_2C, 'rs', markersize=9,
         label='2C max power', zorder=4)
plt.plot(vp_at_emax_2C, vemax_2C, 'r^', markersize=9,
         label='2C max energy', zorder=4)
plt.xlabel('Power [W/L]')
plt.ylabel('Energy [Wh/L]')
plt.title('Ragone Plot')
plt.grid(True)
plt.legend()

params=['$n_{Li,0}$','$m_l^-$','$w_s^-$','$m_l^+$','$w_s^+$']

x0_maxgpow_1C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_gopt_1C[idx_maxgpow_1C])
x0_maxgpow_2C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_gopt_2C[idx_maxgpow_2C])
x0_maxvpow_1C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_vopt_1C[idx_maxvpow_1C])
x0_maxvpow_2C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_vopt_2C[idx_maxvpow_2C])

x0_maxgene_1C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_gopt_1C[idx_maxgene_1C])
x0_maxgene_2C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_gopt_2C[idx_maxgene_2C])
x0_maxvene_1C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_vopt_1C[idx_maxvene_1C])
x0_maxvene_2C,lb_ref,ub_ref,refs = extract_opt_vectors(battery_vopt_2C[idx_maxvene_2C])


x0_ref,lb_ref,ub_ref,refs = extract_opt_vectors(battery_json)


plt.figure()
plt.scatter(params,x0_maxgpow_1C,c='b',marker='s')
plt.scatter(params,x0_maxgene_1C,c='b',marker='^')
plt.scatter(params,x0_maxgpow_2C,c='r',marker='s')
plt.scatter(params,x0_maxgene_2C,c='r',marker='^')
plt.scatter(params,x0_ref,c='g',marker='*')
plt.scatter(params,ub_ref,c='black',marker='s')
plt.scatter(params,lb_ref,c='black',marker='s')
plt.xlabel("Optimization parameters (gravimetric)")
plt.ylabel("Normalized value")
plt.grid(True)

plt.figure()
plt.scatter(params,x0_maxvpow_1C,c='b',marker='s')
plt.scatter(params,x0_maxvene_1C,c='b',marker='^')
plt.scatter(params,x0_maxvpow_2C,c='r',marker='s')
plt.scatter(params,x0_maxvene_2C,c='r',marker='^')
plt.scatter(params,x0_ref,c='g',marker='*')
plt.scatter(params,ub_ref,c='black',marker='s')
plt.scatter(params,lb_ref,c='black',marker='s')
plt.xlabel("Optimization parameters (volumetric)")
plt.ylabel("Normalized value")
plt.grid(True)

plt.figure()
#for i in range(len(gx0_list_1C)):
#    plt.scatter(params,gx0_list_1C[i],c='b',marker='o')
plt.scatter(params,(x0_maxgpow_1C-lb_ref)/(ub_ref-lb_ref),c='b',marker='s')
plt.scatter(params,(x0_maxgene_1C-lb_ref)/(ub_ref-lb_ref),c='b',marker='^')
#for i in range(len(gx0_list_2C)):
#    plt.scatter(params,gx0_list_2C[i],c='r',marker='o')
plt.scatter(params,(x0_maxgpow_2C-lb_ref)/(ub_ref-lb_ref),c='r',marker='s')
plt.scatter(params,(x0_maxgene_2C-lb_ref)/(ub_ref-lb_ref),c='r',marker='^')

plt.scatter(params,(x0_ref-lb_ref)/(ub_ref-lb_ref),c='g',marker='*')
plt.xlabel("Optimization parameter (gavimetric)")
plt.ylabel("Normalized value")
plt.ylim(0,1)
plt.grid(True)

plt.figure()
#for i in range(len(vx0_list_1C)):
#    plt.scatter(params,vx0_list_1C[i],c='b',marker='o')
plt.scatter(params,(x0_maxvpow_1C-lb_ref)/(ub_ref-lb_ref),c='b',marker='s')
plt.scatter(params,(x0_maxvene_1C-lb_ref)/(ub_ref-lb_ref),c='b',marker='^')
#for i in range(len(vx0_list_2C)):
#    plt.scatter(params,vx0_list_2C[i],c='r',marker='o')
plt.scatter(params,(x0_maxvpow_2C-lb_ref)/(ub_ref-lb_ref),c='r',marker='s')
plt.scatter(params,(x0_maxvene_2C-lb_ref)/(ub_ref-lb_ref),c='r',marker='^')

plt.scatter(params,(x0_ref-lb_ref)/(ub_ref-lb_ref),c='g',marker='*')
plt.xlabel("Optimization parameters (volumetric)")
plt.ylabel("Normalized value")
plt.ylim(0,1)
plt.grid(True)



gene, gpow, vene, vpow, problem_maxgpow_1C = sim_battery(battery_gopt_1C[idx_maxgpow_1C], testplan_1C_json, options_json)
gene, gpow, vene, vpow, problem_maxgene_1C = sim_battery(battery_gopt_1C[idx_maxgene_1C], testplan_1C_json, options_json)
gene, gpow, vene, vpow, problem_maxvpow_1C = sim_battery(battery_vopt_1C[idx_maxvpow_1C], testplan_1C_json, options_json)
gene, gpow, vene, vpow, problem_maxvene_1C = sim_battery(battery_vopt_1C[idx_maxvene_1C], testplan_1C_json, options_json)

gene, gpow, vene, vpow, problem_maxgpow_2C = sim_battery(battery_gopt_2C[idx_maxgpow_2C], testplan_2C_json, options_json)
gene, gpow, vene, vpow, problem_maxgene_2C = sim_battery(battery_gopt_2C[idx_maxgene_2C], testplan_2C_json, options_json)
gene, gpow, vene, vpow, problem_maxvpow_2C = sim_battery(battery_vopt_2C[idx_maxvpow_2C], testplan_2C_json, options_json)
gene, gpow, vene, vpow, problem_maxvene_2C = sim_battery(battery_vopt_2C[idx_maxvene_2C], testplan_2C_json, options_json)

plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.v_list,color='b',label='1C original')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.v_list,color='r',label='2C original')
try:    
    plt.plot(problem_maxvpow_1C.t_list/problem_maxvpow_1C.t_list[-1],problem_maxvpow_1C.v_list,linestyle='--',marker='s',color='b',label='1C max vpow',markevery=10)
except:
    print('Max v pow 1C diverged')
try:
    plt.plot(problem_maxvene_1C.t_list/problem_maxvene_1C.t_list[-1],problem_maxvene_1C.v_list,linestyle='--',marker='^',color='b',label='1C max vene',markevery=10)
except:
    print('Max v ene 1C diverged')
try:
    plt.plot(problem_maxvpow_2C.t_list/problem_maxvpow_2C.t_list[-1],problem_maxvpow_2C.v_list,linestyle='--',marker='s',color='r',label='2C max vpow',markevery=10)
except:
    print('Max v pow 2C diverged')
try:
    plt.plot(problem_maxvene_2C.t_list/problem_maxvene_2C.t_list[-1],problem_maxvene_2C.v_list,linestyle='--',marker='^',color='r',label='2C max vene',markevery=10)
except:
    print('Max v ene 2C diverged')
plt.xlabel("DOD (-)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.title("Volumetric designs")
plt.legend()

plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.k_list,color='b',label='1C original')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.k_list,color='r',label='2C original')
try:    
    plt.plot(problem_maxvpow_1C.t_list/problem_maxvpow_1C.t_list[-1],problem_maxvpow_1C.k_list,linestyle='--',marker='s',color='b',label='1C max vpow',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvene_1C.t_list/problem_maxvene_1C.t_list[-1],problem_maxvene_1C.k_list,linestyle='--',marker='^',color='b',label='1C max vene',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvpow_2C.t_list/problem_maxvpow_2C.t_list[-1],problem_maxvpow_2C.k_list,linestyle='--',marker='s',color='r',label='2C max vpow',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvene_2C.t_list/problem_maxvene_2C.t_list[-1],problem_maxvene_2C.k_list,linestyle='--',marker='^',color='r',label='2C max vene',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Temperature (K)")
plt.grid(True)
plt.title("Volumetric designs")
plt.legend()

plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.v_list,color='b',label='1C original')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.v_list,color='r',label='2C original')
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.v_list,linestyle='--',marker='s',color='b',label='1C max gpow',markevery=10)
except:
    print('Max g pow 1C diverged')
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.v_list,linestyle='--',marker='^',color='b',label='1C max gene',markevery=10)
except:
    print('Max g ene 1C diverged')
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.v_list,linestyle='--',marker='s',color='r',label='2C max gpow',markevery=10)
except:
    print('Max g pow 2C diverged')
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.v_list,linestyle='--',marker='^',color='r',label='2C max gene',markevery=10)
except:
    print('Max g ene 2C diverged')
plt.xlabel("DOD (-)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.title("Gravimetric designs")
plt.legend()

plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.k_list,color='b',label='1C original')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.k_list,color='r',label='2C original')
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.k_list,linestyle='--',marker='s',color='b',label='1C max gpow',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.k_list,linestyle='--',marker='^',color='b',label='1C max gene',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.k_list,linestyle='--',marker='s',color='r',label='2C max gpow',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.k_list,linestyle='--',marker='^',color='r',label='2C max gene',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Temperature (K)")
plt.grid(True)
plt.title("Gravimetric designs")
plt.legend()


plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.xs_sur_a_list,linestyle='--',color='b')
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.xs_sur_c_list,linestyle='--',color='r')
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.xs_sur_a_list,linestyle='--',marker='s',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.xs_sur_a_list,linestyle='--',marker='^',color='b',markevery=10)    
except:
    pass
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.xs_sur_c_list,linestyle='--',marker='s',color='r',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.xs_sur_c_list,linestyle='--',marker='^',color='r',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Surface concentration (-)")
plt.grid(True)
plt.title("1C, Gravimetric")
plt.legend()

plt.figure()
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.xs_sur_a_list,linestyle='--',color='b')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.xs_sur_c_list,linestyle='--',color='r')
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.xs_sur_a_list,linestyle='--',marker='s',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.xs_sur_a_list,linestyle='--',marker='^',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.xs_sur_c_list,linestyle='--',marker='s',color='r',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.xs_sur_c_list,linestyle='--',marker='^',color='r',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Surface concentration (-)")
plt.grid(True)
plt.title("2C, Gravimetric")
plt.legend()

plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.xs_sur_a_list,linestyle='--',color='b')
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.xs_sur_c_list,linestyle='--',color='r')
try:    
    plt.plot(problem_maxvpow_1C.t_list/problem_maxvpow_1C.t_list[-1],problem_maxvpow_1C.xs_sur_a_list,linestyle='--',marker='s',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvene_1C.t_list/problem_maxvene_1C.t_list[-1],problem_maxvene_1C.xs_sur_a_list,linestyle='--',marker='^',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvpow_1C.t_list/problem_maxvpow_1C.t_list[-1],problem_maxvpow_1C.xs_sur_c_list,linestyle='--',marker='s',color='r',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvene_1C.t_list/problem_maxvene_1C.t_list[-1],problem_maxvene_1C.xs_sur_c_list,linestyle='--',marker='^',color='r',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Surface concentration (-)")
plt.grid(True)
plt.title("1C, Volumetric")
plt.legend()

plt.figure()
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.xs_sur_a_list,linestyle='--',color='b')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.xs_sur_c_list,linestyle='--',color='r')
try:    
    plt.plot(problem_maxvpow_2C.t_list/problem_maxvpow_2C.t_list[-1],problem_maxvpow_2C.xs_sur_a_list,linestyle='--',marker='s',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvene_2C.t_list/problem_maxvene_2C.t_list[-1],problem_maxvene_2C.xs_sur_a_list,linestyle='--',marker='^',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvpow_2C.t_list/problem_maxvpow_2C.t_list[-1],problem_maxvpow_2C.xs_sur_c_list,linestyle='--',marker='s',color='r',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxvene_2C.t_list/problem_maxvene_2C.t_list[-1],problem_maxvene_2C.xs_sur_c_list,linestyle='--',marker='^',color='r',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Surface concentration (-)")
plt.grid(True)
plt.title("2C, Volumetric")
plt.legend()


plt.figure()
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.ce_avg_a_list,linestyle='--',color='b')
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.ce_avg_s_list,linestyle='--',color='g')
plt.plot(problem_1C.t_list/problem_1C.t_list[-1],problem_1C.ce_avg_c_list,linestyle='--',color='r')
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.ce_avg_a_list,linestyle='--',marker='s',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.ce_avg_a_list,linestyle='--',marker='^',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.ce_avg_s_list,linestyle='--',marker='s',color='g',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.ce_avg_s_list,linestyle='--',marker='^',color='g',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgpow_1C.t_list/problem_maxgpow_1C.t_list[-1],problem_maxgpow_1C.ce_avg_c_list,linestyle='--',marker='s',color='r',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_1C.t_list/problem_maxgene_1C.t_list[-1],problem_maxgene_1C.ce_avg_c_list,linestyle='--',marker='^',color='r',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Electrolyte concentration (-)")
plt.grid(True)
plt.title("1C")
plt.legend()

plt.figure()
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.ce_avg_a_list,linestyle='--',color='b')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.ce_avg_s_list,linestyle='--',color='g')
plt.plot(problem_2C.t_list/problem_2C.t_list[-1],problem_2C.ce_avg_c_list,linestyle='--',color='r')
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.ce_avg_a_list,linestyle='--',marker='s',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.ce_avg_a_list,linestyle='--',marker='^',color='b',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.ce_avg_s_list,linestyle='--',marker='s',color='g',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.ce_avg_s_list,linestyle='--',marker='^',color='g',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgpow_2C.t_list/problem_maxgpow_2C.t_list[-1],problem_maxgpow_2C.ce_avg_c_list,linestyle='--',marker='s',color='r',markevery=10)
except:
    pass
try:    
    plt.plot(problem_maxgene_2C.t_list/problem_maxgene_2C.t_list[-1],problem_maxgene_2C.ce_avg_c_list,linestyle='--',marker='^',color='r',markevery=10)
except:
    pass
plt.xlabel("DOD (-)")
plt.ylabel("Electrolyte concentration (-)")
plt.grid(True)
plt.title("2C")
plt.legend()

plt.show()