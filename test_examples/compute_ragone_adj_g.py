# Date: 15/02/2026
# Auth: Manuel Cremades

# -------------------------
# Imports
# -------------------------

import sys
sys.path.insert(0,'..')
from fatDAE.base.basic_import import *

import fatDAE.class_solvers
import fatDAE.class_problem

import fatDAE.dolfin_interface.class_problem
import fatDAE.dolfin_interface.class_control

import batFEM.class_battery
import batFEM.class_machine

from batFEM.PE_PE import class_PE_PE_P2D, class_PE_PE_SPM
from batFEM.ME_PE import class_ME_PE_P2D

import scipy.optimize
import matplotlib.pyplot as plt
import numpy
import copy
import json


# -------------------------
# Settings
# -------------------------

cell='Ecker2015'
method='trust-constr'
maxiter=250

Delta_gpow = 50
Npow = 1

crate = 2
Delta_gpow = Delta_gpow*crate


# -------------------------
# Load JSON
# -------------------------

with open('json_battery/cells/cell_'+cell+'_dopt_adj.json') as data_file:
    battery_json = json.load(data_file)

if crate == 1:
    with open('json_testplan/DCC_1C.json') as data_file:
        testplan_json = json.load(data_file)
elif crate == 2:
    with open('json_testplan/DCC_2C.json') as data_file:
        testplan_json = json.load(data_file)

with open('json_options/p2d_adjoint.json') as data_file:
    options_json = json.load(data_file)


# -------------------------
# Solver definition
# -------------------------

if options_json['time discretization']['embedded'] == 0:
    embedded_1 = True
    embedded_2 = False
elif options_json['time discretization']['embedded'] == 1:
    embedded_1 = False
    embedded_2 = True
else:
    raise NameError('Embedded should be 0 or 1...')


with open('../fatDAE/json_butcher/'+options_json['time discretization']['type']+'/'+options_json['time discretization']['name']+'.json') as data_file:
    butcher_json = json.load(data_file)


solver = fatDAE.class_solvers.build(
    butcher_json,
    embedded_1,
    embedded_2,
    a_tol=options_json['timestepping properties']['abs tolerance'],
    r_tol=options_json['timestepping properties']['rel tolerance'],
    h_max=options_json['timestepping properties']['max step size'],
    h_min=options_json['timestepping properties']['min step size']
)


# -------------------------
# Optimization vector tools
# -------------------------

def extract_opt_vectors(data):

    x0, lb, ub, paths = [], [], [], []

    def walk(obj, path):

        if isinstance(obj, dict):

            if obj.get("opt",0)==1:
                x0.append(obj["value"])
                lb.append(obj["lb"])
                ub.append(obj["ub"])
                paths.append(path.copy())

            for k,v in obj.items():
                walk(v,path+[k])

        elif isinstance(obj,list):
            for i,item in enumerate(obj):
                walk(item,path+[i])

    walk(data,[])
    return x0,lb,ub,paths


def apply_opt_vector(data,paths,x_new):

    for path,val in zip(paths,x_new):

        obj=data
        for key in path:
            obj=obj[key]

        obj["value"]=float(val)

    return data


x0,lb,ub,refs = extract_opt_vectors(battery_json)


# IMPORTANT FIX
opt_params = [{"path":p} for p in refs]

#bounds = list(zip(lb,ub))

print("Initial values:",x0)
print("Lower bound:",lb)
print("Upper bound:",ub)

battery_json_0 = copy.deepcopy(battery_json)


# -------------------------
# Gravimetric energy model
# -------------------------

def cost_g(x,adj):

    battery_json = apply_opt_vector(copy.deepcopy(battery_json_0),refs,x)

    battery_json = batFEM.class_battery.parse_json(battery_json,path='json_battery/')

    battery_model = batFEM.class_battery.Cell(
        battery_json,
        testplan_json['initial values']['exterior temperature'],
        testplan_json['initial values']['SOC'],
        formulation='vf',
        compute_stoichiometries=False
    )

    testplan = batFEM.class_machine.build_machine(testplan_json,print_level=-1)

   
    try:
        problem = class_PE_PE_P2D.RK_PE_PE_P2D_GravimetricEnergy(
            battery_model,
            opt_params,
            0.,
            options_json['general properties']['max simulation time'],
            options_json,
            save_path=None
        )
    except:
        cost_gene=-1e6
        cost_gpow=-1e6
        grad_gene=numpy.ones(len(x))
        grad_gpow=numpy.ones(len(x))
        return cost_gene,cost_gpow,grad_gene,grad_gpow
    
    problem.build_pvd(testplan)
    problem.setup_machine(testplan)

    if options_json['time discretization']['mode']=='adaptive':
        if adj:
            cost_gene,grad_gene = problem.solve(
                solver,
                state_machine=testplan,
                h=options_json['timestepping properties']['initial step size'],
                adp=True,
                adj=True,
                print_level=-1,
            )
        else:
            problem.solve(
                solver,
                state_machine=testplan,
                h=options_json['timestepping properties']['initial step size'],
                adp=True,
                adj=False,
                print_level=-1,
            )
            cost_gene=solver.cst
            grad_gene=numpy.ones(len(x))

    else:
        if adj:
            cost_gene,grad_gene = problem.solve(
                solver,
                state_machine=testplan,
                h=options_json['timestepping properties']['initial step size'],
                adp=False,
                adj=True,
                print_level=-1
            )
        else:
            problem.solve(
                solver,
                state_machine=testplan,
                h=options_json['timestepping properties']['initial step size'],
                adp=False,
                adj=False,
                print_level=-1
            )
            cost_gene=solver.cst
            grad_gene=numpy.random(len(x))

    if not numpy.isfinite(cost_gene):
        cost_gene=-1e6

    if numpy.any(~numpy.isfinite(grad_gene)):
        grad_gene=-numpy.random.rand(len(x))

    if len(problem.t_list) > 1:
        cost_gpow = cost_gene*3600/problem.t_list[-1]
        grad_gpow = grad_gene*3600/problem.t_list[-1]
    else:
        cost_gpow=-1e6
        grad_gpow=-numpy.random.rand(len(x))
    
    #print("Energy [Wh/L]:",cost_gene,"Power [W/L]:",cost_gpow)

    return -cost_gene,-cost_gpow,-grad_gene,-grad_gpow


# -------------------------
# Optimization functions
# -------------------------

def objective_g(x,adj):

    E,P,dE,dP = cost_g(x,adj)
    if adj:
        return -E,-dE
    else:
        return -E
    


def power_constraint_g(x,Pmin,adj):

    E,P,dE,dP = cost_g(x,adj)
    return P-Pmin,dP




# -------------------------
# Ragone optimization
# -------------------------
x_current = x0

gene_og,gpow_og,_,_ = cost_g(x_current,adj=False)

print(gene_og,gpow_og)

opt_gene=[]
opt_gpow=[]

#for i in range(Npow+1):
for i in range(Npow):

    Pmin = gpow_og #+ (Npow-i)*Delta_gpow

    print("Minimum power:",Pmin)

    cons={
        'type':'ineq',
        'fun':lambda x: power_constraint_g(x,Pmin,adj=False)[0],
        'jac':lambda x: power_constraint_g(x,Pmin,adj=True)[1],
    }
    cons={
        'type':'ineq',
        'fun':lambda x: power_constraint_g(x,Pmin,adj=False)[0]
    }
    #cons={}
    
    bounds = scipy.optimize.Bounds(lb, ub, keep_feasible=True)

    #jac=lambda x: objective_v(x,adj=True)[1],

    res = scipy.optimize.minimize(
        fun=lambda x: objective_g(x,adj=False),
        x0=x_current,
        jac=False,   
        bounds=bounds,
        constraints=cons,
        method=method,
        options={'disp':True,'verbose':3,'maxiter':maxiter}
    )

    print(res)

    #x_current = res.x

    E,P,_,_ = cost_g(res.x,adj=False)

    opt_gene.append(E)
    opt_gpow.append(P)

    battery_json_opt = copy.deepcopy(battery_json_0)
    battery_json_opt = apply_opt_vector(battery_json_0, refs, res.x)

    battery_json_opt = batFEM.class_battery.parse_json(battery_json_opt, path='json_battery/')

    opt_summary = {
    "pow_min": float(Pmin),
    "x_opt": res.x.tolist(),
    "objective": float(res.fun),
    "success": bool(res.success),
    "status": int(res.status),
    "message": res.message,
    "nfev": res.nfev,
    "nit": res.nit,
    "opt_gene": float(E),
    "opt_gpow": float(P)
    }

    with open(f"results/{cell}_dopt/{crate}C/opt_result_min_gpow_{Pmin}.json", "w") as f:
        json.dump(opt_summary, f, indent=2)

    with open(f"results/{cell}_dopt/{crate}C/battery_min_gpow_{Pmin}.json", "w") as f:
        json.dump(battery_json_opt, f, indent=2)

    print('Result:', res)


# -------------------------
# Plot Ragone
# -------------------------

plt.figure()

plt.plot(opt_gpow,opt_gene,'bx',label='Pareto')
plt.plot(gpow_og,gene_og,'ro',label='Original')

plt.xlabel("Power [W/L]")
plt.ylabel("Energy [Wh/L]")

plt.title("Ragone Plot")

plt.grid(True)
plt.legend()

plt.show()