# Date: 18/01/2026
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

# Parser

parser = argparse.ArgumentParser(description = """batFEM""")

parser.add_argument("battery_json", help = "")
parser.add_argument("testplan_json", help = "")
parser.add_argument("options_json", help = "")
parser.add_argument("-output", help="")

args = parser.parse_args()

save_path = 'results/' + str(args.output) + '/'

try:
    os.stat(save_path); shutil.rmtree(save_path); os.mkdir(save_path)
except:
    os.mkdir(save_path)

with open(args.battery_json) as data_file:
    battery_json = json.load(data_file)
with open(args.testplan_json) as data_file:
    testplan_json = json.load(data_file)
with open(args.options_json) as json_data:
    options_json = json.load(json_data)


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
                visit(v, path + [k])

        elif isinstance(node, list):

            for i,v in enumerate(node):
                visit(v, path + [i])

    visit(data, [])
    return params

def set_param(data, path, new_value):
    node = data

    for k in path[:-1]:
        node = node[k]

    node[path[-1]]["value"] = new_value

    return data

def solve_problem(battery_json,testplan_json,options_json,adj,mode,print_level=1):

    opt_params = extract_opt_params(battery_json)

    battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/')

    battery = batFEM.class_battery.Cell(battery_json, testplan_json['initial values']['exterior temperature'], testplan_json['initial values']['SOC']); 

    if mode=="volumetric":
        problem = class_PE_PE_P2D.RK_PE_PE_P2D_VolumetricEnergy(battery, opt_params, 0., options_json['general properties']['max simulation time'], options_json, save_path=save_path)
    else:
        problem = class_PE_PE_P2D.RK_PE_PE_P2D_GravimetricEnergy(battery, opt_params, 0., options_json['general properties']['max simulation time'], options_json, save_path=save_path)

        
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

    # Solve vproblem
    testplan = batFEM.class_machine.build_machine(testplan_json)

    problem.build_pvd(testplan); problem.setup_machine(testplan)
    
    if options_json['time discretization']['mode'] == 'adaptive':
        if adj == True:
            cost_ene, grad_ene = problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=True, adj=True, print_level=print_level)
        else:
            problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=True, adj=False, print_level=print_level)
    else:
        if adj == True:
            cost_ene, grad_ene = problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=False, adj=True, print_level=print_level)
        else:
            problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=False, adj=False, print_level=print_level)

    if adj == True:
        return cost_ene, grad_ene
    else:
        return solver.cst

cell_name='Prada2013'

cost_vene, grad_vene = solve_problem(battery_json,testplan_json,options_json,adj=True,mode="volumetric")
numpy.savetxt('test_examples/gradients_adjoint/'+cell_name+'/grad_vene_'+options_json['time discretization']['name']+'.txt',grad_vene)

cost_gene, grad_gene = solve_problem(battery_json,testplan_json,options_json,adj=True,mode="gravimetric")
numpy.savetxt('test_examples/gradients_adjoint/'+cell_name+'/grad_gene_'+options_json['time discretization']['name']+'.txt',grad_gene)

print(cost_vene, grad_vene)
print(cost_gene, grad_gene)



opt_params = extract_opt_params(battery_json)

fd_grad_vene = numpy.zeros(len(opt_params))
fd_grad_gene = numpy.zeros(len(opt_params))
par = numpy.zeros(len(opt_params))

rtol = 1e-4
atol = 1e-12

start=time.time()

for i, p in enumerate(opt_params):

    path = p["path"]
    value = p["value"]

    eps = max(abs(value)*rtol, atol)

    print("Parameter:", path)
    print("Value:", value)
    print("Eps:", eps)

    battery_json=set_param(battery_json, path, value + eps)
    cost_vene_plus = solve_problem(
        battery_json, testplan_json, options_json,
        adj=False, mode="volumetric", print_level=0
    )
    cost_gene_plus = solve_problem(
        battery_json, testplan_json, options_json,
        adj=False, mode="gravimetric", print_level=0
    )
    battery_json=set_param(battery_json, path, value - eps)
    cost_vene_minus = solve_problem(
        battery_json, testplan_json, options_json,
        adj=False, mode="volumetric", print_level=0
    )
    cost_gene_minus = solve_problem(
        battery_json, testplan_json, options_json,
        adj=False, mode="gravimetric", print_level=0
    )
    

    fd_grad_vene[i] = (cost_vene_plus - cost_vene_minus) / (2*eps)
    fd_grad_gene[i] = (cost_gene_plus - cost_gene_minus) / (2*eps)

    par[i]=value


    battery_json=set_param(battery_json, path, value)

print(time.time()-start)
print("FD volumetric gradient:", fd_grad_vene)
print("FD gravimetric gradient:", fd_grad_gene)

numpy.savetxt('test_examples/gradients_adjoint/'+cell_name+'/fd_grad_vene_'+options_json['time discretization']['name']+'.txt',fd_grad_vene)
numpy.savetxt('test_examples/gradients_adjoint/'+cell_name+'/fd_grad_gene_'+options_json['time discretization']['name']+'.txt',fd_grad_gene)
numpy.savetxt('test_examples/gradients_adjoint/'+cell_name+'/par.txt',par)

import matplotlib.pyplot as plt

params=['$L^-$','$\\varepsilon_s^-$','$\\varepsilon_i^-$','$R_p^-$','$L^+$','$\\varepsilon_s^+$','$\\varepsilon_i^+$','$R_p^+$']


plt.figure()
plt.scatter(params, par*fd_grad_vene,  c='b',marker='o', label="Finite-difference gradient")
plt.scatter(params, par*grad_vene, c='r',marker='x', label="Adjoint gradient")
plt.ylabel("$\\theta\cdot \\nabla J(\\theta)$")
plt.title('Volumetric energy cost')
plt.legend()

plt.figure()
plt.scatter(params, par*fd_grad_gene, c='b',marker='o', label="Finite-difference gradient")
plt.scatter(params, par*grad_gene, c='r',marker='x', label="Adjoint gradient")
plt.ylabel("$\\theta\cdot \\nabla J(\\theta)$")
plt.title('Gravimetric energy cost')
plt.legend()

plt.figure()
idx = numpy.arange(len(grad_vene))
plt.plot(idx, grad_vene, "o-", label="Adjoint gradient")
plt.plot(idx, fd_grad_vene, "s--", label="Finite-difference gradient")
plt.xlabel("Control parameter")
plt.ylabel("dJ/dp")
plt.title('Volumetric energy cost')
plt.legend()

plt.figure()
idx = numpy.arange(len(grad_gene))
plt.plot(idx, grad_gene, "o-", label="Adjoint gradient")
plt.plot(idx, fd_grad_gene, "s--", label="Finite-difference gradient")
plt.xlabel("Control parameter")
plt.ylabel("dJ/dp")
plt.title('Gravimetric energy cost')
plt.legend()

plt.show()
