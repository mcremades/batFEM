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

def build(battery_json, testplan_json, options_json, save_path):

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

    # Problem definition

    battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/')

    battery = batFEM.class_battery.Cell(battery_json, testplan_json['initial values']['exterior temperature'], testplan_json['initial values']['SOC']); testplan = batFEM.class_machine.build_machine(testplan_json)
 
    if battery_json['negativeElectrode']['type'] == 'PE' and battery_json['positiveElectrode']['type'] == 'PE':
        if options_json['general properties']['model'] == 'P2D':
            problem = class_PE_PE_P2D.RK_PE_PE_P2D(battery, 0., options_json['general properties']['max simulation time'], options_json, save_path=save_path)
        elif options_json['general properties']['model'] == 'SPM':
            problem = class_PE_PE_SPM.RK_PE_PE_SPM(battery, 0., options_json['general properties']['max simulation time'], options_json, save_path=save_path)
        elif options_json['general properties']['model'] == 'SPME':
            problem = class_PE_PE_SPM.RK_PE_PE_SPME(battery, 0., options_json['general properties']['max simulation time'], options_json, save_path=save_path)
        else:
            raise NameError('Model not implemented yet')
    elif battery_json['negativeElectrode']['type'] == 'ME' and battery_json['positiveElectrode']['type'] == 'PE':
        if options_json['general properties']['model'] == 'P2D':
            problem = class_ME_PE_P2D.RK_ME_PE_P2D(battery, 0., options_json['general properties']['max simulation time'], options_json, save_path=save_path)
        else:
            raise NameError('Model not implemented yet')
    elif battery_json['negativeElectrode']['type'] == 'PE' and battery_json['positiveElectrode']['type'] == 'ME':
        raise NameError('Model not implemented yet')
    else:
        raise NameError('Model not implemented yet')

    # Solve problem

    problem.build_pvd(testplan); problem.setup_machine(testplan)

    if options_json['time discretization']['mode'] == 'adaptive':
        problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=True, print_level=1)
    else:
        problem.solve(solver, state_machine=testplan, h=options_json['timestepping properties']['initial step size'], adp=False, print_level=1)

    # Post-process and plot

    #ene = abs(problem.get_ene())
    #pow = abs(problem.get_pow())

    #ene_weight = ene / battery.weight
    #ene_volume = ene / (battery.volume * 1000)
    #pow_weight = pow / battery.weight
    #pow_volume = pow / (battery.volume * 1000)

    #print('Energy [Wh/kg]:', ene_weight, 'Power [W/kg]:', pow_weight)
    #print('Energy [Wh/m3]:', ene_volume, 'Power [W/m3]:', pow_volume)

    #numpy.savetxt(save_path+'time.txt', problem.t_list)
    #numpy.savetxt(save_path+'current.txt', problem.i_list)
    #numpy.savetxt(save_path+'voltage.txt', problem.v_list)
    #numpy.savetxt(save_path+'temperature.txt', problem.k_list)

    #matplotlib.pyplot.figure()
    #for i in range(1,len(testplan.states[3].data['T']), 50):
    #    matplotlib.pyplot.plot(abs(numpy.array(testplan.states[3].data['Ah'][i])-testplan.states[3].data['Ah'][i][0]),testplan.states[3].data['V'][i],label=i)
    #matplotlib.pyplot.xlabel('Capacity [Ah]')
    #matplotlib.pyplot.ylabel('Voltage [V]')
    #matplotlib.pyplot.legend()

    #Ah_list = []
    #for i in range(1,len(testplan.states[3].data['Ah'])):
    #    Ah_list.append(abs(testplan.states[3].data['Ah'][i][-1]))

    #matplotlib.pyplot.figure()
    #matplotlib.pyplot.plot(Ah_list,'bo')
    #matplotlib.pyplot.xlabel('Cicles [-]')
    #matplotlib.pyplot.ylabel('Capacity [Ah]')
    #matplotlib.pyplot.show()
    return problem

problem = build(battery_json, testplan_json, options_json, save_path)


matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.i_list)
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Current [A]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.v_list)
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Voltage [V]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.k_list)
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Temperature [K]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.delta_film_a_list)
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Film thickness [m]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.eps_e_a_list)
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Neg. electrode porosity [-]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.c_sei_a_list)
matplotlib.pyplot.plot(problem.t_list,problem.c_lpl_a_list)
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('SEI concentration [mol/m3]')


matplotlib.pyplot.show()
