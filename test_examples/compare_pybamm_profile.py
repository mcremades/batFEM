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


cell_name = 'Ecker2015'
test_name = 'UDDS'
soc_ini=0.5
# Parser

parser = argparse.ArgumentParser(description = """batFEM""")

parser.add_argument("options_json", help = "")
parser.add_argument("-output", help="")

args = parser.parse_args()

save_path = 'results/' + str(args.output) + '/'

try:
    os.stat(save_path); shutil.rmtree(save_path); os.mkdir(save_path)
except:
    os.mkdir(save_path)

json_battery = 'json_battery/cells/cell_'+cell_name+'.json'

with open(json_battery) as data_file:
    battery_json = json.load(data_file)
with open(args.options_json) as json_data:
    options_json = json.load(json_data)

battery_json = batFEM.class_battery.parse_json(battery_json, path='json_battery/'); cell = batFEM.class_battery.Cell(battery_json, 298.15,soc_ini); 

if options_json['general properties']['model'] == 'P2D':
    problem = class_PE_PE_P2D.PE_PE_P2D(cell, options_json,save_path=save_path)
elif options_json['general properties']['model'] == 'SPM':
    problem = class_PE_PE_SPM.PE_PE_SPM(cell, options_json,save_path=save_path)
elif options_json['general properties']['model'] == 'SPME':
    problem = class_PE_PE_SPM.PE_PE_SPME(cell, options_json,save_path=save_path)
else:
    raise NameError('Model not implemented yet')

input_profile = numpy.load('test_examples/'+cell_name+'/'+test_name+'_input_profile_p2d.npz')
output_profile = numpy.load('test_examples/'+cell_name+'/'+test_name+'_output_profile_p2d.npz')

new_time = numpy.linspace(0,input_profile['time'][-1],int(input_profile['time'][-1]/1))
h = numpy.concatenate(([0],numpy.diff(new_time)))
i_app = - numpy.interp(new_time,input_profile['time'],input_profile['current'])
#print(new_time)
#quit()
#dup_idx = numpy.where(numpy.diff(input_profile['time']) >1e-8)[0]
h = numpy.concatenate(([0],numpy.diff(input_profile['time'])))
i_app = -input_profile['current']

#status = problem.solve_dcc(h=5., v_min=2.8, i_app=2.0, t_f=1800.)

status = problem.solve_profile(h, i_app, t_f=14000, store_level=options_json["output and storage"]["store level"], save_path=save_path)

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.i_list)
matplotlib.pyplot.plot(input_profile['time'],-input_profile['current'], '--')
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Current [A]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,problem.v_list)
matplotlib.pyplot.plot(input_profile['time'],input_profile['voltage'], '--')
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Voltage [V]')

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(problem.t_list,1e3*abs(problem.v_list-input_profile['voltage']), '--')
matplotlib.pyplot.xlabel('Time [s]')
matplotlib.pyplot.ylabel('Error [mV]')

print(numpy.linalg.norm(1000*(problem.v_list-input_profile['voltage'])) / numpy.sqrt(len(problem.v_list)))

matplotlib.pyplot.show()
