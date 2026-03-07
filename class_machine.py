## Date: 19/02/2019
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'../..'); from fatDAE.base.basic_import import *

# Dolfin package
from dolfin import *

# User defined
import fatDAE.class_machine


def build_machine(json_1,print_level=0):

    machine = fatDAE.class_machine.Machine()

    steps_dict = {}
    steps_name = []

    for step in json_1['steps']:

        steps_name.append(step['name'])

        if step['type'] == 'Pause':
            steps_dict[step['name']] = ConstantCurrent(0., name=step['name'],print_level=print_level)

        elif step['type'] == 'CC':
            if 'value' in step:
                steps_dict[step['name']] = ConstantCurrent(step['value'],name=step['name'],print_level=print_level)
            else:
                steps_dict[step['name']] = ConstantCurrent(name=step['name'],print_level=print_level)

        elif step['type'] == 'CV':
            if 'value' in step:
                steps_dict[step['name']] = ConstantVoltage(step['value'], name=step['name'],print_level=print_level)
            else:
                steps_dict[step['name']] = ConstantVoltage(name=step['name'], print_level=print_level)

        else:
            raise NameError('Incorrect step type...')

    steps_dict['End'] = fatDAE.class_machine.End(); steps_name.append('End')

    i = 0

    for step in json_1['steps']:
        for event in step['events']:

            if event['go to'] == 'End':
                aux = fatDAE.class_machine.Transition(steps_dict[steps_name[i]], steps_dict['End'],print_level=print_level)
            else:
                if event['go to'] == 'Next':
                    aux = fatDAE.class_machine.Transition(steps_dict[steps_name[i]], steps_dict[steps_name[i+1]],print_level=print_level)
                else:
                    aux = fatDAE.class_machine.Transition(steps_dict[steps_name[i]], steps_dict[event['go to']],print_level=print_level)

            if event['type'] == 'Time':
                evt = fatDAE.class_machine.Wait(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)

            elif event['type'] == 'Voltage':
                evt = Voltage(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)

            elif event['type'] == 'Current':
                evt = Current(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)

            elif event['type'] == 'Ah':
                evt = Ah(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)
            elif event['type'] == 'Wh':
                evt = Wh(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)

            elif event['type'] == 'AhTotal':
                evt = AhTotal(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)
            elif event['type'] == 'WhTotal':
                evt = WhTotal(event['value'],event['tol_a'],event['tol_r'],print_level=print_level)

            elif event['type'] == 'MaxCycles':
                evt = fatDAE.class_machine.MaxCycles(event['value'],print_level=print_level)

            aux.add_events(evt)

            steps_dict[step['name']].add_transitions(aux)

        i += 1

    for key in steps_dict:

        machine.add_states([steps_dict[key]])

    machine.actual_state = steps_dict[json_1['steps'][0]['name']]

    return machine

def parse_json(json):

    json['actual_state'] = json['name'] + ' ' + json['actual_state']

    for state in json['states']:
        state['key'] = json['name'] + ' ' + state['key']

    for event in json['events']:
        event['key'] = json['name'] + ' ' + event['key']

    for transition in json['transitions']:

        transition['ini'] = json['name'] + ' ' + transition['ini']

        if 'father' in transition:
            transition['end'] = transition['father'] + ' ' + transition['end']
        else:
            transition['end'] = json['name'] + ' ' + transition['end']

        for i in range(len(transition['events'])):
            transition['events'][i] = json['name'] + ' ' + transition['events'][i]

    return json

def joint_json(json):
    pass


class Ah(fatDAE.class_machine.Event):

    def __init__(self, Ah, tol_a=1e-2, tol_r=1e-2,print_level=0):

        fatDAE.class_machine.Event.__init__(self,print_level=print_level)

        self.Ah = Ah

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']

        h_k = params['h_k']

        i_k = problem.get_current(x_k)

        Ah_0 = params['state_params']['Ah']
        Ah_k = params['state_params']['Ah'] + (1./3600.) * h_k * i_k

        f_0 = self.f(Ah_0)
        f_k = self.f(Ah_k)

        if abs(f_k) / (self.tol_a + self.tol_r * abs(f_k)) < 1.0:
            if self.print_level > -1:
                print('Event located')
            return x_k, h_k, True, True

        else:

            if f_0 * f_k < 0.0:
                if self.print_level > -1:
                    print('Locating event...')

                def g(t):
                    return self.f((1.0 - t) * Ah_0 + t * Ah_k)

                h_k = scipy.optimize.newton(g, 0.5) * h_k

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True

    def f(self, Ah):

        return abs(Ah) - abs(self.Ah)

class AhTotal(Ah):

    def __init__(self, Ah, tol_a=1e-2, tol_r=1e-2, print_level=0):

        fatDAE.class_machine.Event.__init__(self, print_level=print_level)

        self.Ah = Ah

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']

        h_k = params['h_k']

        i_k = problem.get_current(x_k)

        Ah_0 = params['state_params']['Ah_total'] + params['state_params']['Ah']
        Ah_k = params['state_params']['Ah_total'] + params['state_params']['Ah'] + (1./3600.) * h_k * i_k

        f_0 = self.f(Ah_0)
        f_k = self.f(Ah_k)

        if abs(f_k) / (self.tol_a + self.tol_r * abs(f_k)) < 1.0:

            if self.print_level > -1:
                print('Event located'); 
            return x_k, h_k, True, True

        else:

            if f_0 * f_k < 0.0:

                if self.print_level > -1:
                    print('Locating event...')

                def g(t):
                    return self.f((1.0 - t) * Ah_0 + t * Ah_k)

                h_k = scipy.optimize.newton(g, 0.5) * h_k

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True

    def f(self, Ah):

        return abs(Ah) - abs(self.Ah)

class Wh(fatDAE.class_machine.Event):

    def __init__(self, Wh, tol_a=1e-2, tol_r=1e-2, print_level=0):

        fatDAE.class_machine.Event.__init__(self, print_level=print_level)

        self.Wh = Wh

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']

        h_k = params['h_k']

        i_k = problem.get_current(x_k)
        v_k = problem.get_voltage(x_k)

        Wh_0 = params['Wh']
        Wh_k = params['Wh'] + (1./3600.) * h_k * i_k * v_k

        f_0 = self.f(Wh_0)
        f_k = self.f(Wh_k)

        if abs(f_k) / (self.tol_a + self.tol_r * abs(f_k)) < 1.0:

            if self.print_level > -1:
                print('Event located')
            return x_k, h_k, True, True

        else:

            if f_0 * f_k < 0.0:

                if self.print_level > -1:
                    print('Locating event...')

                def g(t):
                    return self.f((1.0 - t) * Wh_0 + t * Wh_k)

                h_k = scipy.optimize.newton(g, 0.5) * h_k

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True

    def f(self, Wh):

        return abs(Wh) - abs(self.Wh)

class WhTotal(Wh):

    def __init__(self, Wh, tol_a=1e-2, tol_r=1e-2, print_level=0):

        fatDAE.class_machine.Event.__init__(self, print_level=print_level)

        self.Wh = Wh

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']

        h_k = params['h_k']

        i_k = problem.get_current(x_k)
        v_k = problem.get_voltage(x_k)

        Wh_0 = params['state_params']['Wh_total'] + params['state_params']['Wh']
        Wh_k = params['state_params']['Wh_total'] + params['state_params']['Wh'] + (1./3600.) * h_k * i_k * v_k

        f_0 = self.f(Wh_0)
        f_k = self.f(Wh_k)

        if abs(f_k) / (self.tol_a + self.tol_r * abs(f_k)) < 1.0:
            if self.print_level > -1:
                print('Event located')
            return x_k, h_k, True, True 

        else:

            if f_0 * f_k < 0.0:

                if self.print_level > -1:
                    print('Locating event...')

                def g(t):
                    return self.f((1.0 - t) * Wh_0 + t * Wh_k)

                h_k = scipy.optimize.newton(g, 0.5) * h_k

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True

    def f(self, Wh):

        return abs(Wh) - abs(self.Wh)

class Voltage(fatDAE.class_machine.Event):
    ''' Voltage trigger.

    Attributes:
        v (:obj:`float`): Voltage.
    '''

    def __init__(self, v, tol_a = 1e-2, tol_r = 1e-2, print_level=0):

        fatDAE.class_machine.Event.__init__(self, print_level=print_level)

        self.v = v

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']

        h_k = params['h_k']

        v_0 = problem.get_voltage(x_0)
        v_k = problem.get_voltage(x_k)

        f_0 = self.f(v_0)
        f_k = self.f(v_k)

        if abs(f_k) / (self.tol_a + self.tol_r * abs(f_k)) < 1.0:
            if self.print_level > -1:
                print('Event located')
            return x_k, h_k, True, True     

        else:

            if f_0 * f_k < 0.0:

                if self.print_level > -1:
                    print('Locating event...')

                def g(t):
                    return self.f((1.0 - t) * v_0 + t * v_k)

                h_k = scipy.optimize.newton(g, 0.5) * h_k

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True

    def f(self, v):

        return v - self.v

class Current(fatDAE.class_machine.Event):
    ''' Current trigger.

    Attributes:
        i (:obj:`float`): Current.
    '''

    def __init__(self, i, tol_a = 1e-2, tol_r = 1e-2, print_level=0):

        fatDAE.class_machine.Event.__init__(self, print_level=print_level)

        self.i = i
        self.i_r = i

        self.tol_a = tol_a
        self.tol_r = tol_r

    def check(self, params):

        problem = params['problem']

        x_0 = params['x_0']
        x_k = params['x_k']

        h_k = params['h_k']

        i_0 = problem.get_current(x_0)
        i_k = problem.get_current(x_k)

        f_0 = self.f(i_0)
        f_k = self.f(i_k)

        if abs(f_k) / (self.tol_a + self.tol_r * abs(f_k)) < 1.0:

            if self.print_level > -1:
                print('Event located')
            return x_k, h_k, True, True

        else:

            if f_0 * f_k < 0.0:

                if self.print_level > -1:
                    print('Locating event...')

                def g(t):
                    return self.f((1.0 - t) * i_0 + t * i_k)

                h_k = scipy.optimize.newton(g, 0.5) * h_k

                return x_k, h_k, True, False

            else:

                return x_k, h_k, False, True

    def f(self, i):
        return abs(i) - abs(self.i_r)

class BatteryState(fatDAE.class_machine.State):

    def plot_cycle(self, n=0, save_path=None, format='svg'):

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot([t - self.data['T'][n][0] for t in self.data['T'][n]], self.data['A'][n])
        matplotlib.pyplot.xlabel('Time [s]')
        matplotlib.pyplot.ylabel('Current [A]')
        matplotlib.pyplot.title(self.name + ' Cycle ' + str(n+1))
        if save_path == None:
            pass
        else:
            matplotlib.pyplot.savefig(save_path+'current'+'_'+self.name+'_'+str(n+1)+'.'+format,format=format)

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot([t - self.data['T'][n][0] for t in self.data['T'][n]], self.data['V'][n])
        matplotlib.pyplot.xlabel('Time [s]')
        matplotlib.pyplot.ylabel('Voltage [V]')
        matplotlib.pyplot.title(self.name + ' Cycle ' + str(n+1))
        if save_path == None:
            pass
        else:
            matplotlib.pyplot.savefig(save_path+'voltage'+'_'+self.name+'_'+str(n+1)+'.'+format,format=format)

    def plot(self, save_path=None):

        for i in range(len(self.data['T'])):
            self.plot_cycle(i, save_path)

    def store(self, params):

        problem = params['problem']

        i_k = problem.get_current(params['x_k'])
        v_k = problem.get_voltage(params['x_k'])

        t_k = params['t_0']

        Ah = (1./3600) * params['h_k'] * i_k
        Wh = (1./3600) * params['h_k'] * i_k * v_k

        self.params['Ah'] += Ah
        self.params['Wh'] += Wh

        self.data['V'][-1].append(v_k);
        self.data['A'][-1].append(i_k); self.data['W'][-1].append(i_k*v_k)

        self.data['T'][-1].append(t_k)

        self.data['Ah'][-1].append(self.params['Ah'])
        self.data['Wh'][-1].append(self.params['Wh'])

    def exec_dur(self, params, accept=False):
        ''' To be executed every time :meth:`check` is called.
        '''

        if accept:
            self.store(params)

    def exec_out(self, params, reset=False):

        fatDAE.class_machine.State.exec_out(self, params)

        if self.print_level > -1:
            print('Ah_total:', self.params['Ah_total'])
            print('Wh_total:', self.params['Wh_total'])
            print('Ah:', self.params['Ah'])
            print('Wh:', self.params['Wh']); print('Time', self.t)

        if reset:
            self.params['Ah_total'] = 0.
            self.params['Wh_total'] = 0.
        else:
            self.params['Ah_total'] += self.params['Ah']
            self.params['Wh_total'] += self.params['Wh']

        self.data['Ah_cycle'].append(self.params['Ah'])
        self.data['Wh_cycle'].append(self.params['Wh'])

        self.data['Ah_total'].append(self.params['Ah_total'])
        self.data['Wh_total'].append(self.params['Wh_total'])

        self.params['Ah']=0.
        self.params['Wh']=0.

class ConstantCurrent(BatteryState):
    ''' Constant current operation mode.
    '''

    def __init__(self, i=None, name='CC',print_level=0):

        fatDAE.class_machine.State.__init__(self, name)

        self.i = i
        self.t = 0.

        self.params['Ah']=0.
        self.params['Ah_total']=0.
        self.params['Wh']=0.
        self.params['Wh_total']=0.

        self.print_level = print_level

        self.data={'V': [], 'A': [], 'W': [], 'T': [], \
                   'Ah': [], \
                   'Ah_total': [], \
                   'Ah_cycle': [], \
                   'Wh': [], \
                   'Wh_total': [], \
                   'Wh_cycle': []}

    def exec_ini(self, params):
        ''' Changes boundary condition to applied current.
        '''

        fatDAE.class_machine.State.exec_ini(self, params)

        if self.print_level > -1:
            print('Control changed.')

        self.t = params['t_0'] + params['h_k']

        if self.i == None:
            params['problem'].set_current()
        else:
            params['problem'].set_current(self.i)

        self.params['Ah']=0.
        self.params['Wh']=0.

        self.data['V'].append([]);
        self.data['A'].append([]); self.data['W'].append([])

        self.data['T'].append([])

        self.data['Ah'].append([])
        self.data['Wh'].append([])

class ConstantVoltage(BatteryState):
    ''' Constant voltage operation mode.
    '''

    def __init__(self, v=None, name='CV',print_level=0):

        fatDAE.class_machine.State.__init__(self, name)

        self.v = v
        self.t = 0.

        self.params['Ah']=0.
        self.params['Ah_total']=0.
        self.params['Wh']=0.
        self.params['Wh_total']=0.

        self.print_level = print_level

        self.data={'V': [], 'A': [], 'W': [], 'T': [], \
                   'Ah': [], \
                   'Ah_total': [], \
                   'Ah_cycle': [], \
                   'Wh': [], \
                   'Wh_total': [], \
                   'Wh_cycle': []}

    def exec_ini(self, params):
        ''' Changes boundary condition to applied voltage.
        '''

        fatDAE.class_machine.State.exec_ini(self, params)

        if self.print_level > -1:
            print('Control changed.')

        self.t = params['t_0'] + params['h_k']

        if self.v == None:
            params['problem'].set_voltage()
        else:
            params['problem'].set_voltage(self.v)

        self.params['Ah']=0.
        self.params['Wh']=0.

        self.data['V'].append([]);
        self.data['A'].append([]); self.data['W'].append([])

        self.data['T'].append([])

        self.data['Ah'].append([])
        self.data['Wh'].append([])

if __name__ == '__main__':

    with open('DCC_CV_CCC_CV.json') as data_file:
        machine_json = json.load(data_file)

    build_machine(machine_json)
