from dolfin import *; import numpy; import matplotlib.pyplot as plt; import json; import scipy; import scipy.sparse.linalg; import os #; from dolfin_adjoint import *

set_log_active(False)

def get_interpolation(xv, type, opts):

    x_array = numpy.zeros(len(xv))
    v_array = numpy.zeros(len(xv))

    for j in range(len(xv)):
        x_array[j] = xv[j][0]
        v_array[j] = xv[j][1]

    if type == 'CubicSpline':
        c = scipy.interpolate.CubicSpline(x_array, v_array, bc_type=opts['bc_type']).c
        k = len(c) - 1
    elif type == 'PchipInterpolator':
        c = scipy.interpolate.PchipInterpolator(x_array, v_array).c
        k = len(c) - 1
    else:
        raise NameError('Unknown type of interpolation')

    def f(y):

        S_list = []

        for j in range(len(x_array)-1):
            S_list.append(sum(c[m, j] * (y - x_array[j])**(k-m) for m in range(k+1)))

        fy = 0

        for j in range(len(S_list)):
            fy += S_list[j] * conditional(ge(y,x_array[j]), conditional(lt(y,x_array[j+1]), 1, 0), 0)

        fy += S_list[+0]*conditional(lt(y,x_array[+0]), 1, 0)
        fy += S_list[-1]*conditional(ge(y,x_array[-1]), 1, 0)

        return fy

    return f

class Model:

    def setup(self,i_app=0):

        self.build_fs()

        self.Deltat = Expression('value', degree=0, value=1.0); self.beta = Expression('value', degree=0, value=0)

        self.v_app = Expression('value', degree=0, value=0)
        self.i_app = Expression('value', degree=0, value=i_app)

        self.initial_guess()

        self.u_1.assign(self.u_0)

        self.build_wf_0()

        self.problem_0 = NonlinearVariationalProblem(self.F_var_0,  self.u_1, [], self.J_var_0); self.solver_0 = NonlinearVariationalSolver(self.problem_0);  prm = self.solver_0.parameters

        prm["newton_solver"]["absolute_tolerance"] = 1E-6
        prm["newton_solver"]["relative_tolerance"] = 1E-6
        prm["newton_solver"]["maximum_iterations"] = 200; prm["newton_solver"]["relaxation_parameter"] = 0.9

        self.solver_0.solve()

        self.u_0.assign(self.u_1)

        self.build_wf_ie()

        self.problem_1 = NonlinearVariationalProblem(self.F_var_1,  self.u_1, [], self.J_var_1); self.solver_1 = NonlinearVariationalSolver(self.problem_1)

        self.build_wf_rk()

    def build_mesh(self):

        self.mesh = UnitIntervalMesh(self.N_x)
        self.mesh_plot = UnitIntervalMesh(100)

        boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1); boundaries.set_all(0)

        west = West()
        east = East()

        west.mark(boundaries, 1)
        east.mark(boundaries, 2)

        # Measures

        self.dx = Measure('dx', domain=self.mesh)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundaries)

    def build_sgm(self):

        self.M_c_s_a = numpy.zeros((self.SGM_order_a, self.SGM_order_a))
        self.K_c_s_a = numpy.zeros((self.SGM_order_a, self.SGM_order_a))

        self.P_c_s_a = numpy.zeros(self.SGM_order_a)
        self.Q_c_s_a = numpy.zeros(self.SGM_order_a)

        for n in range(self.SGM_order_a):

            for m in range(self.SGM_order_a):

                L_n = numpy.zeros(2*self.SGM_order_a, dtype=int); L_n[2*n] = 1
                L_m = numpy.zeros(2*self.SGM_order_a, dtype=int); L_m[2*m] = 1

                D_n = numpy.polynomial.legendre.legder(L_n)
                D_m = numpy.polynomial.legendre.legder(L_m)

                L_nx = numpy.polynomial.legendre.legmulx(L_n)
                L_mx = numpy.polynomial.legendre.legmulx(L_m)
                L_nx2 = numpy.polynomial.legendre.legmulx(L_nx)

                D_nx = numpy.polynomial.legendre.legmulx(D_n)
                D_mx = numpy.polynomial.legendre.legmulx(D_m)

                self.M_c_s_a[n, m] = 0.5 * numpy.polynomial.legendre.legval(1.0, numpy.polynomial.legendre.legint(numpy.polynomial.legendre.legmul(L_nx, L_mx), lbnd=-1))
                self.K_c_s_a[n, m] = 0.5 * numpy.polynomial.legendre.legval(1.0, numpy.polynomial.legendre.legint(numpy.polynomial.legendre.legmul(D_nx, D_mx), lbnd=-1))

                if self.M_c_s_a[n, m] < 1.e-16:
                    self.M_c_s_a[n, m] = 0.

                if self.K_c_s_a[n, m] < 1.e-16:
                    self.K_c_s_a[n, m] = 0.

            self.P_c_s_a[n] = numpy.polynomial.legendre.legval(1.0, L_n)
            self.Q_c_s_a[n] = 3*numpy.polynomial.legendre.legval(1.0, numpy.polynomial.legendre.legint(L_nx2, lbnd=0))

        self.M_c_s_c = numpy.zeros((self.SGM_order_c, self.SGM_order_c))
        self.K_c_s_c = numpy.zeros((self.SGM_order_c, self.SGM_order_c))

        self.P_c_s_c = numpy.zeros(self.SGM_order_c)
        self.Q_c_s_c = numpy.zeros(self.SGM_order_c)

        for n in range(self.SGM_order_c):

            for m in range(self.SGM_order_c):

                L_n = numpy.zeros(2*self.SGM_order_c, dtype=int); L_n[2*n] = 1
                L_m = numpy.zeros(2*self.SGM_order_c, dtype=int); L_m[2*m] = 1

                D_n = numpy.polynomial.legendre.legder(L_n)
                D_m = numpy.polynomial.legendre.legder(L_m)

                L_nx = numpy.polynomial.legendre.legmulx(L_n)
                L_mx = numpy.polynomial.legendre.legmulx(L_m)
                L_nx2 = numpy.polynomial.legendre.legmulx(L_nx)

                D_nx = numpy.polynomial.legendre.legmulx(D_n)
                D_mx = numpy.polynomial.legendre.legmulx(D_m)

                self.M_c_s_c[n, m] = 0.5 * numpy.polynomial.legendre.legval(1.0, numpy.polynomial.legendre.legint(numpy.polynomial.legendre.legmul(L_nx, L_mx), lbnd=-1))
                self.K_c_s_c[n, m] = 0.5 * numpy.polynomial.legendre.legval(1.0, numpy.polynomial.legendre.legint(numpy.polynomial.legendre.legmul(D_nx, D_mx), lbnd=-1))

                if self.M_c_s_c[n, m] < 1.e-16:
                    self.M_c_s_c[n, m] = 0.

                if self.K_c_s_c[n, m] < 1.e-16:
                    self.K_c_s_c[n, m] = 0.

            self.P_c_s_c[n] = numpy.polynomial.legendre.legval(1.0, L_n)
            self.Q_c_s_c[n] = 3*numpy.polynomial.legendre.legval(1.0, numpy.polynomial.legendre.legint(L_nx2, lbnd=0))

    def setup_machine(self, state_machine):

        self.t_dict={}
        self.i_dict={}
        self.v_dict={}
        self.k_dict={}
        for state in state_machine.states:
            self.t_dict[state.name] = []
            self.i_dict[state.name] = []
            self.v_dict[state.name] = []
            self.k_dict[state.name] = []
            for i in range(1000):
                self.t_dict[state.name].append([])
                self.i_dict[state.name].append([])
                self.v_dict[state.name].append([])
                self.k_dict[state.name].append([])

    def store(self, t, x, state_name=None, state_number=0):
        
        if state_name is not None:
            self.t_dict[state_name][state_number].append(t); 
        self.t_list.append(t)
        
        
        i = self.get_current(x)
        if state_name is not None:
            self.i_dict[state_name][state_number].append(i)
        self.i_list.append(i)

        v = self.get_voltage(x)
        if state_name is not None:
            self.v_dict[state_name][state_number].append(v)
        self.v_list.append(v)

        k = self.get_temperature(x)
        if state_name is not None:
            self.k_dict[state_name][state_number].append(k)
        self.k_list.append(k)

        xs_avg_a=self.get_xs_avg_a(x)
        xs_avg_c=self.get_xs_avg_c(x)
        self.xs_avg_a_list.append(xs_avg_a)
        self.xs_avg_c_list.append(xs_avg_c)

        xs_sur_a=self.get_xs_sur_a(x)
        xs_sur_c=self.get_xs_sur_c(x)
        self.xs_sur_a_list.append(xs_sur_a)
        self.xs_sur_c_list.append(xs_sur_c)

        ce_avg_a=self.get_ce_avg_a(x)
        ce_avg_s=self.get_ce_avg_s(x)
        ce_avg_c=self.get_ce_avg_c(x)
        self.ce_avg_a_list.append(ce_avg_a)
        self.ce_avg_s_list.append(ce_avg_s)
        self.ce_avg_c_list.append(ce_avg_c)

    def get_ene(self):
        return numpy.trapezoid(numpy.array(self.i_list)*numpy.array(self.v_list), x=self.t_list) / 3600.

    def get_pow(self):
        return numpy.trapezoid(numpy.array(self.i_list)*numpy.array(self.v_list), x=self.t_list) / self.t_list[-1]

    def set_voltage(self, v=None):

        self.beta.value = 1

        if v == None:
            self.v_app.value = self.get_voltage(self.u_0.vector()[:])
        else:
            self.v_app.value = v

    def set_current(self, i=None):

        self.beta.value = 0

        if i == None:
            self.i_app.value = self.get_current(self.u_0.vector()[:])
        else:
            self.i_app.value = i

    def get_brug_e_a(self, x):
        return x * self.eps_e_a_1 / self.tortuosity_e_a
    def get_brug_e_s(self, x):
        return x * self.eps_e_s_1 / self.tortuosity_e_s
    def get_brug_e_c(self, x):
        return x * self.eps_e_c_1 / self.tortuosity_e_c

    def get_brug_s_a(self, x):
        return x * (1 - self.eps_e_a_1) / self.tortuosity_s_a
    def get_brug_s_c(self, x):
        return x * (1 - self.eps_e_c_1) / self.tortuosity_s_c

    def get_arr_a(self, x, Ea, T_ref):
        return x * exp((Ea / self.R)*(1/T_ref - 1/self.T_a_1))
    def get_arr_s(self, x, Ea, T_ref):
        return x * exp((Ea / self.R)*(1/T_ref - 1/self.T_s_1))
    def get_arr_c(self, x, Ea, T_ref):
        return x * exp((Ea / self.R)*(1/T_ref - 1/self.T_c_1))

    def plot(self):

        plt.figure()
        plt.plot(self.t_list, self.i_list)
        plt.xlabel('Time [s]')
        plt.ylabel('Current [A]')

        plt.figure()
        plt.plot(self.t_list, self.v_list)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')

    def write(self, write_level, state_name=None, state_number=0):

        numpy.savetxt(os.path.join(self.save_path,'time.txt'), self.t_list)
        numpy.savetxt(os.path.join(self.save_path,'current.txt'), self.i_list)
        numpy.savetxt(os.path.join(self.save_path,'voltage.txt'), self.v_list)
        numpy.savetxt(os.path.join(self.save_path,'temperature.txt'), self.k_list)

        numpy.savetxt(os.path.join(self.save_path,'xs_avg_a.txt'), self.xs_avg_a_list)
        numpy.savetxt(os.path.join(self.save_path,'xs_avg_c.txt'), self.xs_avg_c_list)
        numpy.savetxt(os.path.join(self.save_path,'xs_sur_a.txt'), self.xs_sur_a_list)
        numpy.savetxt(os.path.join(self.save_path,'xs_sur_c.txt'), self.xs_sur_c_list)
        
        #if write_level > 0:
        #    dir_path = os.path.join(self.save_path, state_name, str(state_number))
        #    os.makedirs(dir_path, exist_ok=True)
        
            

        #    numpy.savetxt(os.path.join(dir_path,'current.txt'), self.i_dict[state_name][state_number])
        #    numpy.savetxt(os.path.join(dir_path,'voltage.txt'), self.v_dict[state_name][state_number])

        #    numpy.savetxt(os.path.join(dir_path,'temperature.txt'), self.k_dict[state_name][state_number])

        #    numpy.savetxt(os.path.join(dir_path,'time.txt'), self.t_dict[state_name][state_number])

        #    numpy.savetxt(os.path.join(dir_path,'current.txt'), self.i_dict[state_name][state_number])
        #    numpy.savetxt(os.path.join(dir_path,'voltage.txt'), self.v_dict[state_name][state_number])

        #    numpy.savetxt(os.path.join(dir_path,'temperature.txt'), self.k_dict[state_name][state_number])

            #self.t_dict[state_name][state_number] = []

            #self.i_dict[state_name][state_number] = []
            #self.v_dict[state_name][state_number] = []

            #self.k_dict[state_name][state_number] = []

    def tstep_ie(self, h=10, i_app = 30.0):

        self.Deltat.value = h; self.i_app.value = - i_app

        prm = self.solver_1.parameters

        prm["newton_solver"]["absolute_tolerance"] = 1E-8
        prm["newton_solver"]["relative_tolerance"] = 1E-6

        try:
            self.solver_1.solve(); self.u_0.assign(self.u_1)
        except:
            print('Error in solving')
            return 0
    
    def solve_profile(self, h, i_app, t_f, store_level=0, write_level=0, save_path='results/', already_setup=False):
        
        self.save_path = save_path

        if not already_setup:
            self.setup(i_app[0]/self.Q)
            t_0 = 0.

            self.build_pvd()
        else:
            t_0 = self.t_list[-1]
            t_f = t_0 + t_f

        import time; start = time.time()

        t = t_0

        self.store(t, self.u_1.vector()[:], store_level)

        print('Solving...')

        for i in range(1,len(i_app)):

            if t > t_f:
                return 2

            self.Deltat.value = h[i]; self.i_app.value = i_app[i]/self.Q
            
            prm = self.solver_1.parameters

            prm["newton_solver"]["absolute_tolerance"] = 1E-6
            prm["newton_solver"]["relative_tolerance"] = 1E-6

            try:
                self.solver_1.solve(); self.u_0.assign(self.u_1)
            except:
                print('Error in solving')
                return 0

            t += h[i]

            self.store(t, self.u_1.vector()[:], store_level)

        print('Elapsed time: ', time.time() - start); print('Steps: ', len(self.t_list))

        self.write(write_level, state_name=None, state_number=0)

        return 1
        

    def solve_dcc(self, h=10, v_min=3.0, i_app=1.0, t_f=3600, store_level=0, save_path='results/', already_setup=False):

        self.save_path = save_path

        if not already_setup:
            self.setup()
            t_0 = 0.

            self.build_pvd()
        else:
            t_0 = self.t_list[-1]
            t_f = t_0 + t_f

        self.i_app.value = 0
        self.v_app.value = 0

        self.beta.value = 0

        import time; start = time.time()

        t = t_0

        self.store(t, self.u_1.vector()[:], store_level)

        print('Solving...')

        while self.get_voltage(self.u_1.vector()[:]) > v_min:

            if t > t_f:
                return 2

            self.Deltat.value = h; self.i_app.value = - i_app

            prm = self.solver_1.parameters

            prm["newton_solver"]["absolute_tolerance"] = 1E-6
            prm["newton_solver"]["relative_tolerance"] = 1E-6

            try:
                self.solver_1.solve(); self.u_0.assign(self.u_1)
            except:
                print('Error in solving')
                return 0

            t += h

            self.store(t, self.u_1.vector()[:], store_level)

        print('Elapsed time: ', time.time() - start); print('Steps: ', len(self.t_list))

        return 1

    def solve_ccc(self, h=10, v_max=4.2, i_app=1.0, t_f=3600, store_level=0, save_path='results/', already_setup=False):

        self.save_path = save_path

        if not already_setup:
            self.setup()
            t_0 = 0.

            self.build_pvd()
        else:
            t_0 = self.t_list[-1]
            t_f = t_0 + t_f

        self.i_app.value = 0
        self.v_app.value = 0

        self.beta.value = 0

        import time; start = time.time()

        t = t_0

        self.store(t, self.u_1.vector()[:], store_level)

        print('Solving...')

        while self.get_voltage(self.u_1.vector()[:]) < v_max:

            if t > t_f:
                return 2

            self.Deltat.value = h; self.i_app.value = +i_app

            prm = self.solver_1.parameters

            prm["newton_solver"]["absolute_tolerance"] = 1E-6
            prm["newton_solver"]["relative_tolerance"] = 1E-6

            try:
                self.solver_1.solve(); self.u_0.assign(self.u_1)
            except:
                print('Error in solving')
                return 0

            t += h

            self.store(t, self.u_1.vector()[:], store_level)

        print('Elapsed time: ', time.time() - start); print('Steps: ', len(self.t_list))

        return 1
class West(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.0 + DOLFIN_EPS and on_boundary

class East(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS and on_boundary
