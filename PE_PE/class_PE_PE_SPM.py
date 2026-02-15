from dolfin import *; import numpy; import matplotlib.pyplot as plt#; from dolfin_adjoint import *

import sys; sys.path.insert(0,'..');

import batFEM.class_battery_model; import batFEM.PE_PE.class_PE_PE; import fatDAE.dolfin_interface.class_problem

class PE_PE_SPM(batFEM.PE_PE.class_PE_PE.PE_PE):

    def initial_guess(self, x=None):

        assign(self.u_0.sub(0), interpolate(self.c_s_a_ini, self.V))
        assign(self.u_0.sub(1), interpolate(self.c_s_c_ini, self.V))
        assign(self.u_0.sub(2), interpolate(self.c_s_a_ini, self.P))
        assign(self.u_0.sub(3), interpolate(self.c_s_c_ini, self.P))
    
    def build_arr(self):

        self.k_0_a = self.get_arr_a(self.k_0_a(split(self.c_s_a_1)[0]/self.c_s_a_max), self.k_0_a_Ea, self.k_0_a_Tref)
        self.k_0_c = self.get_arr_c(self.k_0_c(split(self.c_s_c_1)[0]/self.c_s_c_max), self.k_0_c_Ea, self.k_0_c_Tref)

        self.D_s_a = self.get_arr_a(self.D_s_a(split(self.c_s_a_1)[0]/self.c_s_a_max), self.D_s_a_Ea, self.D_s_a_Tref)
        self.D_s_c = self.get_arr_c(self.D_s_c(split(self.c_s_c_1)[0]/self.c_s_c_max), self.D_s_c_Ea, self.D_s_c_Tref)

        self.kappa_D_a = 2 * (self.R * self.T_a_1 / self.F) * (1 - self.t_p_a) * self.kappa_a
        self.kappa_D_s = 2 * (self.R * self.T_s_1 / self.F) * (1 - self.t_p_s) * self.kappa_s
        self.kappa_D_c = 2 * (self.R * self.T_c_1 / self.F) * (1 - self.t_p_c) * self.kappa_c

    def build_fs(self):

        P1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        P0 = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        LM = FiniteElement('R', self.mesh.ufl_cell(), 0);

        ME = MixedElement([P1, P1, LM, LM, LM, LM])

        self.V = FunctionSpace(self.mesh, P1)
        self.P = FunctionSpace(self.mesh, LM)

        self.W = FunctionSpace(self.mesh, ME)

        self.dudt = TrialFunction(self.W)

        self.u_1 = Function(self.W)
        self.u_0 = Function(self.W); self.u = TestFunction(self.W)

        self.dc_s_adt, self.dc_s_cdt, self.dc_s_sur_adt, self.dc_s_sur_cdt, self.dvdt, self.didt = split(self.dudt)

        self.c_s_a_1, self.c_s_c_1, self.c_s_sur_a_1, self.c_s_sur_c_1, self.v_1, self.i_1 = split(self.u_1)
        self.c_s_a_0, self.c_s_c_0, self.c_s_sur_a_0, self.c_s_sur_c_0, self.v_0, self.i_0 = split(self.u_0)

        self.c_s_a, self.c_s_c, self.c_s_sur_a, self.c_s_sur_c, self.v, self.i = split(self.u)

        self.c_e_a_1 = self.c_e_ini
        self.c_e_s_1 = self.c_e_ini
        self.c_e_c_1 = self.c_e_ini

        self.T_a_1 = self.T_ini
        self.T_s_1 = self.T_ini
        self.T_c_1 = self.T_ini

    def build_pvd(self, state_machine=None):
        if self.store_level > 0:
            self.c_s_a_pvd = File(self.save_path+'c_s/c_s_a.pvd'); self.c_s_a_fnc = Function(self.V)
            self.c_s_c_pvd = File(self.save_path+'c_s/c_s_c.pvd'); self.c_s_c_fnc = Function(self.V)
    
    def get_wf_c_s(self,td=True):

        if td:
            dc_s_adt = (self.c_s_a_1 - self.c_s_a_0) / self.Deltat
            dc_s_cdt = (self.c_s_c_1 - self.c_s_c_0) / self.Deltat
        else:
            dc_s_adt = self.dc_s_adt
            dc_s_cdt = self.dc_s_cdt
        
        r = SpatialCoordinate(self.mesh)

        F_c_s_a = self.R_s_a[0] ** 1 * r**2 * dc_s_adt * self.c_s_a * self.dx \
                + (1./self.R_s_a[0]) * r**2 * inner(self.D_s_a * grad(self.c_s_a_1), grad(self.c_s_a)) * self.dx \
                - self.R_s_a[0] ** 0 * (self.i_1 / (self.F * self.a_s_a[0] * self.area * self.L_a)) * self.c_s_a * self.ds(2)

        F_c_s_c = self.R_s_c[0] ** 1 * r**2 * dc_s_cdt * self.c_s_c * self.dx \
                + (1./self.R_s_c[0]) * r**2 * inner(self.D_s_c * grad(self.c_s_c_1), grad(self.c_s_c)) * self.dx \
                + self.R_s_c[0] ** 0 * (self.i_1 / (self.F * self.a_s_c[0] * self.area * self.L_c)) * self.c_s_c * self.ds(2)

        F_c_s = F_c_s_a + self.F_c_s_sur_a \
              + F_c_s_c + self.F_c_s_sur_c

        return F_c_s

    def build_wf_0(self):

        if self.solve_sei_a or self.solve_lpl_a:
            pass
        else:
            self.eps_e_a_1 = self.eps_e_a_ini

        self.eps_e_s_1 = self.eps_e_s_ini

        if self.solve_sei_c or self.solve_lpl_c:
            pass
        else:
            self.eps_e_c_1 = self.eps_e_c_ini

        self.tortuosity_e_a = self.eps_e_a_1 ** (1 - self.bruggeman_e_a)
        self.tortuosity_s_a = (1 - self.eps_e_a_1) ** (1 - self.bruggeman_s_a)

        self.tortuosity_e_s = self.eps_e_s_1 ** (1 - self.bruggeman_e_s)

        self.tortuosity_e_c = self.eps_e_c_1 ** (1 - self.bruggeman_e_c)
        self.tortuosity_s_c = (1 - self.eps_e_c_1) ** (1 - self.bruggeman_s_c)

        self.build_brg()
        self.build_arr()

        self.i_0_a = self.F * self.k_0_a * self.c_s_sur_a_1 ** self.alpha * self.c_e_a_1 ** self.alpha * (self.c_s_a_max - self.c_s_sur_a_1) ** self.alpha
        self.i_0_c = self.F * self.k_0_c * self.c_s_sur_c_1 ** self.alpha * self.c_e_c_1 ** self.alpha * (self.c_s_c_max - self.c_s_sur_c_1) ** self.alpha

        def arcsinh(x):
            return ln(x + (x**2 + 1) ** 0.5)

        F_c_s_a_0 = (self.c_s_a_1 - self.c_s_a_0) * self.c_s_a * self.dx
        F_c_s_c_0 = (self.c_s_c_1 - self.c_s_c_0) * self.c_s_c * self.dx

        self.F_c_s_sur_a = self.c_s_sur_a_1*self.c_s_sur_a*self.dx - self.c_s_a_1*self.c_s_sur_a*self.ds(2)
        self.F_c_s_sur_c = self.c_s_sur_c_1*self.c_s_sur_c*self.dx - self.c_s_c_1*self.c_s_sur_c*self.ds(2)

        F_c_s = F_c_s_a_0 + self.F_c_s_sur_a \
              + F_c_s_c_0 + self.F_c_s_sur_c

        eta_a = 2 * (self.R * self.T_a_1 / self.F) * arcsinh(- self.i_1 / (2. * self.a_s_a[0] * self.area * self.L_a * self.i_0_a))
        eta_c = 2 * (self.R * self.T_c_1 / self.F) * arcsinh(+ self.i_1 / (2. * self.a_s_c[0] * self.area * self.L_c * self.i_0_c))

        v_bat = ((self.U_c(self.c_s_sur_c_1 / self.c_s_c_max) + eta_c) * self.v * self.dx) \
              - ((self.U_a(self.c_s_sur_a_1 / self.c_s_a_max) + eta_a) * self.v * self.dx) \
              + (self.L_a / (2 * self.area * self.kappa_a)) * self.i_1 * self.v * self.dx \
              + (self.L_s / (self.area * self.kappa_s)) * self.i_1 * self.v * self.dx \
              + (self.L_c / (2 * self.area * self.kappa_c)) * self.i_1 * self.v * self.dx
        
        self.F_v = self.v_1 * self.v * self.dx - v_bat

        self.F_i = (0. + self.beta) * (self.v_1 - self.v_app) * self.i * self.dx \
                 + (1. - self.beta) * (self.i_1 - self.i_app * self.Q) * self.i * self.dx

        self.F_var_0 = F_c_s + self.F_v + self.F_i; self.J_var_0 = derivative(self.F_var_0, self.u_1)

    def build_wf_ie(self):

        F_c_s = self.get_wf_c_s(td=True)

        self.F_var_1 = F_c_s + self.F_v + self.F_i; self.J_var_1 = derivative(self.F_var_1, self.u_1)

    def build_wf_rk(self):

        F_c_s = self.get_wf_c_s(td=False)

        self.F_var = F_c_s + self.F_v + self.F_i; self.J_var = derivative(self.F_var, self.u_1)

    def store(self, t, x, level=0):

        batFEM.class_battery_model.Model.store(self, t, x)

        self.u_0.vector()[:] = x

        if level > 0:
            assign(self.c_s_a_fnc, self.u_0.sub(0))
            assign(self.c_s_c_fnc, self.u_0.sub(1))

            self.c_s_a_pvd << (self.c_s_a_fnc, t)
            self.c_s_c_pvd << (self.c_s_c_fnc, t)

    def get_voltage(self, x):
        self.u_1.vector()[:] = x
        return assemble(self.v_1*self.ds(2))

    def get_current(self, x):
        self.u_1.vector()[:] = x
        return assemble(self.i_1*self.ds(2))
    
    def get_temperature(self, x):
        self.u_1.vector()[:] = x

        return 298.15
    
    def get_xs_avg_a(self,x):
        self.u_1.vector()[:] = x
        r = SpatialCoordinate(self.mesh)
        return assemble(3*(split(self.c_s_a_1)[0]/self.c_s_a_max)*r[0]**2*self.dx)
    
    def get_xs_avg_c(self,x):
        self.u_1.vector()[:] = x
        r = SpatialCoordinate(self.mesh)
        return assemble(3*(split(self.c_s_c_1)[0]/self.c_s_c_max)*r[0]**2*self.dx)
    
    def get_xs_sur_a(self,x):
        self.u_1.vector()[:] = x
        return assemble((split(self.c_s_a_1)[0]/self.c_s_a_max)*self.ds(2))
    
    def get_xs_sur_c(self,x):
        self.u_1.vector()[:] = x
        return assemble((split(self.c_s_c_1)[0]/self.c_s_c_max)*self.ds(2))

    def get_ce_avg_a(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.c_e_a_1*self.dx)

    def get_ce_avg_s(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.c_e_s_1*self.dx)
    
    def get_ce_avg_c(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.c_e_c_1*self.dx)

class RK_PE_PE_SPM(PE_PE_SPM, fatDAE.dolfin_interface.class_problem.UFL_Problem):

    def __init__(self, cell, t_0, t_f, simulation_options, save_path='results/'):

        PE_PE_SPM.__init__(self, cell, simulation_options)

        self.setup()

        self.build_pvd()

        self.t = Expression("value", degree=1, value = t_0); self.t_v = variable(self.t)

        self.time_dependent_expresions = []

        fatDAE.dolfin_interface.class_problem.UFL_Problem.__init__(self, self.F_var, self.u_0.vector()[:], t_0, t_f)

        def boundary(x, on_boundary):
            return on_boundary

        self.set_boundary()

        self.M = self.M(self.t_0, self.x_0)

    def solve_initial(self, x):

        self.u_0.vector()[:] = x

        self.solver_0.solve()

        return numpy.array(self.u_1.vector()[:])

class PE_PE_SPME(PE_PE_SPM):

    def __init__(self, cell, simulation_options):

        PE_PE_SPM.__init__(self, cell, simulation_options)

    def initial_guess(self):

        assign(self.u_0.sub(0), interpolate(self.c_s_a_ini, self.V))
        assign(self.u_0.sub(1), interpolate(self.c_s_c_ini, self.V))
        assign(self.u_0.sub(2), interpolate(self.c_s_a_ini, self.P))
        assign(self.u_0.sub(3), interpolate(self.c_s_c_ini, self.P))

        assign(self.u_0.sub(6), interpolate(self.c_e_ini, self.V))
        assign(self.u_0.sub(7), interpolate(self.c_e_ini, self.V))
        assign(self.u_0.sub(8), interpolate(self.c_e_ini, self.V))

    def build_fs(self):

        P1 = FiniteElement('CG', self.mesh.ufl_cell(), self.FEM_order)
        P0 = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        LM = FiniteElement('R', self.mesh.ufl_cell(), 0)

        ME = MixedElement([P1, P1, LM, LM, LM, LM, P1, P1, P1, LM, LM])

        self.V = FunctionSpace(self.mesh, P1)
        self.P = FunctionSpace(self.mesh, LM)

        self.W = FunctionSpace(self.mesh, ME)

        self.dudt = TrialFunction(self.W)

        self.u_1 = Function(self.W)
        self.u_0 = Function(self.W); self.u = TestFunction(self.W)

        self.dc_s_adt, self.dc_s_cdt, self.dc_s_sur_adt, self.dc_s_sur_cdt, self.dvdt, self.didt, self.dc_e_adt, self.dc_e_sdt, self.dc_e_cdt, self.dlm_asdt, self.dlm_scdt = split(self.dudt)

        self.c_s_a_1, self.c_s_c_1, self.c_s_sur_a_1, self.c_s_sur_c_1, self.v_1, self.i_1, self.c_e_a_1, self.c_e_s_1, self.c_e_c_1, self.lm_as_1, self.lm_sc_1 = split(self.u_1)
        self.c_s_a_0, self.c_s_c_0, self.c_s_sur_a_0, self.c_s_sur_c_0, self.v_0, self.i_0, self.c_e_a_0, self.c_e_s_0, self.c_e_c_0, self.lm_as_0, self.lm_sc_0 = split(self.u_0)

        self.c_s_a, self.c_s_c, self.c_s_sur_a, self.c_s_sur_c,self.v, self.i, self.c_e_a, self.c_e_s, self.c_e_c, self.lm_as, self.lm_sc = split(self.u)

        self.T_a_1 = self.T_ini
        self.T_s_1 = self.T_ini
        self.T_c_1 = self.T_ini

    def build_pvd(self, state_machine=None):
        self.c_s_a_pvd = File(self.save_path+'c_s/c_s_a.pvd'); self.c_s_a_fnc = Function(self.V)
        self.c_s_c_pvd = File(self.save_path+'c_s/c_s_c.pvd'); self.c_s_c_fnc = Function(self.V)

        self.c_e_a_pvd = File(self.save_path+'c_e/c_e_a.pvd'); self.c_e_a_fnc = Function(self.V)
        self.c_e_s_pvd = File(self.save_path+'c_e/c_e_s.pvd'); self.c_e_s_fnc = Function(self.V)
        self.c_e_c_pvd = File(self.save_path+'c_e/c_e_c.pvd'); self.c_e_c_fnc = Function(self.V)
    
    
        
    def get_wf_c_e(self,td=True):

        if td:
            dc_e_adt = (self.c_e_a_1 - self.c_e_a_0) / self.Deltat
            dc_e_sdt = (self.c_e_s_1 - self.c_e_s_0) / self.Deltat
            dc_e_cdt = (self.c_e_c_1 - self.c_e_c_0) / self.Deltat
        else:
            dc_e_adt = self.dc_e_adt
            dc_e_sdt = self.dc_e_sdt
            dc_e_cdt = self.dc_e_cdt

        F_e_a_1 = self.eps_e_a_1 * self.L_a * dc_e_adt * self.c_e_a * self.dx + (1. / self.L_a) * self.D_e_a * inner(grad(self.c_e_a_1), grad(self.c_e_a)) * self.dx \
                + ((1. - self.t_p_a) / (self.F)) * (self.i_1 / self.area) * self.c_e_a * self.dx \
                - self.lm_as_1 * self.c_e_a * self.ds(2)

        F_e_s_1 = self.eps_e_s_1 * self.L_s * dc_e_sdt * self.c_e_s * self.dx + (1. / self.L_s) * self.D_e_s * inner(grad(self.c_e_s_1), grad(self.c_e_s)) * self.dx \
                + self.lm_as_1 * self.c_e_s * self.ds(1) \
                - self.lm_sc_1 * self.c_e_s * self.ds(2)

        F_e_c_1 = self.eps_e_c_1 * self.L_c * dc_e_cdt * self.c_e_c * self.dx + (1. / self.L_c) * self.D_e_c * inner(grad(self.c_e_c_1), grad(self.c_e_c)) * self.dx \
                - ((1. - self.t_p_c) / (self.F)) * (self.i_1 / self.area) * self.c_e_c * self.dx \
                + self.lm_sc_1 * self.c_e_c * self.ds(1)

        F_c_e = F_e_a_1 + self.F_lm_as \
              + F_e_s_1 \
              + F_e_c_1 + self.F_lm_sc
        
        return F_c_e

    def build_wf_0(self):

        if self.solve_sei_a or self.solve_lpl_a:
            pass
        else:
            self.eps_e_a_1 = self.eps_e_a_ini

        self.eps_e_s_1 = self.eps_e_s_ini

        if self.solve_sei_c or self.solve_lpl_c:
            pass
        else:
            self.eps_e_c_1 = self.eps_e_c_ini

        self.tortuosity_e_a = self.eps_e_a_1 ** (1 - self.bruggeman_e_a)
        self.tortuosity_s_a = (1 - self.eps_e_a_1) ** (1 - self.bruggeman_s_a)

        self.tortuosity_e_s = self.eps_e_s_1 ** (1 - self.bruggeman_e_s)

        self.tortuosity_e_c = self.eps_e_c_1 ** (1 - self.bruggeman_e_c)
        self.tortuosity_s_c = (1 - self.eps_e_c_1) ** (1 - self.bruggeman_s_c)

        self.build_brg()
        self.build_arr()

        self.i_0_a = self.F * self.k_0_a * self.c_s_sur_a_1 ** self.alpha * self.c_e_a_1 ** self.alpha * (self.c_s_a_max - self.c_s_sur_a_1) ** self.alpha
        self.i_0_c = self.F * self.k_0_c * self.c_s_sur_c_1 ** self.alpha * self.c_e_c_1 ** self.alpha * (self.c_s_c_max - self.c_s_sur_c_1) ** self.alpha

        def arcsinh(x):
            return ln(x + (x**2 + 1) ** 0.5)

        F_c_s_a_0 = (self.c_s_a_1 - self.c_s_a_0) * self.c_s_a * self.dx
        F_c_s_c_0 = (self.c_s_c_1 - self.c_s_c_0) * self.c_s_c * self.dx

        self.F_c_s_sur_a = self.c_s_sur_a_1*self.c_s_sur_a*self.dx - self.c_s_a_1*self.c_s_sur_a*self.ds(2)
        self.F_c_s_sur_c = self.c_s_sur_c_1*self.c_s_sur_c*self.dx - self.c_s_c_1*self.c_s_sur_c*self.ds(2)

        F_c_s = F_c_s_a_0 + self.F_c_s_sur_a \
              + F_c_s_c_0 + self.F_c_s_sur_c

        F_e_a_0 = (self.c_e_a_1 - self.c_e_a_0) * self.c_e_a * self.dx + self.lm_as_1 * self.c_e_a * self.ds(2)
        F_e_s_0 = (self.c_e_s_1 - self.c_e_s_0) * self.c_e_s * self.dx - self.lm_as_1 * self.c_e_s * self.ds(1) + self.lm_sc_1 * self.c_e_s * self.ds(2)
        F_e_c_0 = (self.c_e_c_1 - self.c_e_c_0) * self.c_e_c * self.dx - self.lm_sc_1 * self.c_e_c * self.ds(1)

        self.F_lm_as = self.lm_as * self.c_e_a_1 * self.ds(2) - self.lm_as * self.c_e_s_1 * self.ds(1)
        self.F_lm_sc = self.lm_sc * self.c_e_s_1 * self.ds(2) - self.lm_sc * self.c_e_c_1 * self.ds(1)

        F_c_e = F_e_a_0 + self.F_lm_as \
              + F_e_s_0 \
              + F_e_c_0 + self.F_lm_sc
        
        eta_a = 2 * (self.R * self.T_a_1 / self.F) * arcsinh(- self.i_1 / (2. * self.a_s_a[0] * self.area * self.L_a * self.i_0_a))
        eta_c = 2 * (self.R * self.T_c_1 / self.F) * arcsinh(+ self.i_1 / (2. * self.a_s_c[0] * self.area * self.L_c * self.i_0_c))

        v_bat = ((self.U_c(self.c_s_sur_c_1 / self.c_s_c_max) + eta_c) * self.v * self.dx + 2 * (self.R*self.T_c_1/self.F) * (1. - self.t_p_c) * ln(self.c_e_c_1) * self.v * self.ds(2)) \
              - ((self.U_a(self.c_s_sur_a_1 / self.c_s_a_max) + eta_a) * self.v * self.dx + 2 * (self.R*self.T_a_1/self.F) * (1. - self.t_p_a) * ln(self.c_e_a_1) * self.v * self.ds(1)) \
              + (self.L_a / (2 * self.area * self.kappa_a)) * self.i_1 * self.v * self.dx \
              + (self.L_s / (self.area * self.kappa_s)) * self.i_1 * self.v * self.dx \
              + (self.L_c / (2 * self.area * self.kappa_c)) * self.i_1 * self.v * self.dx
        
        self.F_v = self.v_1 * self.v * self.dx - v_bat

        self.F_i = (0. + self.beta) * (self.v_1 - self.v_app) * self.i * self.dx \
                 + (1. - self.beta) * (self.i_1 - self.i_app * self.Q) * self.i * self.dx

        self.F_var_0 = F_c_s + self.F_v + self.F_i \
                     + F_c_e

        self.J_var_0 = derivative(self.F_var_0, self.u_1)

    def build_wf_ie(self):
        
        
        # c_s
        F_c_s = self.get_wf_c_s(td=True)

        # c_e
        F_c_e = self.get_wf_c_e(td=True)

        self.F_var_1 = F_c_s + self.F_v + self.F_i \
                     + F_c_e

        self.J_var_1 = derivative(self.F_var_1, self.u_1)

    def build_wf_rk(self):

        # c_s
        F_c_s = self.get_wf_c_s(td=False)

        # c_e
        F_c_e = self.get_wf_c_e(td=False)

        self.F_var = F_c_s + self.F_v + self.F_i \
                   + F_c_e
        
        self.J_var = derivative(self.F_var, self.u_1)

    def store(self, t, x, level=0, state_name=None, state_number=0):

        batFEM.class_battery_model.Model.store(self, t, x, state_name, state_number)

        self.u_0.vector()[:] = x

        if level > 0:
            assign(self.c_s_a_fnc, self.u_0.sub(0))
            assign(self.c_s_c_fnc, self.u_0.sub(1))

            assign(self.c_e_a_fnc, self.u_0.sub(6))
            assign(self.c_e_s_fnc, self.u_0.sub(7))
            assign(self.c_e_c_fnc, self.u_0.sub(8))

            self.c_s_a_pvd << (self.c_s_a_fnc, t)
            self.c_s_c_pvd << (self.c_s_c_fnc, t)

            self.c_e_a_pvd << (self.c_e_a_fnc, t)
            self.c_e_s_pvd << (self.c_e_s_fnc, t)
            self.c_e_c_pvd << (self.c_e_c_fnc, t)

class RK_PE_PE_SPME(PE_PE_SPME, fatDAE.dolfin_interface.class_problem.UFL_Problem):

    def __init__(self, cell, t_0, t_f, simulation_options, save_path='reuslts/'):

        PE_PE_SPME.__init__(self, cell, simulation_options)

        self.setup()

        self.store_level = simulation_options['output and storage']['store level']; self.save_path=save_path

        self.build_pvd()

        self.t = Expression("value", degree=1, value = t_0); self.t_v = variable(self.t)

        self.time_dependent_expresions = []

        fatDAE.dolfin_interface.class_problem.UFL_Problem.__init__(self, self.F_var, self.u_0.vector()[:], t_0, t_f)

        def boundary(x, on_boundary):
            return on_boundary

        self.set_boundary()

        self.M = self.M(self.t_0, self.x_0)

    def solve_initial(self, x):

        self.u_0.vector()[:] = x

        self.solver_0.solve()

        return numpy.array(self.u_1.vector()[:])

class PE_PE_SPMT(PE_PE_SPM):
    pass

class RK_PE_PE_SPMT(PE_PE_SPMT, fatDAE.dolfin_interface.class_problem.UFL_Problem):
    pass

if __name__ == '__main__':

    import os; import shutil; import json; import argparse; import batFEM.class_battery

    parser = argparse.ArgumentParser(description = """batFEM""")

    parser.add_argument("battery_json", help = "")
    parser.add_argument("options_json", help = "")
    parser.add_argument("-output", help="")

    args = parser.parse_args()

    save_path = 'results/' + str(args.output) + '/'

    try:
        os.stat(save_path); shutil.rmtree(save_path); os.mkdir(save_path)
    except:
        os.mkdir(save_path)

    with open(args.battery_json) as json_data:
        json_battery = json.load(json_data)
    with open(args.options_json) as json_data:
        json_options = json.load(json_data)

    json_battery = batFEM.class_battery.parse_json(json_battery); cell = batFEM.class_battery.Cell(json_battery, temperature=298.15, SOC=1.0)

    if json_options['general properties']['model'] == 'SPM':
        problem = PE_PE_SPM(cell, json_options)
    else:
        if json_options['general properties']['model'] == 'SPME':
            problem = PE_PE_SPME(cell, json_options)
        else:
            raise NameError('Unknown type of model')


    print('Weight [kg]:', cell.weight)
    print('Volume [L]:', cell.volume * 1000)

    status = problem.solve_dcc(h=5., v_min=3.0, i_app=2.0, t_f=1800., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=0)
    status = problem.solve_ccc(h=10., v_max=3.6, i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    status = problem.solve_ccc(h=5., v_max=3.6, i_app=1.0, t_f=3600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    status = problem.solve_dcc(h=10., v_min=3.0, i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)

    ene = abs(problem.get_ene())
    pow = abs(problem.get_pow())

    ene_weight = ene / cell.weight
    ene_volume = ene / (cell.volume * 1000)
    pow_weight = pow / cell.weight
    pow_volume = pow / (cell.volume * 1000)

    print('Energy [Wh/kg]:', ene_weight, 'Power [W/kg]:', pow_weight)
    print('Energy [Wh/m3]:', ene_volume, 'Power [W/m3]:', pow_volume)

    numpy.savetxt(save_path+'time.txt', problem.t_list)
    numpy.savetxt(save_path+'current.txt', problem.i_list)
    numpy.savetxt(save_path+'voltage.txt', problem.v_list)

    plt.figure()
    plt.plot(problem.t_list,problem.i_list)
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')

    plt.figure()
    plt.plot(problem.t_list,problem.v_list)
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    
    plt.figure()
    plt.plot(problem.t_list,problem.k_list)
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')

    plt.figure()
    plt.plot(problem.t_list,problem.xs_avg_a_list,'b-')
    plt.plot(problem.t_list,problem.xs_sur_a_list,'b--')
    plt.plot(problem.t_list,problem.xs_avg_c_list,'r-')
    plt.plot(problem.t_list,problem.xs_sur_c_list,'r--')
    plt.xlabel('Time [s]')
    plt.ylabel('Avg. Degree of Lithiation [-]')

    plt.figure()
    plt.plot(problem.t_list,problem.ce_avg_a_list,'b-')
    plt.plot(problem.t_list,problem.ce_avg_s_list,'g-')
    plt.plot(problem.t_list,problem.ce_avg_c_list,'r-')
    plt.xlabel('Time [s]')
    plt.ylabel('Avg. Electrolyte concentration [mol/m^3]')

    plt.show()
