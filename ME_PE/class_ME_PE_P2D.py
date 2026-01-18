from dolfin import *; import numpy; import matplotlib.pyplot as plt#; from dolfin_adjoint import *

import sys; sys.path.insert(0,'..');

import batFEM.class_battery_model; import batFEM.ME_PE.class_ME_PE; import fatDAE.dolfin_interface.class_problem

class ME_PE_P2D(batFEM.ME_PE.class_ME_PE.ME_PE):

    def initial_guess(self):

        assign(self.u_0.sub(0).sub(0), interpolate(self.c_e_ini, self.V))
        assign(self.u_0.sub(0).sub(1), interpolate(self.c_e_ini, self.V))

        assign(self.u_0.sub(1).sub(0), interpolate(self.c_s_c_ini, self.V))

        assign(self.u_0.sub(3), project(self.U_c(self.c_s_c_ini/self.c_s_c_max), self.V))

        assign(self.u_0.sub(5).sub(0), interpolate(self.T_ini, self.V))
        assign(self.u_0.sub(5).sub(1), interpolate(self.T_ini, self.V))

    def build_fs(self):

        P1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        P0 = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        LM = FiniteElement('R', self.mesh.ufl_cell(), 0);

        E_c_e = [P1, P1, LM]

        E_c_s_c = []

        E_j_Li_c = []

        for j in range(len(self.eps_s_c)):
            for i in range(self.SGM_order_c):
                E_c_s_c.append(P1)
            E_j_Li_c.append(P1)

        E_phi_e = [P1, P1, LM]
        E_phi_s = P1

        E_T = [P1, P1, LM]

        ME = MixedElement([E_c_e, E_c_s_c, E_phi_e, E_phi_s, E_j_Li_c, E_T, LM])

        self.V = FunctionSpace(self.mesh, P1)
        self.P = FunctionSpace(self.mesh, LM)

        self.W = FunctionSpace(self.mesh, ME)

        self.dudt = TrialFunction(self.W)

        self.u_1 = Function(self.W)
        self.u_0 = Function(self.W); self.u = TestFunction(self.W)

        self.dc_edt, self.dc_s_cdt, self.dphi_edt, self.dphi_s_cdt, self.dj_Li_cdt, self.dTdt, self.dlm_appdt = split(self.dudt)

        self.c_e_1, self.c_s_c_1, self.phi_e_1, self.phi_s_c_1, self.j_Li_c_1, self.T_1, self.lm_app_1 = split(self.u_1)
        self.c_e_0, self.c_s_c_0, self.phi_e_0, self.phi_s_c_0, self.j_Li_c_0, self.T_0, self.lm_app_0 = split(self.u_0)

        self.c_e, self.c_s_c, self.phi_e, self.phi_s_c, self.j_Li_c, self.T, self.lm_app = split(self.u)

        self.dc_e_sdt, self.dc_e_cdt, self.dlm_c_e_scdt = split(self.dc_edt)

        self.c_e_s_1, self.c_e_c_1, self.lm_c_e_sc_1 = split(self.c_e_1)
        self.c_e_s_0, self.c_e_c_0, self.lm_c_e_sc_0 = split(self.c_e_0)

        self.c_e_s, self.c_e_c, self.lm_c_e_sc = split(self.c_e)

        self.phi_e_s_1, self.phi_e_c_1, self.lm_phi_e_sc_1 = split(self.phi_e_1)
        self.phi_e_s_0, self.phi_e_c_0, self.lm_phi_e_sc_0 = split(self.phi_e_0)

        self.phi_e_s, self.phi_e_c, self.lm_phi_e_sc = split(self.phi_e)

        self.dT_sdt, self.dT_cdt, self.dlm_T_scdt = split(self.dTdt)

        self.T_s_1, self.T_c_1, self.lm_T_sc_1 = split(self.T_1)
        self.T_s_0, self.T_c_0, self.lm_T_sc_0 = split(self.T_0)

        self.T_s, self.T_c, self.lm_T_sc = split(self.T)

    def build_pvd(self):

        self.c_e_s_pvd = File(self.save_path+'c_e/c_e_s.pvd'); self.c_e_s_fnc = Function(self.V)
        self.c_e_c_pvd = File(self.save_path+'c_e/c_e_c.pvd'); self.c_e_c_fnc = Function(self.V)

        self.phi_e_s_pvd = File(self.save_path+'phi_e/phi_e_s.pvd'); self.phi_e_s_fnc = Function(self.V)
        self.phi_e_c_pvd = File(self.save_path+'phi_e/phi_e_c.pvd'); self.phi_e_c_fnc = Function(self.V)

        self.phi_s_c_pvd = File(self.save_path+'phi_s/phi_s_c.pvd'); self.phi_s_c_fnc = Function(self.V)

        self.T_s_pvd = File(self.save_path+'T/T_s.pvd'); self.T_s_fnc = Function(self.V)
        self.T_c_pvd = File(self.save_path+'T/T_c.pvd'); self.T_c_fnc = Function(self.V)

    def build_wf_0(self):

        self.c_s_c_sur = []

        for j in range(len(self.eps_s_c)):
            c_s_c_sur = split(self.c_s_c_1)[0]

            for i in range(1, self.SGM_order_c):
                c_s_c_sur += split(self.c_s_c_1)[i] * self.P_c_s_c[i]
            self.c_s_c_sur.append(c_s_c_sur)

        self.build_brg()
        self.build_arr()

        self.eta_a = - self.phi_e_s_1

        self.eta_c = []

        for i in range(len(self.eps_s_c)):
            self.eta_c.append(self.phi_s_c_1 - self.phi_e_c_1 - self.U_c(self.c_s_c_sur[i]/self.c_s_c_max))

        i_0_a = self.F * self.k_0_a * self.c_e_s_1 ** self.alpha
        i_0_c = []

        for i in range(len(self.eps_s_c)):
            i_0_c.append(self.F * self.k_0_c * self.c_e_c_1 ** self.alpha * (self.c_s_c_max - self.c_s_c_sur[i]) ** self.alpha * (self.c_s_c_sur[i]) ** self.alpha)

        E_a = exp(self.alpha * (self.F / (self.R * self.T_s_1)) * self.eta_a) - exp(- self.alpha * (self.F / (self.R * self.T_s_1)) * self.eta_a)
        E_c = []

        for i in range(len(self.eps_s_c)):
            E_c.append(exp(self.alpha * (self.F / (self.R * self.T_c_1)) * self.eta_c[i]) - exp(- self.alpha * (self.F / (self.R * self.T_c_1)) * self.eta_c[i]))

        self.q_s = 0
        self.q_s += self.kappa_s * (1/self.L_s**2) * inner(grad(self.phi_e_s_1), grad(self.phi_e_s_1))
        self.q_s += self.kappa_D_s * (1/self.c_e_s_1) * (1/self.L_s**2) *  inner(grad(self.c_e_s_1), grad(self.phi_e_s_1))

        self.q_c = 0
        self.q_c += self.sigma_c * (1/self.L_c**2) * inner(grad(self.phi_s_c_1), grad(self.phi_s_c_1))
        self.q_c += self.kappa_c * (1/self.L_c**2) * inner(grad(self.phi_e_c_1), grad(self.phi_e_c_1))
        self.q_c += self.kappa_D_c * (1/self.c_e_c_1) * (1/self.L_c**2) *  inner(grad(self.c_e_c_1), grad(self.phi_e_c_1))
        for i in range(len(self.eps_s_c)):
            self.q_c += self.j_Li_c_1[i] * self.eta_c[i]

        self.j_Li_a = i_0_a * E_a

        # c_e_0

        F_c_e_s_0 = (self.c_e_s_1 - self.c_e_s_0) * self.c_e_s * self.dx + self.lm_c_e_sc_1 * self.c_e_s * self.ds(2)
        F_c_e_c_0 = (self.c_e_c_1 - self.c_e_c_0) * self.c_e_c * self.dx - self.lm_c_e_sc_1 * self.c_e_c * self.ds(1)

        self.F_lm_c_e_sc = self.lm_c_e_sc * self.c_e_s_1 * self.ds(2) - self.lm_c_e_sc * self.c_e_c_1 * self.ds(1)

        F_c_e_0 = F_c_e_s_0 \
                + F_c_e_c_0 + self.F_lm_c_e_sc

        # c_s_0

        F_c_s_c_0 = split(self.c_s_c_1)[0] * split(self.c_s_c)[0] * self.dx - split(self.c_s_c_0)[0] * split(self.c_s_c)[0] * self.dx

        for j in range(1, self.SGM_order_c):
            F_c_s_c_0 += split(self.c_s_c_1)[j] * split(self.c_s_c)[j] * self.dx

        F_c_s_0 = F_c_s_c_0

        # phi_e

        F_phi_e_s = (1/self.L_s) * inner(self.kappa_s * grad(self.phi_e_s_1), grad(self.phi_e_s)) * self.dx \
                  + self.lm_phi_e_sc_1 * self.phi_e_s * self.ds(2) \
                  + (1/self.L_s) * inner(self.kappa_D_s * (1/self.c_e_s_1) * grad(self.c_e_s_1), grad(self.phi_e_s)) * self.dx

        F_phi_e_c = (1/self.L_c) * inner(self.kappa_c * grad(self.phi_e_c_1), grad(self.phi_e_c)) * self.dx \
                  - self.lm_phi_e_sc_1 * self.phi_e_c * self.ds(1) \
                  + (1/self.L_c) * inner(self.kappa_D_c * (1/self.c_e_c_1) * grad(self.c_e_c_1), grad(self.phi_e_c)) * self.dx

        for i in range(len(self.eps_s_c)):
            F_phi_e_c -= self.L_c * self.j_Li_c_1[i] * self.phi_e_c * self.dx

        F_lm_phi_e_sc = self.phi_e_s_1 * self.lm_phi_e_sc * self.ds(2) - self.phi_e_c_1 * self.lm_phi_e_sc * self.ds(1)

        self.F_phi_e = F_phi_e_s \
                     + F_phi_e_c + F_lm_phi_e_sc

        # li-metal

        self.F_phi_e -= self.j_Li_a * self.phi_e_s * self.ds(1)

        # phi_s

        F_phi_s_c = (1. / self.L_c) * inner(self.sigma_c * grad(self.phi_s_c_1), grad(self.phi_s_c)) * self.dx \
                  - self.lm_app_1 * self.phi_s_c * self.ds(2)

        for i in range(len(self.eps_s_c)):
            F_phi_s_c += self.L_c * self.j_Li_c_1[i] * self.phi_s_c * self.dx

        self.F_phi_s = F_phi_s_c

        # j_Li

        F_j_Li_c = 0

        for i in range(len(self.eps_s_c)):
            F_j_Li_c += self.j_Li_c_1[i] * self.j_Li_c[i] * self.dx - self.a_s_c[i] * i_0_c[i] * E_c[i] * self.j_Li_c[i] * self.dx

        self.F_j_Li = F_j_Li_c

        # T_0

        F_T_s_0 = (self.T_s_1 - self.T_s_0) * self.T_s * self.dx + self.lm_T_sc_1 * self.T_s * self.ds(2)
        F_T_c_0 = (self.T_c_1 - self.T_c_0) * self.T_c * self.dx - self.lm_T_sc_1 * self.T_c * self.ds(1)

        self.F_lm_T_sc = self.lm_T_sc * self.T_s_1 * self.ds(2) - self.lm_T_sc * self.T_c_1 * self.ds(1)

        F_T_0 = F_T_s_0 \
              + F_T_c_0 + self.F_lm_T_sc

        # lm_app

        self.F_lm_app = self.beta * self.phi_s_c_1 * self.lm_app * self.ds(2) \
                      - self.beta * self.v_app * self.lm_app * self.dx \
                      + (1 - self.beta) * (self.lm_app_1 - (self.Q * self.i_app / self.area)) * self.lm_app * self.dx


        self.F_var_0 = F_c_e_0 + F_c_s_0 \
                     + self.F_phi_e \
                     + self.F_phi_s \
                     + self.F_j_Li \
                     + F_T_0 + self.F_lm_app

        self.J_var_0 = derivative(self.F_var_0, self.u_1)

    def build_wf_ie(self):

        # c_e

        F_c_e_s = self.L_s * self.eps_e_s * ((self.c_e_s_1 - self.c_e_s_0) / self.Deltat) * self.c_e_s * self.dx + (1. / self.L_s) * inner(self.D_e_s * grad(self.c_e_s_1), grad(self.c_e_s)) * self.dx \
                + self.lm_c_e_sc_1 * self.c_e_s * self.ds(2)

        F_c_e_c = self.L_c * self.eps_e_c * ((self.c_e_c_1 - self.c_e_c_0) / self.Deltat) * self.c_e_c * self.dx + (1. / self.L_c) * inner(self.D_e_c * grad(self.c_e_c_1), grad(self.c_e_c)) * self.dx \
                - self.lm_c_e_sc_1 * self.c_e_c * self.ds(1)

        for i in range(len(self.eps_s_c)):
            F_c_e_c -= self.L_c * ((1 - self.t_p_c) / self.F) * self.j_Li_c_1[i] * self.c_e_c * self.dx

        F_c_e = F_c_e_s \
              + F_c_e_c + self.F_lm_c_e_sc

        # li-metal

        F_c_e += ((1 - self.t_p_s) / self.F) * self.lm_app_1 * self.c_e_s * self.ds(1)

        # c_s

        F_c_s_c = 0

        for k in range(len(self.eps_s_c)):
            for j in range(self.SGM_order_c):
                for i in range(self.SGM_order_c):

                    F_c_s_c += self.M_c_s_c[i, j] * ((split(self.c_s_c_1)[i] - split(self.c_s_c_0)[i]) / self.Deltat) * split(self.c_s_c)[j] * self.dx

                    F_c_s_c += (self.D_s_c / self.R_s_c[k] ** 2) * self.K_c_s_c[i, j] * split(self.c_s_c_1)[i] * split(self.c_s_c)[j] * self.dx

                F_c_s_c = F_c_s_c + (1. / self.R_s_c[k]) * (1. / (self.a_s_c[k] * self.F)) * self.P_c_s_c[j] * self.j_Li_c_1[k] * split(self.c_s_c)[j] * self.dx


        F_c_s = F_c_s_c

        # T

        if self.solve_thermal:

            F_T_s = self.L_s * self.rho_s * self.c_p_s * ((self.T_s_1 - self.T_s_0) / self.Deltat) * self.T_s * self.dx + (1. / self.L_s) * inner(self.k_t_s * grad(self.T_s_1), grad(self.T_s)) * self.dx - self.L_s * self.q_s * self.T_s * self.dx \
                  + self.lm_T_sc_1 * self.T_s * self.ds(2) \
                  + self.h_t * (self.T_s_1 - self.T_ext) * self.T_s * self.ds(1)

            F_T_c = self.L_c * self.rho_c * self.c_p_c * ((self.T_c_1 - self.T_c_0) / self.Deltat) * self.T_c * self.dx + (1. / self.L_c) * inner(self.k_t_c * grad(self.T_c_1), grad(self.T_c)) * self.dx - self.L_c * self.q_c * self.T_c * self.dx \
                  - self.lm_T_sc_1 * self.T_c * self.ds(1) \
                  + self.h_t * (self.T_c_1 - self.T_ext) * self.T_c * self.ds(2)
        else:
            F_T_s = (self.T_s_1 - self.T_ini) * self.T_s * self.dx + self.lm_T_sc_1 * self.T_s * self.ds(2)
            F_T_c = (self.T_c_1 - self.T_ini) * self.T_c * self.dx - self.lm_T_sc_1 * self.T_c * self.ds(1)

        self.F_lm_T_sc = self.lm_T_sc * self.T_s_1 * self.ds(2) - self.lm_T_sc * self.T_c_1 * self.ds(1)

        F_T = F_T_s \
            + F_T_c + self.F_lm_T_sc

        self.F_var_1 = F_c_e + F_c_s \
                     + self.F_phi_e \
                     + self.F_phi_s \
                     + self.F_j_Li \
                     + F_T + self.F_lm_app

        self.J_var_1 = derivative(self.F_var_1, self.u_1)

    def build_wf_rk(self):

        # c_e

        F_c_e_s = self.L_s * self.eps_e_s * self.dc_e_sdt * self.c_e_s * self.dx + (1. / self.L_s) * inner(self.D_e_s * grad(self.c_e_s_1), grad(self.c_e_s)) * self.dx \
                + self.lm_c_e_sc_1 * self.c_e_s * self.ds(2)

        F_c_e_c = self.L_c * self.eps_e_c * self.dc_e_cdt * self.c_e_c * self.dx + (1. / self.L_c) * inner(self.D_e_c * grad(self.c_e_c_1), grad(self.c_e_c)) * self.dx \
                - self.lm_c_e_sc_1 * self.c_e_c * self.ds(1)

        for i in range(len(self.eps_s_c)):
            F_c_e_c -= self.L_c * ((1 - self.t_p_c) / self.F) * self.j_Li_c_1[i] * self.c_e_c * self.dx

        F_c_e = F_c_e_s \
              + F_c_e_c + self.F_lm_c_e_sc

        # li-metal

        F_c_e += ((1 - self.t_p_s) / self.F) * self.lm_app_1 * self.c_e_s * self.ds(1)

        # c_s

        F_c_s_c = 0

        for k in range(len(self.eps_s_c)):
            for j in range(self.SGM_order_c):
                for i in range(self.SGM_order_c):

                    F_c_s_c += self.M_c_s_c[i, j] * split(self.dc_s_cdt)[i] * split(self.c_s_c)[j] * self.dx

                    F_c_s_c += (self.D_s_c / self.R_s_c[k] ** 2) * self.K_c_s_c[i, j] * split(self.c_s_c_1)[i] * split(self.c_s_c)[j] * self.dx

                F_c_s_c = F_c_s_c + (1. / self.R_s_c[k]) * (1. / (self.a_s_c[k] * self.F)) * self.P_c_s_c[j] * self.j_Li_c_1[k] * split(self.c_s_c)[j] * self.dx


        F_c_s = F_c_s_c

        # T

        if self.solve_thermal:

            F_T_s = self.L_s * self.rho_s * self.c_p_s * self.dT_sdt * self.T_s * self.dx + (1. / self.L_s) * inner(self.k_t_s * grad(self.T_s_1), grad(self.T_s)) * self.dx - self.L_s * self.q_s * self.T_s * self.dx \
                  + self.lm_T_sc_1 * self.T_s * self.ds(2) \
                  + self.h_t * (self.T_s_1 - self.T_ext) * self.T_s * self.ds(1)

            F_T_c = self.L_c * self.rho_c * self.c_p_c * self.dT_cdt * self.T_c * self.dx + (1. / self.L_c) * inner(self.k_t_c * grad(self.T_c_1), grad(self.T_c)) * self.dx - self.L_c * self.q_c * self.T_c * self.dx \
                  - self.lm_T_sc_1 * self.T_c * self.ds(1) \
                  + self.h_t * (self.T_c_1 - self.T_ext) * self.T_c * self.ds(2)
        else:
            F_T_s = (self.T_s_1 - self.T_ini) * self.T_s * self.dx + self.lm_T_sc_1 * self.T_s * self.ds(2)
            F_T_c = (self.T_c_1 - self.T_ini) * self.T_c * self.dx - self.lm_T_sc_1 * self.T_c * self.ds(1)

        self.F_lm_T_sc = self.lm_T_sc * self.T_s_1 * self.ds(2) - self.lm_T_sc * self.T_c_1 * self.ds(1)

        F_T = F_T_s \
            + F_T_c + self.F_lm_T_sc

        self.F_var = F_c_e + F_c_s \
                   + self.F_phi_e \
                   + self.F_phi_s \
                   + self.F_j_Li \
                   + F_T + self.F_lm_app

    def store(self, t, x, level=0):

        batFEM.class_battery_model.Model.store(self, t, x)

        self.u_0.vector()[:] = x

        if level > 0:
            assign(self.c_e_s_fnc, self.u_0.sub(0).sub(0))
            assign(self.c_e_c_fnc, self.u_0.sub(0).sub(1))

            assign(self.phi_e_s_fnc, self.u_0.sub(2).sub(0))
            assign(self.phi_e_c_fnc, self.u_0.sub(2).sub(1))

            assign(self.phi_s_c_fnc, self.u_0.sub(3))

            if self.solve_thermal:
                assign(self.T_s_fnc, self.u_0.sub(5).sub(0))
                assign(self.T_c_fnc, self.u_0.sub(5).sub(1))

                self.T_s_pvd << (self.T_s_fnc, t)
                self.T_c_pvd << (self.T_c_fnc, t)

            self.c_e_s_pvd << (self.c_e_s_fnc, t)
            self.c_e_c_pvd << (self.c_e_c_fnc, t)

            self.phi_e_s_pvd << (self.phi_e_s_fnc, t)
            self.phi_e_c_pvd << (self.phi_e_c_fnc, t)

            self.phi_s_c_pvd << (self.phi_s_c_fnc, t)

    def get_voltage(self, x):

        self.u_1.vector()[:] = x

        return assemble(self.phi_s_c_1 * self.ds(2))

    def get_current(self, x):

        self.u_1.vector()[:] = x

        return assemble(self.lm_app_1 * self.dx)

class RK_ME_PE_P2D(ME_PE_P2D, fatDAE.dolfin_interface.class_problem.UFL_Problem):

    def __init__(self, cell, t_0, t_f, simulation_options, save_path='results/'):

        ME_PE_P2D.__init__(self, cell, simulation_options)

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

if __name__ == '__main__':

    import os; import shutil; import json; import argparse; import batFEM.class_battery

    parser = argparse.ArgumentParser(description = """batFEM""");

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

    problem = ME_PE_P2D(cell, json_options)

    print('Weight [kg]:', cell.weight)
    print('Volume [L]:', cell.volume * 1000)

    status = problem.solve_ie(h=10., v_min=3., i_app=1.0, t_f=3600., store_level=json_options['output and storage']['store level'], save_path=save_path)

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

    plt.show()
