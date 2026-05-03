from dolfin import *; import numpy; import matplotlib.pyplot as plt; import os#; from dolfin_adjoint import *

import sys; sys.path.insert(0,'..');

import batFEM.class_battery_model; import batFEM.PE_PE.class_PE_PE; import fatDAE.dolfin_interface.class_problem

class PE_PE_P2D(batFEM.PE_PE.class_PE_PE.PE_PE):

    def initial_guess(self):

        assign(self.u_0.sub(0).sub(0), interpolate(self.c_e_ini, self.V))
        assign(self.u_0.sub(0).sub(1), interpolate(self.c_e_ini, self.V))
        assign(self.u_0.sub(0).sub(2), interpolate(self.c_e_ini, self.V))

        assign(self.u_0.sub(1).sub(0).sub(0), interpolate(self.c_s_a_ini, self.V))
        assign(self.u_0.sub(1).sub(1).sub(0), interpolate(self.c_s_c_ini, self.V))

        if len(self.eps_s_a) > 1:
            for i in range(len(self.eps_s_a)):
                assign(self.u_0.sub(2).sub(0).sub(i), interpolate(self.c_s_a_ini, self.V))
                assign(self.u_0.sub(3).sub(0).sub(i), interpolate(self.c_s_a_ini, self.V))
        else:
            assign(self.u_0.sub(2).sub(0), interpolate(self.c_s_a_ini, self.V))
            assign(self.u_0.sub(3).sub(0), interpolate(self.c_s_a_ini, self.V))

        if len(self.eps_s_c) > 1:
            for i in range(len(self.eps_s_c)):
                assign(self.u_0.sub(2).sub(1).sub(i), interpolate(self.c_s_c_ini, self.V))
                assign(self.u_0.sub(3).sub(1).sub(i), interpolate(self.c_s_c_ini, self.V))
        else:
            assign(self.u_0.sub(2).sub(1), interpolate(self.c_s_c_ini, self.V))
            assign(self.u_0.sub(3).sub(1), interpolate(self.c_s_c_ini, self.V))

        assign(self.u_0.sub(5).sub(0), project(self.U_a(self.c_s_a_ini/self.c_s_a_max), self.V))
        assign(self.u_0.sub(5).sub(1), project(self.U_c(self.c_s_c_ini/self.c_s_c_max), self.V))

        if self.thermal_model=="adiabatic" or self.thermal_model=="lumped":
            assign(self.u_0.sub(7), interpolate(self.T_ini, self.P))
        else:
            assign(self.u_0.sub(7).sub(0), interpolate(self.T_ini, self.V))
            assign(self.u_0.sub(7).sub(1), interpolate(self.T_ini, self.V))
            assign(self.u_0.sub(7).sub(2), interpolate(self.T_ini, self.V))

        if self.solve_sei_a:
            assign(self.u_0.sub(10), interpolate(self.R_film_a_ini[0], self.V))
            #assign(self.u_0.sub(10+j), interpolate(self.eps_e_a_ini, self.V))
            assign(self.u_0.sub(11), project((self.rho_sei_a[0] / self.M_sei_a[0]) * self.a_s_a[0] * self.R_film_a_ini[0], self.V))
            #if self.solve_lpl_a:
            #    assign(self.u_0.sub(12+j), project((self.rho_lpl[0] / self.M_lpl[0]) * self.a_s_a[0] * self.R_film_a_0[0], self.V))
        #else:
            #assign(self.u_0.sub(9+j), interpolate(self.R_film_a_ini[0], self.V))
            #assign(self.u_0.sub(10+j), interpolate(self.eps_e_a_ini, self.V))
            #if self.solve_lpl_a:
            #    assign(self.u_0.sub(11+j), project((self.rho_lpl[0] / self.M_lpl[0]) * self.a_s_a[0] * self.R_film_a_0[0], self.V))

    def build_fs(self):

        P1 = FiniteElement('CG', self.mesh.ufl_cell(), self.FEM_order)
        P0 = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        LM = FiniteElement('R', self.mesh.ufl_cell(), 0)

        E_c_e = [P1, P1, P1, LM, LM]

        E_c_s_a = []
        E_c_s_c = []
        E_c_s_sur_a = []
        E_c_s_sur_c = []
        E_c_s_avg_a = []
        E_c_s_avg_c = []

        E_j_Li_a = []
        E_j_Li_c = []
        for j in range(len(self.eps_s_a)):
            for i in range(self.SGM_order_a):
                E_c_s_a.append(P1)
            E_c_s_sur_a.append(P1)
            E_c_s_avg_a.append(P1)
            E_j_Li_a.append(P1)

        for j in range(len(self.eps_s_c)):
            for i in range(self.SGM_order_c):
                E_c_s_c.append(P1)
            E_c_s_sur_c.append(P1)
            E_c_s_avg_c.append(P1)
            E_j_Li_c.append(P1)

        E_phi_e = [P1, P1, P1, LM, LM]
        E_phi_s = [P1, P1]

        if self.thermal_model=="adiabatic" or self.thermal_model=="lumped":
            E_T = LM
        else:
            E_T = [P1, P1, P1, LM, LM]

        E_R_film_a = []; E_c_r_a = []; E_eps_e_a = P1

        E_j_lpl_a = []; E_c_lpl_a = []
        E_j_sei_a = []; E_c_sei_a = []

        if self.solve_lpl_a or self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                E_R_film_a.append(P1)
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                E_j_lpl_a.append(P1)
                E_c_lpl_a.append(P1)
        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                E_j_sei_a.append(P1)
                E_c_sei_a.append(P1)
        
        if self.solve_sei_a:
            if self.solve_lpl_a:
                ME = MixedElement([E_c_e, \
                                        [E_c_s_a, E_c_s_c], [E_c_s_sur_a, E_c_s_sur_c], [E_c_s_avg_a, E_c_s_avg_c], \
                                        E_phi_e, E_phi_s, \
                                        [E_j_Li_a, E_j_Li_c], \
                                        E_T, LM, LM, \
                                        E_R_film_a, \
                                        E_c_sei_a, \
                                        E_c_lpl_a, \
                                        E_j_sei_a, \
                                        E_j_lpl_a])
            else:
                ME = MixedElement([E_c_e, \
                                        [E_c_s_a, E_c_s_c], [E_c_s_sur_a, E_c_s_sur_c], [E_c_s_avg_a, E_c_s_avg_c], \
                                        E_phi_e, E_phi_s, \
                                        [E_j_Li_a, E_j_Li_c], \
                                        E_T, LM, LM, \
                                        E_R_film_a, \
                                        E_c_sei_a, \
                                        E_j_sei_a])
        else:
            if self.solve_lpl_a:
                ME = MixedElement([E_c_e, \
                                        [E_c_s_a, E_c_s_c], [E_c_s_sur_a, E_c_s_sur_c], [E_c_s_avg_a, E_c_s_avg_c], \
                                        E_phi_e, E_phi_s, \
                                        [E_j_Li_a, E_j_Li_c], \
                                        E_T, LM, LM, \
                                        E_R_film_a, \
                                        E_c_lpl_a, \
                                        E_j_lpl_a])
            else:
                ME = MixedElement([E_c_e, \
                                        [E_c_s_a, E_c_s_c], [E_c_s_sur_a, E_c_s_sur_c],[E_c_s_avg_a, E_c_s_avg_c], \
                                        E_phi_e, E_phi_s, \
                                        [E_j_Li_a, E_j_Li_c], \
                                        E_T, LM, LM])
        
        self.V = FunctionSpace(self.mesh, P1)
        self.P = FunctionSpace(self.mesh, LM)

        self.V_plot = FunctionSpace(self.mesh_plot, P1)
        self.W = FunctionSpace(self.mesh, ME)

        self.dudt = TrialFunction(self.W)

        self.u_1 = Function(self.W)
        self.u_0 = Function(self.W); self.u = TestFunction(self.W)

        if self.solve_sei_a:
            if self.solve_lpl_a:
                self.dc_edt, self.dc_sdt, self.dc_s_surdt, self.dc_s_avgdt,self.dphi_edt, self.dphi_sdt, self.dj_Lidt, self.dTdt, self.dlm_phidt, self.dlm_appdt, self.dR_film_adt, self.dc_sei_adt, self.dc_lpl_adt, self.dj_sei_adt, self.dj_lpl_adt = split(self.dudt)
                self.c_e_1, self.c_s_1, self.c_s_sur_1, self.c_s_avg_1, self.phi_e_1, self.phi_s_1, self.j_Li_1, self.T_1, self.lm_phi_1, self.lm_app_1, self.R_film_a_1, self.c_sei_a_1, self.c_lpl_a_1, self.j_sei_a_1, self.j_lpl_a_1 = split(self.u_1)
                self.c_e_0, self.c_s_0, self.c_s_sur_0, self.c_s_avg_0, self.phi_e_0, self.phi_s_0, self.j_Li_0, self.T_0, self.lm_phi_0, self.lm_app_0, self.R_film_a_0, self.c_sei_a_0, self.c_lpl_a_0, self.j_sei_a_0, self.j_lpl_a_0 = split(self.u_0)
                self.c_e, self.c_s, self.c_s_sur, self.c_s_avg, self.phi_e, self.phi_s, self.j_Li, self.T, self.lm_phi, self.lm_app, self.R_film_a, self.c_sei_a, self.c_lpl_a, self.j_sei_a, self.j_lpl_a = split(self.u)
            else:
                self.dc_edt, self.dc_sdt, self.dc_s_surdt, self.dc_s_avgdt,self.dphi_edt, self.dphi_sdt, self.dj_Lidt, self.dTdt, self.dlm_phidt, self.dlm_appdt, self.dR_film_adt, self.dc_sei_adt, self.dj_sei_adt = split(self.dudt)
                self.c_e_1, self.c_s_1, self.c_s_sur_1, self.c_s_avg_1, self.phi_e_1, self.phi_s_1, self.j_Li_1, self.T_1, self.lm_phi_1, self.lm_app_1, self.R_film_a_1, self.c_sei_a_1, self.j_sei_a_1 = split(self.u_1)
                self.c_e_0, self.c_s_0, self.c_s_sur_0, self.c_s_avg_0, self.phi_e_0, self.phi_s_0, self.j_Li_0, self.T_0, self.lm_phi_0, self.lm_app_0, self.R_film_a_0, self.c_sei_a_0, self.j_sei_a_0 = split(self.u_0)
                self.c_e, self.c_s, self.c_s_sur, self.c_s_avg, self.phi_e, self.phi_s, self.j_Li, self.T, self.lm_phi, self.lm_app, self.R_film_a, self.c_sei_a, self.j_sei_a = split(self.u)
        else:
            if self.solve_lpl_a:
                self.dc_edt, self.dc_sdt, self.dc_s_surdt, self.dc_s_avgdt, self.dphi_edt, self.dphi_sdt, self.dj_Lidt, self.dTdt, self.dlm_phidt, self.dlm_appdt, self.dR_film_adt, self.dc_lpl_adt, self.dj_lpl_adt = split(self.dudt)
                self.c_e_1, self.c_s_1, self.c_s_sur_1, self.c_s_avg_1, self.phi_e_1, self.phi_s_1, self.j_Li_1, self.T_1, self.lm_phi_1, self.lm_app_1, self.R_film_a_1, self.c_lpl_a_1, self.j_lpl_a_1 = split(self.u_1)
                self.c_e_0, self.c_s_0, self.c_s_sur_0, self.c_s_avg_0, self.phi_e_0, self.phi_s_0, self.j_Li_0, self.T_0, self.lm_phi_0, self.lm_app_0, self.R_film_a_0, self.c_lpl_a_0, self.j_lpl_a_0 = split(self.u_0)
                self.c_e, self.c_s, self.c_s_sur, self.c_s_avg, self.phi_e, self.phi_s, self.j_Li, self.T, self.lm_phi, self.lm_app, self.R_film_a, self.c_lpl_a, self.j_lpl_a = split(self.u)
            else:
                self.dc_edt, self.dc_sdt, self.dc_s_surdt, self.dc_s_avgdt, self.dphi_edt, self.dphi_sdt, self.dj_Lidt, self.dTdt, self.dlm_phidt, self.dlm_appdt = split(self.dudt)
                self.c_e_1, self.c_s_1, self.c_s_sur_1, self.c_s_avg_1, self.phi_e_1, self.phi_s_1, self.j_Li_1, self.T_1, self.lm_phi_1, self.lm_app_1 = split(self.u_1)
                self.c_e_0, self.c_s_0, self.c_s_sur_0, self.c_s_avg_0, self.phi_e_0, self.phi_s_0, self.j_Li_0, self.T_0, self.lm_phi_0, self.lm_app_0 = split(self.u_0)
                self.c_e, self.c_s, self.c_s_sur, self.c_s_avg, self.phi_e, self.phi_s, self.j_Li, self.T, self.lm_phi, self.lm_app = split(self.u)

                self.R_film_a_1 = self.R_film_a_ini
        self.R_film_c_1 = self.R_film_c_ini
       
        self.dc_e_adt, self.dc_e_sdt, self.dc_e_cdt, self.dlm_c_e_asdt, self.dlm_c_e_scdt = split(self.dc_edt)

        self.c_e_a_1, self.c_e_s_1, self.c_e_c_1, self.lm_c_e_as_1, self.lm_c_e_sc_1 = split(self.c_e_1)
        self.c_e_a_0, self.c_e_s_0, self.c_e_c_0, self.lm_c_e_as_0, self.lm_c_e_sc_0 = split(self.c_e_0)

        self.c_e_a, self.c_e_s, self.c_e_c, self.lm_c_e_as, self.lm_c_e_sc = split(self.c_e)

        self.dc_s_adt, self.dc_s_cdt = split(self.dc_sdt)
        self.c_s_a_1, self.c_s_c_1 = split(self.c_s_1)
        self.c_s_a_0, self.c_s_c_0 = split(self.c_s_0)
        self.c_s_a, self.c_s_c = split(self.c_s)
        
        self.dc_s_sur_adt, self.dc_s_sur_cdt = split(self.dc_s_surdt)
        self.dc_s_avg_adt, self.dc_s_avg_cdt = split(self.dc_s_avgdt)

        self.c_s_sur_a_1, self.c_s_sur_c_1 = split(self.c_s_sur_1)
        self.c_s_sur_a_0, self.c_s_sur_c_0 = split(self.c_s_sur_0)
        self.c_s_avg_a_1, self.c_s_avg_c_1 = split(self.c_s_avg_1)
        self.c_s_avg_a_0, self.c_s_avg_c_0 = split(self.c_s_avg_0)

        self.c_s_sur_a, self.c_s_sur_c = split(self.c_s_sur)
        self.c_s_avg_a, self.c_s_avg_c = split(self.c_s_avg)

        self.phi_e_a_1, self.phi_e_s_1, self.phi_e_c_1, self.lm_phi_e_as_1, self.lm_phi_e_sc_1 = split(self.phi_e_1)
        self.phi_e_a_0, self.phi_e_s_0, self.phi_e_c_0, self.lm_phi_e_as_0, self.lm_phi_e_sc_0 = split(self.phi_e_0)

        self.phi_e_a, self.phi_e_s, self.phi_e_c, self.lm_phi_e_as, self.lm_phi_e_sc = split(self.phi_e)

        self.phi_s_a_1, self.phi_s_c_1 = split(self.phi_s_1)
        self.phi_s_a_0, self.phi_s_c_0 = split(self.phi_s_0)

        self.phi_s_a, self.phi_s_c = split(self.phi_s)

        self.j_Li_a_1, self.j_Li_c_1 = split(self.j_Li_1)
        self.j_Li_a_0, self.j_Li_c_0 = split(self.j_Li_0)

        self.j_Li_a, self.j_Li_c = split(self.j_Li)

        if self.thermal_model=="adiabatic" or self.thermal_model=="lumped":
            self.T_a_1 = self.T_1
            self.T_s_1 = self.T_1
            self.T_c_1 = self.T_1
        else:
            self.dT_adt, self.dT_sdt, self.dT_cdt, self.dlm_T_asdt, self.dlm_T_scdt = split(self.dTdt)

            self.T_a_1, self.T_s_1, self.T_c_1, self.lm_T_as_1, self.lm_T_sc_1 = split(self.T_1)
            self.T_a_0, self.T_s_0, self.T_c_0, self.lm_T_as_0, self.lm_T_sc_0 = split(self.T_0)

            self.T_a, self.T_s, self.T_c, self.lm_T_as, self.lm_T_sc = split(self.T)

    def build_arr(self):

        self.k_0_a = self.get_arr_a(self.k_0_a(split(self.c_s_sur_a_1)[0]/self.c_s_a_max), self.k_0_a_Ea, self.k_0_a_Tref)
        self.k_0_c = self.get_arr_c(self.k_0_c(split(self.c_s_sur_c_1)[0]/self.c_s_c_max), self.k_0_c_Ea, self.k_0_c_Tref)

        self.D_s_a = self.get_arr_a(self.D_s_a(split(self.c_s_sur_a_1)[0]/self.c_s_a_max), self.D_s_a_Ea, self.D_s_a_Tref)
        self.D_s_c = self.get_arr_c(self.D_s_c(split(self.c_s_sur_c_1)[0]/self.c_s_c_max), self.D_s_c_Ea, self.D_s_c_Tref)

        self.kappa_D_a = -2 * (self.R * self.T_a_1 / self.F) * (1 -  self.t_p_a) * self.kappa_a
        self.kappa_D_s = -2 * (self.R * self.T_s_1 / self.F) * (1 -  self.t_p_s) * self.kappa_s
        self.kappa_D_c = -2 * (self.R * self.T_c_1 / self.F) * (1 -  self.t_p_c) * self.kappa_c

    def build_pvd(self, state_machine=None):

        if self.store_level > 0:
            self.c_s_sur_a_pvd = File(self.save_path+'c_s_sur/c_s_sur_a.pvd'); self.c_s_sur_a_fnc = Function(self.V)
            self.c_s_sur_c_pvd = File(self.save_path+'c_s_sur/c_s_sur_c.pvd'); self.c_s_sur_c_fnc = Function(self.V)
            self.c_s_avg_a_pvd = File(self.save_path+'c_s_avg/c_s_avg_a.pvd'); self.c_s_avg_a_fnc = Function(self.V)
            self.c_s_avg_c_pvd = File(self.save_path+'c_s_avg/c_s_avg_c.pvd'); self.c_s_avg_c_fnc = Function(self.V)

            #if self.solve_sei_a or self.solve_lpl_a:
            #    self.R_film_a_pvd = File(self.save_path+'R_film/R_film_a.pvd'); self.R_film_a_fnc = Function(self.V)
            #    self.eps_e_a_pvd = File(self.save_path+'eps_e/eps_e_a.pvd'); self.eps_e_a_fnc = Function(self.V)

        if self.store_level > 1:

            self.c_e_a_pvd = File(self.save_path+'c_e/c_e_a.pvd'); self.c_e_a_fnc = Function(self.V)
            self.c_e_s_pvd = File(self.save_path+'c_e/c_e_s.pvd'); self.c_e_s_fnc = Function(self.V)
            self.c_e_c_pvd = File(self.save_path+'c_e/c_e_c.pvd'); self.c_e_c_fnc = Function(self.V)

            self.phi_e_a_pvd = File(self.save_path+'phi_e/phi_e_a.pvd'); self.phi_e_a_fnc = Function(self.V)
            self.phi_e_s_pvd = File(self.save_path+'phi_e/phi_e_s.pvd'); self.phi_e_s_fnc = Function(self.V)
            self.phi_e_c_pvd = File(self.save_path+'phi_e/phi_e_c.pvd'); self.phi_e_c_fnc = Function(self.V)

            self.phi_s_a_pvd = File(self.save_path+'phi_s/phi_s_a.pvd'); self.phi_s_a_fnc = Function(self.V)
            self.phi_s_c_pvd = File(self.save_path+'phi_s/phi_s_c.pvd'); self.phi_s_c_fnc = Function(self.V)

            self.j_tot_a_pvd = File(self.save_path+'j_tot/j_tot_a.pvd'); self.j_tot_a_fnc = Function(self.V)
            self.j_tot_c_pvd = File(self.save_path+'j_tot/j_tot_c.pvd'); self.j_tot_c_fnc = Function(self.V)

            if self.thermal_model=="adiabatic" or self.thermal_model=="lumped":
                pass
            else:
                self.T_a_pvd = File(self.save_path+'T/T_a.pvd'); self.T_a_fnc = Function(self.V)
                self.T_s_pvd = File(self.save_path+'T/T_s.pvd'); self.T_s_fnc = Function(self.V)
                self.T_c_pvd = File(self.save_path+'T/T_c.pvd'); self.T_c_fnc = Function(self.V)

        if self.store_level > 2:
            self.eta_a_pvd = File(self.save_path+'eta/eta_a.pvd'); self.eta_a_fnc = Function(self.V)
            self.eta_c_pvd = File(self.save_path+'eta/eta_c.pvd'); self.eta_c_fnc = Function(self.V)

            if self.solve_sei_a:
                self.j_sei_a_pvd = File(self.save_path+'j_sei/j_sei_a.pvd'); self.j_sei_a_fnc = Function(self.V)
                self.eta_sei_a_pvd = File(self.save_path+'eta_sei/eta_sei_a.pvd'); self.eta_sei_a_fnc = Function(self.V)

            if self.solve_lpl_a:
                self.j_lpl_a_pvd = File(self.save_path+'j_lpl/j_lpl_a.pvd'); self.j_lpl_a_fnc = Function(self.V)
                self.eta_lpl_a_pvd = File(self.save_path+'eta_lpl/eta_lpl_a.pvd'); self.eta_lpl_a_fnc = Function(self.V)
    
    def get_wf_c_e(self, td=True):

        if td:
            dc_e_adt = (self.c_e_a_1 - self.c_e_a_0) / self.Deltat
            dc_e_sdt = (self.c_e_s_1 - self.c_e_s_0) / self.Deltat
            dc_e_cdt = (self.c_e_c_1 - self.c_e_c_0) / self.Deltat
        else:
            dc_e_adt = self.dc_e_adt    
            dc_e_sdt = self.dc_e_sdt   
            dc_e_cdt = self.dc_e_cdt   

        F_c_e_a = self.L_a * dc_e_adt * self.c_e_a * self.dx + (1. / self.L_a) * inner((self.D_e_a) * grad(self.c_e_a_1), grad(self.c_e_a / self.eps_e_a_1)) * self.dx \
                + (self.lm_c_e_as_1/self.eps_e_a_1) * self.c_e_a * self.ds(2)

        for i in range(len(self.eps_s_a)):
            F_c_e_a -= self.L_a * ((1 - self.t_p_a) / self.eps_e_a_1) * self.j_tot_a[i] * self.c_e_a * self.dx

        F_c_e_s = self.L_s * dc_e_sdt * self.c_e_s * self.dx + (1. / self.L_s) * inner((self.D_e_s) * grad(self.c_e_s_1), grad(self.c_e_s / self.eps_e_s_1)) * self.dx \
                - (self.lm_c_e_as_1/self.eps_e_s_1) * self.c_e_s * self.ds(1) \
                + (self.lm_c_e_sc_1/self.eps_e_s_1) * self.c_e_s * self.ds(2)

        F_c_e_c = self.L_c * dc_e_cdt * self.c_e_c * self.dx + (1. / self.L_c) * inner((self.D_e_c) * grad(self.c_e_c_1), grad(self.c_e_c / self.eps_e_c_1)) * self.dx \
                - (self.lm_c_e_sc_1/self.eps_e_c_1) * self.c_e_c * self.ds(1)

        for i in range(len(self.eps_s_c)):
            F_c_e_c -= self.L_c * ((1 - self.t_p_c) / self.eps_e_c_1) * self.j_tot_c[i] * self.c_e_c * self.dx

        F_c_e = F_c_e_a + self.F_lm_c_e_as \
              + F_c_e_s \
              + F_c_e_c + self.F_lm_c_e_sc
        
        return F_c_e
    
    def get_wf_c_s(self, td=True):
        if self.microscale_method == 'SGM':
            F_c_s_a = 0
            for j in range(self.SGM_order_a):
                for i in range(self.SGM_order_a):
                    if td:
                        dc_s_adt = (split(self.c_s_a_1)[i] - split(self.c_s_a_0)[i]) / self.Deltat
                    else:
                        dc_s_adt = split(self.dc_s_adt)[i]
                    F_c_s_a = F_c_s_a + self.M_c_s_a[i,j]*dc_s_adt * split(self.c_s_a)[j] * self.dx \
                                      + (self.D_s_a/self.R_s_a[0]**2)*self.K_c_s_a[i,j] * split(self.c_s_a_1)[i] * split(self.c_s_a)[j] * self.dx
                F_c_s_a = F_c_s_a + (1./self.R_s_a[0]) * self.P_c_s_a[j] * self.j_Li_a_1[0] * split(self.c_s_a)[j] * self.dx
            F_c_s_c = 0
            for j in range(self.SGM_order_c):
                for i in range(self.SGM_order_c):
                    if td:
                        dc_s_cdt = (split(self.c_s_c_1)[i] - split(self.c_s_c_0)[i]) / self.Deltat
                    else:
                        dc_s_cdt = split(self.dc_s_cdt)[i]
                    F_c_s_c = F_c_s_c + self.M_c_s_c[i,j]*dc_s_cdt * split(self.c_s_c)[j] * self.dx \
                                      + (self.D_s_c/self.R_s_c[0]**2)*self.K_c_s_c[i,j] * split(self.c_s_c_1)[i] * split(self.c_s_c)[j] * self.dx
                F_c_s_c = F_c_s_c + (1./self.R_s_c[0]) * self.P_c_s_c[j] * self.j_Li_c_1[0] * split(self.c_s_c)[j] * self.dx
            
            F_c_s = F_c_s_a + self.F_c_s_sur_a + self.F_c_s_avg_a \
                  + F_c_s_c + self.F_c_s_sur_c + self.F_c_s_avg_c
        else:
            
            F_c_s_a = 0
            F_c_s_c = 0

            F_c_s_avg_a = 0
            F_c_s_avg_c = 0
            for i in range(len(self.eps_s_a)):
                if td:
                    dc_s_avg_adt = (split(self.c_s_avg_a_1)[i]-split(self.c_s_avg_a_0)[i]) / self.Deltat
                else:
                    dc_s_avg_adt = split(self.dc_s_avg_adt)[i]

                F_c_s_avg_a = F_c_s_avg_a + dc_s_avg_adt * split(self.c_s_avg_a)[i] * self.dx \
                            + 3 * (self.j_Li_a_1[i] / self.R_s_a[i]) * split(self.c_s_avg_a)[i] * self.dx
            for i in range(len(self.eps_s_c)):
                if td:
                    dc_s_avg_cdt = (split(self.c_s_avg_c_1)[i]-split(self.c_s_avg_c_0)[i]) / self.Deltat
                else:
                    dc_s_avg_cdt = split(self.dc_s_avg_cdt)[i]

                F_c_s_avg_c = F_c_s_avg_c + dc_s_avg_cdt * split(self.c_s_avg_c)[i] * self.dx \
                            + 3 * (self.j_Li_c_1[i] / self.R_s_c[i]) * split(self.c_s_avg_c)[i] * self.dx

            F_c_s = F_c_s_a + self.F_c_s_sur_a + F_c_s_avg_a \
                  + F_c_s_c + self.F_c_s_sur_c + F_c_s_avg_c
        return F_c_s
    
    def get_wf_T(self, td=True):

        if self.thermal_model == "adiabatic":
            if td:
                dTdt = (self.T_1 - self.T_0) / self.Deltat
            else:
                dTdt = self.dTdt
            F_T = dTdt * self.T * self.dx
            
        elif self.thermal_model == "lumped":
            L = self.L_a + self.L_s + self.L_c

            rho = (self.L_a * self.rho_a + self.L_s * self.rho_s + self.L_c * self.rho_c) / L
            c_p = (self.L_a * self.c_p_a + self.L_s * self.c_p_s + self.L_c * self.c_p_c) / L

            if td:
                dTdt = (self.T_1 - self.T_0) / self.Deltat
            else:
                dTdt = self.dTdt

            F_T = L * rho * c_p * dTdt * self.T * self.dx - self.L_a * self.q_a * self.T * self.dx \
                                                          - self.L_s * self.q_s * self.T * self.dx \
                                                          - self.L_c * self.q_c * self.T * self.dx \
                                                          + 2. * self.h_t * self.area_t * (self.T_1 - self.T_ext) * self.T * self.dx
        else: 
            if td:
                dT_adt = (self.T_a_1 - self.T_a_0) / self.Deltat
                dT_sdt = (self.T_s_1 - self.T_s_0) / self.Deltat
                dT_cdt = (self.T_c_1 - self.T_c_0) / self.Deltat
            else:
                dT_adt = self.dT_adt
                dT_sdt = self.dT_sdt
                dT_cdt = self.dT_cdt

            F_T_a = self.L_a * self.rho_a * self.c_p_a * dT_adt * self.T_a * self.dx + (1. / self.L_a) * inner(self.k_t_a * grad(self.T_a_1), grad(self.T_a)) * self.dx - self.L_a * self.q_a * self.T_a * self.dx \
                  + self.lm_T_as_1 * self.T_a * self.ds(2) \
                  + self.h_t * (self.T_a_1 - self.T_ext) * self.T_a * self.ds(1)

            F_T_s = self.L_s * self.rho_s * self.c_p_s * dT_sdt * self.T_s * self.dx + (1. / self.L_s) * inner(self.k_t_s * grad(self.T_s_1), grad(self.T_s)) * self.dx - self.L_s * self.q_s * self.T_s * self.dx \
                  - self.lm_T_as_1 * self.T_s * self.ds(1) \
                  + self.lm_T_sc_1 * self.T_s * self.ds(2)

            F_T_c = self.L_c * self.rho_c * self.c_p_c * dT_cdt * self.T_c * self.dx + (1. / self.L_c) * inner(self.k_t_c * grad(self.T_c_1), grad(self.T_c)) * self.dx - self.L_c * self.q_c * self.T_c * self.dx \
                  - self.lm_T_sc_1 * self.T_c * self.ds(1) \
                  + self.h_t * (self.T_c_1 - self.T_ext) * self.T_c * self.ds(2)

            self.F_lm_T_as = self.lm_T_as * self.T_a_1 * self.ds(2) - self.lm_T_as * self.T_s_1 * self.ds(1)
            self.F_lm_T_sc = self.lm_T_sc * self.T_s_1 * self.ds(2) - self.lm_T_sc * self.T_c_1 * self.ds(1)

            F_T = F_T_a + self.F_lm_T_as \
                + F_T_s \
                + F_T_c + self.F_lm_T_sc
            
        return F_T
    
    def get_wf_R_film(self,td=True):

        F_R_film_a = 0
        if self.solve_sei_a or self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                if td:
                    dR_film_adt=(self.R_film_a_1[i]-self.R_film_a_0[i]) / self.Deltat
                else:
                    dR_film_adt=self.dR_film_adt[i]

                F_R_film_a += dR_film_adt * self.R_film_a[i] * self.dx

        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                F_R_film_a += ((self.M_sei_a[i]/self.rho_sei_a[i]) * self.j_sei_a_1[i]) * self.R_film_a[i] * self.dx
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                F_R_film_a += ((self.M_lpl_a[i]/self.rho_lpl_a[i]) * self.j_lpl_a_1[i]) * self.R_film_a[i] * self.dx

        F_R_film = F_R_film_a
        
        return F_R_film
    
    def get_wf_c_sei(self,td=True):

        F_c_sei_a = 0
        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                if td:
                    dc_sei_adt=(self.c_sei_a_1[i]-self.c_sei_a_0[i]) / self.Deltat
                else:
                    dc_sei_adt=self.dc_sei_adt[i]
                #F_c_sei_a += (dc_sei_adt + self.j_sei_a_1[i] / (2.*self.F))* self.c_sei_a[i] * self.dx
                F_c_sei_a += (dc_sei_adt + self.a_s_a[i] * self.j_sei_a_1[i]) * self.c_sei_a[i] * self.dx
        
        F_c_sei = F_c_sei_a 
        
        return F_c_sei

    def get_wf_c_lpl(self,td=True):

        F_c_lpl_a = 0
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                if td:
                    dc_lpl_adt=(self.c_lpl_a_1[i]-self.c_lpl_a_0[i]) / self.Deltat
                else:
                    dc_lpl_adt=self.dc_lpl_adt[i]
                #F_c_lpl_a += (dc_lpl_adt + self.j_lpl_a_1[i] / (1.*self.F)) * self.c_lpl_a[i] * self.dx
                F_c_lpl_a += (dc_lpl_adt + self.a_s_a[i] * self.j_lpl_a_1[i]) * self.c_lpl_a[i] * self.dx
        
        F_c_lpl = F_c_lpl_a
        
        return F_c_lpl

    def build_wf_0(self):


        self.eps_e_a_1 = 1
        for i in range(len(self.eps_s_a)):
            self.eps_e_a_1 -= self.eps_s_a[i]
            self.eps_e_a_1 -= self.a_s_a[i]*self.R_film_a_1[i]
        for i in range(len(self.eps_i_a)):
            self.eps_e_a_1 -= self.eps_i_a[i]

        #self.eps_e_a_1 = self.eps_e_a_ini
        self.eps_e_s_1 = self.eps_e_s_ini
        self.eps_e_c_1 = self.eps_e_c_ini

        #self.eps_e_c_1 = 1
        #for i in range(len(self.eps_s_c)):
        #    self.eps_e_c_1 -= self.eps_s_c[i]
        #    self.eps_e_c_1 -= self.a_s_c[i]*self.R_film_c_1[i]
        #for i in range(len(self.eps_i_c)):
        #    self.eps_e_c_1 -= self.eps_i_c[i]
        
        self.tortuosity_e_a = self.eps_e_a_1 ** (1 - self.bruggeman_e_a)
        self.tortuosity_s_a = (1 - self.eps_e_a_1) ** (1 - self.bruggeman_s_a)

        self.tortuosity_e_s = self.eps_e_s_1 ** (1 - self.bruggeman_e_s)

        self.tortuosity_e_c = self.eps_e_c_1 ** (1 - self.bruggeman_e_c)
        self.tortuosity_s_c = (1 - self.eps_e_c_1) ** (1 - self.bruggeman_s_c)

        self.build_brg()
        self.build_arr()

        self.j_tot_a = []
        self.j_tot_c = []

        self.Z_int_a = []; self.Z_sei_a = []; self.Z_lpl_a =[]
        self.Z_int_c = []; self.Z_sei_c = []

        for i in range(len(self.eps_s_a)):
            j_tot = self.a_s_a[i]*self.j_Li_a_1[i]
            if self.solve_sei_a:
                j_tot += self.a_s_a[i]*self.j_sei_a_1[i]
            if self.solve_lpl_a:
                j_tot += self.a_s_a[i]*self.j_lpl_a_1[i]
            self.j_tot_a.append(j_tot)

            if self.solve_sei_a:
                if self.solve_lpl_a:
                    Mf_sei = self.M_sei_a[i]/self.rho_sei_a[i] # m^3/mol
                    Mf_lpl = self.M_lpl_a[i]/self.rho_lpl_a[i] # m^3/mol
                    eps_lpl = (Mf_lpl*self.c_lpl_a_1[i]/(Mf_sei*self.c_sei_a_1[i]+Mf_lpl*self.c_lpl_a_1[i]))
                    Z_int_a = (self.R_film_a_1[i]/self.kappa_sei_a[i])*(1-eps_lpl)
                    Z_sei_a = (self.R_film_a_1[i]/self.sigma_sei_a[i])*(1-eps_lpl) \
                            + (self.R_film_a_1[i]/self.sigma_lpl_a[i])*(0+eps_lpl)
                    Z_lpl_a = (self.R_film_a_1[i]/self.kappa_sei_a[i])*(1-eps_lpl) \
                            + (self.R_film_a_1[i]/self.sigma_lpl_a[i])*(0+eps_lpl)
                else:
                    Z_int_a = self.R_film_a_1[i]/self.kappa_sei_a[i]
                    Z_sei_a = self.R_film_a_1[i]/self.sigma_sei_a[i]
                    Z_lpl_a = self.R_film_a_1[i]/self.kappa_sei_a[i]
            else:
                Z_int_a = self.R_film_a_1[i]/self.kappa_sei_a[i]
                Z_sei_a = self.R_film_a_1[i]/self.sigma_sei_a[i]
                Z_lpl_a = self.R_film_a_1[i]/self.kappa_sei_a[i]
            
            self.Z_int_a.append(Z_int_a)
            self.Z_sei_a.append(Z_sei_a)
            self.Z_lpl_a.append(Z_lpl_a)

        for i in range(len(self.eps_s_c)):
            j_tot = self.a_s_c[i]*self.j_Li_c_1[i]
            self.j_tot_c.append(j_tot)

            self.Z_int_c.append(self.R_film_c_1[i]/self.kappa_sei_c[i])
            self.Z_sei_c.append(self.R_film_c_1[i]/self.sigma_sei_c[i])

        self.eta_a = []
        self.eta_c = []

        for i in range(len(self.eps_s_a)):
            self.eta_a.append(self.phi_s_a_1 - self.phi_e_a_1 - self.Z_int_a[i] * self.F * self.j_Li_a_1[i] - self.U_a(split(self.c_s_sur_a_1)[i]/self.c_s_a_max))
        for i in range(len(self.eps_s_c)):
            self.eta_c.append(self.phi_s_c_1 - self.phi_e_c_1 - self.Z_int_c[i] * self.F * self.j_Li_c_1[i] - self.U_c(split(self.c_s_sur_c_1)[i]/self.c_s_c_max))

        self.eta_sei_a = []
        self.eta_lpl_a = []
        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                self.eta_sei_a.append(self.phi_s_a_1 - self.phi_e_a_1 - self.Z_sei_a[i] * self.F * self.j_sei_a_1[i] - self.U_sei_a[i])
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                self.eta_lpl_a.append(self.phi_s_a_1 - self.phi_e_a_1 - self.Z_lpl_a[i] * self.F * self.j_lpl_a_1[i] - self.U_lpl_a[i])

        i_0_a = []
        i_0_c = []

        for i in range(len(self.eps_s_a)):
            i_0_a.append(self.F * self.k_0_a * self.c_e_a_1 ** self.alpha_a_c[i] * (self.c_s_a_max - split(self.c_s_sur_a_1)[i]) ** self.alpha_a_a[i] * (split(self.c_s_sur_a_1)[i]) ** self.alpha_a_c[i])
        for i in range(len(self.eps_s_c)):
            i_0_c.append(self.F * self.k_0_c * self.c_e_c_1 ** self.alpha_c_c[i] * (self.c_s_c_max - split(self.c_s_sur_c_1)[i]) ** self.alpha_c_a[i] * (split(self.c_s_sur_c_1)[i]) ** self.alpha_c_c[i])

        i_0_sei_a = []
        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                i_0_sei_a.append(self.F*self.k_0_sei_a[i])
        i_0_lpl_a = []
        i_0_lst_a = []
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                i_0_lpl_a.append(self.i_0_lpl_a[i])
                i_0_lst_a.append(self.i_0_lst_a[i])

        E_a = []
        E_c = []

        for i in range(len(self.eps_s_a)):
            E_a.append(exp(self.alpha_a_a[i] * (self.F / (self.R * self.T_a_1)) * self.eta_a[i]) - exp(-self.alpha_a_c[i] * (self.F / (self.R * self.T_a_1)) * self.eta_a[i]))
        for i in range(len(self.eps_s_c)):
            E_c.append(exp(self.alpha_c_a[i] * (self.F / (self.R * self.T_c_1)) * self.eta_c[i]) - exp(-self.alpha_c_c[i] * (self.F / (self.R * self.T_c_1)) * self.eta_c[i]))

        E_a_sei = []
        E_a_lpl_strip = []
        E_a_lpl_plate = []

        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                E_sei = exp(+self.alpha_sei_a[i] * (self.F / (self.R * self.T_a_1)) * self.eta_sei_a[i]) \
                      - exp(-self.alpha_sei_c[i] * (self.F / (self.R * self.T_a_1)) * self.eta_sei_a[i])
                E_a_sei.append(E_sei)
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                Mf_lpl = self.M_lpl_a[i]/self.rho_lpl_a[i]
                # Sigmoid argument = absolute LPL volume fraction (K&J 2020 formulation).
                # k_lpl scales the sensitivity; sigmoid → 0 only when c_lpl → 0.
                eps_lpl_sigmoid = self.k_lpl_a[i] * Mf_lpl * self.c_lpl_a_1[i]
                E_a_lpl_strip.append(exp(+self.alpha_lpl_a[i] * (self.F / (self.R * self.T_a_1)) * self.eta_lpl_a[i]) * (1-exp(-eps_lpl_sigmoid)))
                E_a_lpl_plate.append(exp(-self.alpha_lpl_c[i] * (self.F / (self.R * self.T_a_1)) * self.eta_lpl_a[i]))

        self.q_a = 0
        self.q_a += self.sigma_a * (1/self.L_a**2) * inner(grad(self.phi_s_a_1), grad(self.phi_s_a_1))
        self.q_a += self.kappa_a * (1/self.L_a**2) * inner(grad(self.phi_e_a_1), grad(self.phi_e_a_1))
        self.q_a += self.kappa_D_c * (1/self.c_e_a_1) * (1/self.L_a**2) *  inner(grad(self.c_e_a_1), grad(self.phi_e_a_1))
        for i in range(len(self.eps_s_a)):
            self.q_a += self.F * self.a_s_a[i] * self.j_Li_a_1[i] * self.eta_a[i]

        self.q_s = 0
        self.q_s += self.kappa_s * (1/self.L_s**2) * inner(grad(self.phi_e_s_1), grad(self.phi_e_s_1))
        self.q_s += self.kappa_D_s * (1/self.c_e_s_1) * (1/self.L_s**2) *  inner(grad(self.c_e_s_1), grad(self.phi_e_s_1))

        self.q_c = 0
        self.q_c += self.sigma_c * (1/self.L_c**2) * inner(grad(self.phi_s_c_1), grad(self.phi_s_c_1))
        self.q_c += self.kappa_c * (1/self.L_c**2) * inner(grad(self.phi_e_c_1), grad(self.phi_e_c_1))
        self.q_c += self.kappa_D_c * (1/self.c_e_c_1) * (1/self.L_c**2) *  inner(grad(self.c_e_c_1), grad(self.phi_e_c_1))
        for i in range(len(self.eps_s_c)):
            self.q_c += self.F * self.a_s_c[i] * self.j_Li_c_1[i] * self.eta_c[i]

        # c_e_0

        F_c_e_a_0 = (self.c_e_a_1 - self.c_e_a_0) * self.c_e_a * self.dx + self.lm_c_e_as_1 * self.c_e_a * self.ds(2)
        F_c_e_s_0 = (self.c_e_s_1 - self.c_e_s_0) * self.c_e_s * self.dx - self.lm_c_e_as_1 * self.c_e_s * self.ds(1) + self.lm_c_e_sc_1 * self.c_e_s * self.ds(2)
        F_c_e_c_0 = (self.c_e_c_1 - self.c_e_c_0) * self.c_e_c * self.dx - self.lm_c_e_sc_1 * self.c_e_c * self.ds(1)

        self.F_lm_c_e_as = self.lm_c_e_as * self.c_e_a_1 * self.ds(2) - self.lm_c_e_as * self.c_e_s_1 * self.ds(1)
        self.F_lm_c_e_sc = self.lm_c_e_sc * self.c_e_s_1 * self.ds(2) - self.lm_c_e_sc * self.c_e_c_1 * self.ds(1)

        F_c_e_0 = F_c_e_a_0 + self.F_lm_c_e_as \
                + F_c_e_s_0 \
                + F_c_e_c_0 + self.F_lm_c_e_sc

        # c_s_0

        F_c_s_a_0 = (split(self.c_s_a_1)[0] - split(self.c_s_a_0)[0]) * split(self.c_s_a)[0] * self.dx
        F_c_s_c_0 = (split(self.c_s_c_1)[0] - split(self.c_s_c_0)[0]) * split(self.c_s_c)[0] * self.dx

        for j in range(1, self.SGM_order_a):
            F_c_s_a_0 += (split(self.c_s_a_1)[j]-split(self.c_s_a_0)[j]) * split(self.c_s_a)[j] * self.dx

        for j in range(1, self.SGM_order_c):
            F_c_s_c_0 += (split(self.c_s_c_1)[j]-split(self.c_s_c_0)[j]) * split(self.c_s_c)[j] * self.dx
            
        self.F_c_s_sur_a = (split(self.c_s_sur_a_1)[0] * split(self.c_s_sur_a)[0]) * self.dx
        for j in range(self.SGM_order_a):
            self.F_c_s_sur_a -= split(self.c_s_a_1)[j] * self.P_c_s_a[j] * split(self.c_s_sur_a)[0] * self.dx
        self.F_c_s_sur_c = (split(self.c_s_sur_c_1)[0] * split(self.c_s_sur_c)[0]) * self.dx
        for j in range(self.SGM_order_c):
            self.F_c_s_sur_c -= split(self.c_s_c_1)[j] * self.P_c_s_c[j] * split(self.c_s_sur_c)[0] * self.dx

        self.F_c_s_avg_a = (split(self.c_s_avg_a_1)[0] * split(self.c_s_avg_a)[0]) * self.dx
        for j in range(self.SGM_order_a):
            self.F_c_s_avg_a -= split(self.c_s_a_1)[j] * self.Q_c_s_a[j] * split(self.c_s_avg_a)[0] * self.dx
            
        self.F_c_s_avg_c = (split(self.c_s_avg_c_1)[0] * split(self.c_s_avg_c)[0]) * self.dx
        for j in range(self.SGM_order_c):
            self.F_c_s_avg_c -= split(self.c_s_c_1)[j] * self.Q_c_s_c[j] * split(self.c_s_avg_c)[0] * self.dx
            
        F_c_s_0 = F_c_s_a_0 + self.F_c_s_sur_a + self.F_c_s_avg_a \
                + F_c_s_c_0 + self.F_c_s_sur_c + self.F_c_s_avg_c

        # phi_e

        F_phi_e_a = (1/self.L_a) * inner(self.kappa_a * grad(self.phi_e_a_1), grad(self.phi_e_a)) * self.dx \
                  + self.lm_phi_e_as_1 * self.phi_e_a * self.ds(2) \
                  + (1/self.L_a) * inner(self.kappa_D_a * (1/self.c_e_a_1) * grad(self.c_e_a_1), grad(self.phi_e_a)) * self.dx

        for i in range(len(self.eps_s_a)):
            F_phi_e_a -= self.L_a * self.F * self.j_tot_a[i] * self.phi_e_a * self.dx

        F_phi_e_s = (1/self.L_s) * inner(self.kappa_s * grad(self.phi_e_s_1), grad(self.phi_e_s)) * self.dx \
                  + self.lm_phi_e_sc_1 * self.phi_e_s * self.ds(2) - self.lm_phi_e_as_1 * self.phi_e_s * self.ds(1) \
                  + (1/self.L_s) * inner(self.kappa_D_s * (1/self.c_e_s_1) * grad(self.c_e_s_1), grad(self.phi_e_s)) * self.dx

        F_phi_e_c = (1/self.L_c) * inner(self.kappa_c * grad(self.phi_e_c_1), grad(self.phi_e_c)) * self.dx \
                  - self.lm_phi_e_sc_1 * self.phi_e_c * self.ds(1) \
                  + (1/self.L_c) * inner(self.kappa_D_c * (1/self.c_e_c_1) * grad(self.c_e_c_1), grad(self.phi_e_c)) * self.dx

        for i in range(len(self.eps_s_c)):
            F_phi_e_c -= self.L_c * self.F * self.j_tot_c[i] * self.phi_e_c * self.dx

        F_lm_phi_e_as = self.phi_e_a_1 * self.lm_phi_e_as * self.ds(2) - self.phi_e_s_1 * self.lm_phi_e_as * self.ds(1)
        F_lm_phi_e_sc = self.phi_e_s_1 * self.lm_phi_e_sc * self.ds(2) - self.phi_e_c_1 * self.lm_phi_e_sc * self.ds(1)

        self.F_phi_e = F_phi_e_a + F_lm_phi_e_as \
                     + F_phi_e_s \
                     + F_phi_e_c + F_lm_phi_e_sc

        # phi_s

        F_phi_s_a = (1. / self.L_a) * inner(self.sigma_a * grad(self.phi_s_a_1), grad(self.phi_s_a)) * self.dx \
                  + self.lm_app_1 * self.phi_s_a * self.ds(1)

        for i in range(len(self.eps_s_a)):
            F_phi_s_a += self.L_a * self.F * self.j_tot_a[i] * self.phi_s_a * self.dx

        F_phi_s_c = (1. / self.L_c) * inner(self.sigma_c * grad(self.phi_s_c_1), grad(self.phi_s_c)) * self.dx \
                  - self.lm_app_1 * self.phi_s_c * self.ds(2)

        for i in range(len(self.eps_s_c)):
            F_phi_s_c += self.L_c * self.F * self.j_tot_c[i] * self.phi_s_c * self.dx

        self.F_phi_s = F_phi_s_a \
                     + F_phi_s_c \

        # j_Li

        F_j_Li_a = 0
        F_j_Li_c = 0

        for i in range(len(self.eps_s_a)):
            F_j_Li_a += self.j_Li_a_1[i] * self.j_Li_a[i] * self.dx - (i_0_a[i]/self.F) * E_a[i] * self.j_Li_a[i] * self.dx
        for i in range(len(self.eps_s_c)):
            F_j_Li_c += self.j_Li_c_1[i] * self.j_Li_c[i] * self.dx - (i_0_c[i]/self.F) * E_c[i] * self.j_Li_c[i] * self.dx

        self.F_j_Li = F_j_Li_a \
                    + F_j_Li_c

        # j_sei

        self.F_j_sei_a = 0
        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                self.F_j_sei_a += self.j_sei_a_1[i] * self.j_sei_a[i] * self.dx - (i_0_sei_a[i]/self.F) * E_a_sei[i] * self.j_sei_a[i] * self.dx
        # j_lpl — asymmetric exchange currents: i0_lpl for plating, i0_lst for stripping
        self.F_j_lpl_a = 0
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                self.F_j_lpl_a += self.j_lpl_a_1[i] * self.j_lpl_a[i] * self.dx \
                                 - (i_0_lst_a[i]/self.F) * E_a_lpl_strip[i] * self.j_lpl_a[i] * self.dx \
                                 + (i_0_lpl_a[i]/self.F) * E_a_lpl_plate[i] * self.j_lpl_a[i] * self.dx

        # R_0
        F_R_film_a_0 = 0

        if self.solve_sei_a or self.solve_lpl_a:
            F_R_film_a_0 += (self.R_film_a_1[i] - self.R_film_a_0[i]) * self.R_film_a[i] * self.dx

        #F_R_film_0 = F_R_film_a_0

        # c_r
        #F_c_r_a = 0
        #F_c_r_c = 0

        #if self.solve_sei_a:
        #    for i in range(len(self.eps_s_a)):
        #        F_c_r_a = F_c_r_a #- self.D_ec * ((self.c_ec_s_a_1[i] - self.c_ec_e) / self.R_film_a_1[i]) * self.c_ec_s_a[i] * self.dx \
                                  #+ (self.j_sei_a_1[i] / self.F) * self.c_ec_s_a[i] * self.dx
        #if self.solve_sei_c:
        #    for i in range(len(self.eps_s_c)):
        #        F_c_r_c = F_c_r_c #- self.D_ec * ((self.c_ec_s_c_1[i] - self.c_ec_e) / self.R_film_c_1[i]) * self.c_ec_s_c[i] * self.dx \
        #                          #+ (self.j_sei_c_1[i] / self.F) * self.c_ec_s_c[i] * self.dx

        #self.F_c_r = F_c_r_a \
        #           + F_c_r_c

        # c_sei
        F_c_sei_a_0 = 0
        if self.solve_sei_a:
            for i in range(len(self.eps_s_a)):
                #F_c_sei_a += (self.c_sei_a_1[i] - (self.rho_sei[i] / self.M_sei[i]) * (self.a_s_a[i] * self.R_film_a_0[i] + self.c_lpl_a_1[i] * self.M_lpl[i]/ self.rho_lpl[i])) * self.c_sei_a[i] * self.dx
                F_c_sei_a_0 += (self.c_sei_a_1[i] - self.c_sei_a_0[i]) * self.c_sei_a[i] * self.dx
                pass
        # c_lpl
        F_c_lpl_a_0 = 0
        if self.solve_lpl_a:
            for i in range(len(self.eps_s_a)):
                F_c_lpl_a_0 += (self.c_lpl_a_1[i] - self.c_lpl_a_0[i]) * self.c_lpl_a[i] * self.dx
                pass

        # eps_e_0
        #self.F_eps_e = 0

        #if self.solve_sei_a or self.solve_lpl_a:
        #    for i in range(len(self.eps_s_a)):
        #        self.F_eps_e = (self.eps_e_a_1 - (1-self.eps_s_a[i]-self.a_s_a[i]*self.R_film_a_1[i])) * self.eps_e_a * self.dx


        # T_0

        if self.thermal_model=="adiabatic" or self.thermal_model=="lumped":
            F_T_0 = (self.T_1 - self.T_0) * self.T * self.dx
        else:
            F_T_a_0 = (self.T_a_1 - self.T_a_0) * self.T_a * self.dx + self.lm_T_as_1 * self.T_a * self.ds(2)
            F_T_s_0 = (self.T_s_1 - self.T_s_0) * self.T_s * self.dx - self.lm_T_as_1 * self.T_s * self.ds(1) + self.lm_T_sc_1 * self.T_s * self.ds(2)
            F_T_c_0 = (self.T_c_1 - self.T_c_0) * self.T_c * self.dx - self.lm_T_sc_1 * self.T_c * self.ds(1)

            self.F_lm_T_as = self.lm_T_as * self.T_a_1 * self.ds(2) - self.lm_T_as * self.T_s_1 * self.ds(1)
            self.F_lm_T_sc = self.lm_T_sc * self.T_s_1 * self.ds(2) - self.lm_T_sc * self.T_c_1 * self.ds(1)

            F_T_0 = F_T_a_0 + self.F_lm_T_as \
                  + F_T_s_0 \
                  + F_T_c_0 + self.F_lm_T_sc

        # lm_phi

        #self.F_lm_phi = self.phi_e_a_1 * self.lm_phi * self.dx \
        #              + self.phi_e_s_1 * self.lm_phi * self.dx \
        #              + self.phi_e_c_1 * self.lm_phi * self.dx

        #self.F_phi_e = self.F_phi_e + self.phi_e_a * self.lm_phi_1 * self.dx \
        #                            + self.phi_e_s * self.lm_phi_1 * self.dx \
        #                            + self.phi_e_c * self.lm_phi_1 * self.dx \
        self.F_lm_phi = self.phi_s_a_1 * self.lm_phi * self.ds(1) 
        
        self.F_phi_s = self.F_phi_s + self.phi_s_a * self.lm_phi_1 * self.ds(1) \

        # lm_app

        self.F_lm_app = self.beta * self.phi_s_c_1 * self.lm_app * self.ds(2) \
                      - self.beta * self.phi_s_a_1 * self.lm_app * self.ds(1) \
                      - self.beta * self.v_app * self.lm_app * self.dx \
                      + (1 - self.beta) * (self.lm_app_1 - (self.Q * self.i_app / self.area)) * self.lm_app * self.dx

        self.F_var_0 = F_c_e_0 + F_c_s_0 \
                     + self.F_phi_e \
                     + self.F_phi_s \
                     + self.F_j_Li \
                     + F_T_0 + self.F_lm_phi + self.F_lm_app \
                     + F_R_film_a_0 \
                     + F_c_sei_a_0 \
                     + F_c_lpl_a_0 \
                     + self.F_j_sei_a \
                     + self.F_j_lpl_a

        self.J_var_0 = derivative(self.F_var_0, self.u_1)

    def build_wf_ie(self):

        # c_e
        F_c_e = self.get_wf_c_e(td=True)
        
        # c_s
        F_c_s = self.get_wf_c_s(td=True)

        # T
        F_T = self.get_wf_T(td=True)

        # R_film
        F_R_film = self.get_wf_R_film(td=True)

        # c_sei
        F_c_sei = self.get_wf_c_sei(td=True)

        # c_lpl
        F_c_lpl = self.get_wf_c_lpl(td=True)

        self.F_var_1 = F_c_e + F_c_s \
                     + self.F_phi_e \
                     + self.F_phi_s \
                     + self.F_j_Li \
                     + F_T + self.F_lm_phi + self.F_lm_app \
                     + F_R_film \
                     + F_c_sei \
                     + F_c_lpl \
                     + self.F_j_sei_a \
                     + self.F_j_lpl_a

        self.J_var_1 = derivative(self.F_var_1, self.u_1)

    def build_wf_rk(self):

        # c_e
        F_c_e = self.get_wf_c_e(td=False)

        # c_s
        F_c_s = self.get_wf_c_s(td=False)
        
        # T
        F_T = self.get_wf_T(td=False)

        # R_film
        F_R_film = self.get_wf_R_film(td=False)
        
        # c_sei
        F_c_sei = self.get_wf_c_sei(td=False)

        # c_lpl
        F_c_lpl = self.get_wf_c_lpl(td=False)
        
        self.F_var = F_c_e + F_c_s \
                   + self.F_phi_e \
                   + self.F_phi_s \
                   + self.F_j_Li \
                   + F_T + self.F_lm_phi + self.F_lm_app \
                   + F_R_film \
                   + F_c_sei \
                   + F_c_lpl \
                   + self.F_j_sei_a \
                   + self.F_j_lpl_a

    def store(self, t, x, level=0, state_name=None, state_number=0):

        batFEM.class_battery_model.Model.store(self, t, x, state_name, state_number)

        self.u_0.vector()[:] = x

        if level > 0:
            assign(self.c_s_sur_a_fnc, self.u_0.sub(2).sub(0))
            assign(self.c_s_sur_c_fnc, self.u_0.sub(2).sub(1))
            assign(self.c_s_avg_a_fnc, self.u_0.sub(3).sub(0))
            assign(self.c_s_avg_c_fnc, self.u_0.sub(3).sub(1))
            #self.c_s_sur_a_fnc.vector()[:]=self.c_s_sur_a_fnc.vector()[:]/float(self.c_s_a_max)
            #self.c_s_sur_c_fnc.vector()[:]=self.c_s_sur_c_fnc.vector()[:]/float(self.c_s_c_max)
            #self.c_s_avg_a_fnc.vector()[:]=self.c_s_avg_a_fnc.vector()[:]/float(self.c_s_a_max)
            #self.c_s_avg_c_fnc.vector()[:]=self.c_s_avg_c_fnc.vector()[:]/float(self.c_s_c_max)
            
            self.c_s_sur_a_pvd << (self.c_s_sur_a_fnc, t)
            self.c_s_sur_c_pvd << (self.c_s_sur_c_fnc, t)
            self.c_s_avg_a_pvd << (self.c_s_avg_a_fnc, t)
            self.c_s_avg_c_pvd << (self.c_s_avg_c_fnc, t)

            #if self.solve_sei_a or self.solve_lpl_a:
            #    assign(self.R_film_a_fnc,self.u_0.sub(9+j))
            #    assign(self.eps_e_a_fnc,self.u_0.sub(10+j))
            #
            #    self.R_film_a_pvd << (self.R_film_a_fnc, t)
            #    self.eps_e_a_pvd << (self.eps_e_a_fnc, t)

            
            
        if level > 1:

            assign(self.c_e_a_fnc, self.u_0.sub(0).sub(0))
            assign(self.c_e_s_fnc, self.u_0.sub(0).sub(1))
            assign(self.c_e_c_fnc, self.u_0.sub(0).sub(2))

            assign(self.phi_e_a_fnc, self.u_0.sub(4).sub(0))
            assign(self.phi_e_s_fnc, self.u_0.sub(4).sub(1))
            assign(self.phi_e_c_fnc, self.u_0.sub(4).sub(2))
            assign(self.phi_s_a_fnc, self.u_0.sub(5).sub(0))
            assign(self.phi_s_c_fnc, self.u_0.sub(5).sub(1))

            assign(self.j_tot_a_fnc, self.u_0.sub(6).sub(0))
            assign(self.j_tot_c_fnc, self.u_0.sub(6).sub(1))

            if self.thermal_model=="adiabatic" or self.thermal_model=="lumped":
                pass
            else:
                assign(self.T_a_fnc, self.u_0.sub(7).sub(0))
                assign(self.T_s_fnc, self.u_0.sub(7).sub(1))
                assign(self.T_c_fnc, self.u_0.sub(7).sub(2))

                self.T_a_pvd << (self.T_a_fnc, t)
                self.T_s_pvd << (self.T_s_fnc, t)
                self.T_c_pvd << (self.T_c_fnc, t)

            self.c_e_a_pvd << (self.c_e_a_fnc, t)
            self.c_e_s_pvd << (self.c_e_s_fnc, t)
            self.c_e_c_pvd << (self.c_e_c_fnc, t)

            self.phi_e_a_pvd << (self.phi_e_a_fnc, t)
            self.phi_e_s_pvd << (self.phi_e_s_fnc, t)
            self.phi_e_c_pvd << (self.phi_e_c_fnc, t)

            self.phi_s_a_pvd << (self.phi_s_a_fnc, t)
            self.phi_s_c_pvd << (self.phi_s_c_fnc, t)

            self.j_tot_a_pvd << (self.j_tot_a_fnc, t)
            self.j_tot_c_pvd << (self.j_tot_c_fnc, t)

        if level > 2:
            assign(self.eta_a_fnc, project(self.eta_a[0], self.V))
            assign(self.eta_c_fnc, project(self.eta_c[0], self.V))

            self.eta_a_pvd << (self.eta_a_fnc, t)
            self.eta_c_pvd << (self.eta_c_fnc, t)

            if self.solve_sei_a:
                assign(self.j_sei_a_fnc, project(self.j_sei_a_1[0], self.V))
                self.j_sei_a_pvd << (self.j_sei_a_fnc, t)
                assign(self.eta_sei_a_fnc, project(self.eta_sei_a[0], self.V))
                self.eta_sei_a_pvd << (self.eta_sei_a_fnc, t)

            if self.solve_lpl_a:
                assign(self.j_lpl_a_fnc, project(self.j_lpl_a_1[0], self.V))
                self.j_lpl_a_pvd << (self.j_lpl_a_fnc, t)
                assign(self.eta_lpl_a_fnc, project(self.eta_lpl_a[0], self.V))
                self.eta_lpl_a_pvd << (self.eta_lpl_a_fnc, t)

    def get_voltage(self, x):
        self.u_1.vector()[:] = x

        return assemble(self.phi_s_c_1 * self.ds(2) - self.phi_s_a_1 * self.ds(1))

    def get_current(self, x):
        self.u_1.vector()[:] = x

        return assemble(self.lm_app_1 * self.area * self.dx)

    def get_temperature(self, x):
        self.u_1.vector()[:] = x

        return assemble(((self.L_a * self.T_a_1 + self.L_s * self.T_s_1 + self.L_c * self.T_c_1) / (self.L_a+self.L_s+self.L_c))* self.dx)
    
    def get_xs_avg_a(self,x):
        self.u_1.vector()[:] = x
        return assemble((split(self.c_s_avg_a_1)[0]/self.c_s_a_max)*self.dx)
    
    def get_xs_avg_c(self,x):
        self.u_1.vector()[:] = x
        return assemble((split(self.c_s_avg_c_1)[0]/self.c_s_c_max)*self.dx)
    
    def get_xs_sur_a(self,x):
        self.u_1.vector()[:] = x
        return assemble((split(self.c_s_sur_a_1)[0]/self.c_s_a_max)*self.dx)
    
    def get_xs_sur_c(self,x):
        self.u_1.vector()[:] = x
        return assemble((split(self.c_s_sur_c_1)[0]/self.c_s_c_max)*self.dx)

    def get_ce_avg_a(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.c_e_a_1*self.dx)

    def get_ce_avg_s(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.c_e_s_1*self.dx)
    
    def get_ce_avg_c(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.c_e_c_1*self.dx)

    def get_delta_film_a(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.R_film_a_1[0]*self.dx)

    def get_eps_e_a(self,x):
        self.u_1.vector()[:] = x
        return assemble(self.eps_e_a_1*self.dx)
    
    def get_c_sei_a(self,x):
        if self.solve_sei_a:
            self.u_1.vector()[:] = x
            return assemble(self.c_sei_a_1[0]*self.dx)
        else:
            return 0
    def get_c_lpl_a(self,x):
        if self.solve_lpl_a:
            self.u_1.vector()[:] = x
            return assemble(self.c_lpl_a_1[0]*self.dx)
        else:
            return 0
    
class RK_PE_PE_P2D(PE_PE_P2D, fatDAE.dolfin_interface.class_problem.UFL_Problem):

    def __init__(self, cell, t_0, t_f, simulation_options, save_path='results/'):

        PE_PE_P2D.__init__(self, cell, simulation_options, save_path)

        self.setup()

        self.t = Expression("value", degree=1, value = t_0); self.t_v = variable(self.t)

        self.time_dependent_expresions = []

        fatDAE.dolfin_interface.class_problem.UFL_Problem.__init__(self, self.F_var, self.u_0.vector()[:], t_0, t_f)

        def boundary(x, on_boundary):
            return on_boundary

        self.set_boundary()

        self.M = self.M(self.t_0, self.x_0)

    def solve_initial(self, x):

        self.u_0.vector()[:] = x

        prm = self.solver_0.parameters

        try:
            prm["newton_solver"]["absolute_tolerance"] = 1E-12
            prm["newton_solver"]["relative_tolerance"] = 1E-12
            prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 1.0

            self.solver_0.solve()
        except:
            prm["newton_solver"]["absolute_tolerance"] = 1E-6
            prm["newton_solver"]["relative_tolerance"] = 1E-6
            prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 0.5

            self.solver_0.solve()
            

        return numpy.array(self.u_1.vector()[:])

class RK_PE_PE_P2D_VolumetricEnergy(PE_PE_P2D, fatDAE.dolfin_interface.class_problem.UFL_Control):

    def __init__(self, cell, opt_params, t_0, t_f, simulation_options, save_path='results/'):

        PE_PE_P2D.__init__(self, cell, simulation_options, save_path)

        self.setup()

        self.t = Expression("value", degree=1, value = t_0); self.t_v = variable(self.t)

        self.time_dependent_expresions = []

        CONTROL_MAP = {
                        ("negativeElectrode","thickness"): lambda self: self.L_a,
                        ("negativeElectrode","composition",0,"volumeFraction"): lambda self: self.eps_s_a[0],
                        ("negativeElectrode","composition",1,"volumeFraction"): lambda self: self.eps_i_a[0],
                        ("negativeElectrode","composition",0,"particleRadius"): lambda self: self.R_s_a[0],
                        ("positiveElectrode","thickness"): lambda self: self.L_c,
                        ("positiveElectrode","composition",0,"volumeFraction"): lambda self: self.eps_s_c[0],
                        ("positiveElectrode","composition",1,"volumeFraction"): lambda self: self.eps_i_c[0],
                        ("positiveElectrode","composition",0,"particleRadius"): lambda self: self.R_s_c[0],
                        ("separator","thickness"): lambda self: self.L_s,
                    }

        control = []

        for p in opt_params:

            key = tuple(p["path"])

            if key in CONTROL_MAP:
                control.append(CONTROL_MAP[key](self))

        g_form = (self.lm_app_1 * self.area * self.phi_s_c_1 / (3600*self.volume*1e3))* self.ds(2) - (self.lm_app_1 * self.area * self.phi_s_a_1 / (3600*self.volume*1e3)) * self.ds(1)
        J_form = g_form

        fatDAE.dolfin_interface.class_problem.UFL_Control.__init__(self, control, J_form, g_form, self.F_var, self.u_0.vector()[:], t_0, t_f)

        def boundary(x, on_boundary):
            return on_boundary

        self.set_boundary()

        self.M = self.M(self.t_0, self.x_0)

    def solve_initial(self, x):

        self.u_0.vector()[:] = x

        prm = self.solver_0.parameters

        try:
            prm["newton_solver"]["absolute_tolerance"] = 1E-12
            prm["newton_solver"]["relative_tolerance"] = 1E-12
            prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 1.0

            self.solver_0.solve()
        except:
            prm["newton_solver"]["absolute_tolerance"] = 1E-6
            prm["newton_solver"]["relative_tolerance"] = 1E-6
            prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 0.5

            self.solver_0.solve()
            

        return numpy.array(self.u_1.vector()[:])
    
class RK_PE_PE_P2D_GravimetricEnergy(PE_PE_P2D, fatDAE.dolfin_interface.class_problem.UFL_Control):

    def __init__(self, cell, opt_params, t_0, t_f, simulation_options, save_path='results/'):

        PE_PE_P2D.__init__(self, cell, simulation_options, save_path)

        self.setup()

        self.t = Expression("value", degree=1, value = t_0); self.t_v = variable(self.t)

        self.time_dependent_expresions = []

        CONTROL_MAP = {
                        ("negativeElectrode","thickness"): lambda self: self.L_a,
                        ("negativeElectrode","composition",0,"volumeFraction"): lambda self: self.eps_s_a[0],
                        ("negativeElectrode","composition",1,"volumeFraction"): lambda self: self.eps_i_a[0],
                        ("negativeElectrode","composition",0,"particleRadius"): lambda self: self.R_s_a[0],
                        ("positiveElectrode","thickness"): lambda self: self.L_c,
                        ("positiveElectrode","composition",0,"volumeFraction"): lambda self: self.eps_s_c[0],
                        ("positiveElectrode","composition",1,"volumeFraction"): lambda self: self.eps_i_c[0],
                        ("positiveElectrode","composition",0,"particleRadius"): lambda self: self.R_s_c[0],
                        ("separator","thickness"): lambda self: self.L_s,
                    }

        control = []

        for p in opt_params:

            key = tuple(p["path"])

            if key in CONTROL_MAP:
                control.append(CONTROL_MAP[key](self))

        g_form = (self.lm_app_1 * self.area * self.phi_s_c_1 / (3600*self.weight))* self.ds(2) - (self.lm_app_1 * self.area * self.phi_s_a_1 / (3600*self.weight)) * self.ds(1)
        J_form = g_form

        fatDAE.dolfin_interface.class_problem.UFL_Control.__init__(self, control, J_form, g_form, self.F_var, self.u_0.vector()[:], t_0, t_f)

        def boundary(x, on_boundary):
            return on_boundary

        self.set_boundary()

        self.M = self.M(self.t_0, self.x_0)

    def solve_initial(self, x):

        self.u_0.vector()[:] = x

        prm = self.solver_0.parameters

        try:
            prm["newton_solver"]["absolute_tolerance"] = 1E-12
            prm["newton_solver"]["relative_tolerance"] = 1E-12
            prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 1.0

            self.solver_0.solve()
        except:
            prm["newton_solver"]["absolute_tolerance"] = 1E-6
            prm["newton_solver"]["relative_tolerance"] = 1E-6
            prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 0.5

            self.solver_0.solve()
            

        return numpy.array(self.u_1.vector()[:])
    
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

    json_battery = batFEM.class_battery.parse_json(json_battery); cell = batFEM.class_battery.Cell(json_battery, temperature=298.15, SOC=1)

    problem = PE_PE_P2D(cell, json_options)

    print('Weight [kg]:', cell.weight)
    print('Volume [L]:', cell.volume * 1000)

    #status = problem.solve_dcc(h=5., v_min=2.8, i_app=1.0, t_f=3600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=0)
    #status = problem.solve_ccc(h=5., v_max=4.1, i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    #status = problem.solve_ccc(h=5., v_max=4.1, i_app=1.0, t_f=3600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    #status = problem.solve_dcc(h=5., v_min=2.8, i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    #status = problem.solve_dcc(h=1., v_min=2.8, i_app=2.0, t_f=1800., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    #status = problem.solve_ccc(h=1., v_max=4.1, i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    #status = problem.solve_ccc(h=1., v_max=4.1, i_app=2.0, t_f=1800., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    #status = problem.solve_dcc(h=1., v_min=2.8, i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    status = problem.solve_dcc(h=0.5, v_min=json_battery['properties']['minVoltage']['value'], i_app=5.0, t_f=900., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=0)
    status = problem.solve_ccc(h=1., v_max=json_battery['properties']['maxVoltage']['value'], i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    status = problem.solve_ccc(h=0.5, v_max=json_battery['properties']['maxVoltage']['value'], i_app=5.0, t_f=900., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    status = problem.solve_dcc(h=1., v_min=json_battery['properties']['minVoltage']['value'], i_app=0.0, t_f=600., store_level=json_options['output and storage']['store level'], save_path=save_path,already_setup=1)
    

    ene = abs(problem.get_ene())
    pow = abs(problem.get_pow())

    ene_weight = ene / cell.weight
    ene_volume = ene / (cell.volume * 1000)
    pow_weight = pow / cell.weight
    pow_volume = pow / (cell.volume * 1000)

    print('Energy [Wh/kg]:', ene_weight, 'Power [W/kg]:', pow_weight)
    print('Energy [Wh/m3]:', ene_volume, 'Power [W/m3]:', pow_volume)

    problem.write(json_options['output and storage']['write level'])

    #numpy.savetxt(save_path+'time.txt', problem.t_list)
    #numpy.savetxt(save_path+'current.txt', problem.i_list)
    #numpy.savetxt(save_path+'voltage.txt', problem.v_list)
    #numpy.savetxt(save_path+'temperature.txt', problem.k_list)

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
