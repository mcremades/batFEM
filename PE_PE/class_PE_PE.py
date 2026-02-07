from ast import Constant
from dolfin import *; import numpy; import matplotlib.pyplot as plt; import json#; from dolfin_adjoint import *

import batFEM.class_battery_model

def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

class PE_PE(batFEM.class_battery_model.Model):

    def __init__(self, cell, simulation_options, save_path=None):

        self.store_level = simulation_options['output and storage']['store level']; self.save_path=save_path
        self.write_level = simulation_options['output and storage']['write level']

        self.N_x = simulation_options['space discretization']['elements']; self.FEM_order = simulation_options['space discretization']['FEM order']

        self.microscale_method = simulation_options['space discretization']['microscale method']

        self.SGM_order_a = simulation_options['space discretization']['SGM order a']
        self.SGM_order_c = simulation_options['space discretization']['SGM order c']

        self.lumped_thermal = simulation_options['multiphysics']['lumped_thermal']

        self.solve_sei_a = simulation_options['multiphysics']['solve_sei_a']
        self.solve_sei_c = simulation_options['multiphysics']['solve_sei_c']
        self.solve_lpl_a = simulation_options['multiphysics']['solve_lpl_a']
        self.solve_lpl_c = simulation_options['multiphysics']['solve_lpl_c']


        self.build_sgm(); self.build_mesh()

        self.t_list = []
        self.i_list = []
        self.v_list = []

        self.xs_avg_a_list=[]
        self.xs_avg_c_list=[]
        self.xs_sur_a_list=[]
        self.xs_sur_c_list=[]

        self.ce_avg_a_list=[]
        self.ce_avg_s_list=[]
        self.ce_avg_c_list=[]

        self.k_list = []
        self.q_list = []

        # TODO split in more functions
        self.R_s_a = []; self.R_film_a_ini = []; self.U_sei_a = []; self.U_lpl_a = []
        self.R_s_c = []; self.R_film_c_ini = []

        self.k_0_sei_a = []; self.M_sei = []; self.rho_sei = []
        self.i_0_lpl_a = []; self.M_lpl = []; self.rho_lpl = []

        self.eps_s_a = []; self.eps_sei_a = []; self.kappa_sei_a = []
        self.eps_s_c = []; self.eps_sei_c = []; self.kappa_sei_c = []

        for material in cell.negativeElectrode.composition:
            if material.active == 1:
                self.R_s_a.append(Constant(material.particleRadius)); self.R_film_a_ini.append(Constant(material.filmRadius))
                self.eps_s_a.append(Constant(material.volumeFraction))

                if self.solve_sei_a or self.solve_lpl_a:
                    self.M_sei.append(Constant(material.seiMass))
                    self.rho_sei.append(Constant(material.seiDensity))
                if self.solve_sei_a:
                    self.U_sei_a.append(Constant(material.seiOCP))
                    self.k_0_sei_a.append(Constant(material.seiKineticConstant))
                if self.solve_lpl_a:
                    self.U_lpl_a.append(Constant(material.lplOCP))
                    self.i_0_lpl_a.append(Constant(material.lplExchangeCurrent))
                    self.M_lpl.append(Constant(material.lplMass))
                    self.rho_lpl.append(Constant(material.lplDensity))

                self.eps_sei_a.append(Constant(material.seiVolumeFraction))
                self.kappa_sei_a.append(Constant(material.seiIonicConductivity))

        for material in cell.positiveElectrode.composition:
            if material.active == 1:
                self.R_s_c.append(Constant(material.particleRadius)); self.R_film_c_ini.append(Constant(material.filmRadius))
                self.eps_s_c.append(Constant(material.volumeFraction))

                self.eps_sei_c.append(Constant(material.seiVolumeFraction))
                self.kappa_sei_c.append(Constant(material.seiIonicConductivity))

        self.eps_e_a_ini = Constant(cell.negativeElectrode.porosity)
        self.eps_e_s_ini = Constant(cell.separator.porosity)
        self.eps_e_c_ini = Constant(cell.positiveElectrode.porosity)

        self.bruggeman_e_a = Constant(cell.negativeElectrode.electrolyteBruggeman)
        self.bruggeman_e_s = Constant(cell.separator.bruggeman)
        self.bruggeman_e_c = Constant(cell.positiveElectrode.electrolyteBruggeman)

        self.bruggeman_s_a = Constant(cell.negativeElectrode.electrodeBruggeman)
        self.bruggeman_s_c = Constant(cell.positiveElectrode.electrodeBruggeman)

        self.sigma_a = Constant(cell.negativeElectrode.electronicConductivity)
        self.compute_sigma_a_eff = cell.negativeElectrode.compute_effective_electronicConductivity
        self.sigma_c = Constant(cell.positiveElectrode.electronicConductivity)
        self.compute_sigma_c_eff = cell.negativeElectrode.compute_effective_electronicConductivity

        self.k_t_a = Constant(cell.negativeElectrode.thermalConductivity)
        self.k_t_s = Constant(cell.separator.thermalConductivity)
        self.k_t_c = Constant(cell.positiveElectrode.thermalConductivity)

        self.rho_a = Constant(cell.negativeElectrode.density)
        self.rho_s = Constant(cell.separator.density)
        self.rho_c = Constant(cell.positiveElectrode.density)

        self.c_p_a = Constant(cell.negativeElectrode.heatCapacity)
        self.c_p_s = Constant(cell.separator.heatCapacity)
        self.c_p_c = Constant(cell.positiveElectrode.heatCapacity)

        self.h_t = Constant(cell.heatConvectionCoefficient)
        self.area_t = Constant(cell.heatConvectionArea)

        self.L_a = Constant(cell.negativeElectrode.thickness)
        self.L_s = Constant(cell.separator.thickness)
        self.L_c = Constant(cell.positiveElectrode.thickness)

        self.c_e_ini = Constant(cell.electrolyte.initialConcentration)

        self.c_ec_e = Constant(cell.electrolyte.solventConcentration)
        self.D_ec = Constant(cell.electrolyte.solventDiffusionConstant)

        if cell.electrolyte.diffusionConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.electrolyte.diffusionConstant['value']); type = cell.electrolyte.diffusionConstant['interpolation_type']

            if 'interpolation_options' in cell.electrolyte.diffusionConstant:
                opts = cell.electrolyte.diffusionConstant['interpolation_options']
            else:
                opts = None

            self.D_e_ref = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.electrolyte.diffusionConstant['type'] == 'function':
            self.D_e_ref = lambda c_e, T: eval(cell.electrolyte.diffusionConstant['value'])
        elif cell.electrolyte.diffusionConstant['type'] == 'constant':
            self.D_e_ref = lambda c_e, T: Constant(cell.electrolyte.diffusionConstant['value'])
        else:
            raise NameError('Unknown variable type')

        # arrhenius
        self.D_e_Ea = Constant(cell.electrolyte.diffusionConstant_Ea)
        self.D_e_Tref = Constant(cell.electrolyte.diffusionConstant_Tref)
        self.D_e = lambda c_e, T: self.D_e_ref(c_e, T) * exp((self.D_e_Ea / self.R) * (1/self.D_e_Tref - 1/T))
        

        if cell.electrolyte.ionicConductivity['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.electrolyte.ionicConductivity['value']); type = cell.electrolyte.ionicConductivity['interpolation_type']

            if 'interpolation_options' in cell.electrolyte.ionicConductivity:
                opts = cell.electrolyte.ionicConductivity['interpolation_options']
            else:
                opts = None

            self.kappa_ref = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.electrolyte.ionicConductivity['type'] == 'function':
            self.kappa_ref = lambda c_e, T: eval(cell.electrolyte.ionicConductivity['value'])
        elif cell.electrolyte.ionicConductivity['type'] == 'constant':
            self.kappa_ref = lambda c_e, T: Constant(cell.electrolyte.ionicConductivity['value'])
        else:
            raise NameError('Unknown variable type')

        # arrhenius
        self.kappa_Ea = Constant(cell.electrolyte.ionicConductivity_Ea)
        self.kappa_Tref = Constant(cell.electrolyte.ionicConductivity_Tref)
        self.kappa = lambda c_e, T: self.kappa_ref(c_e, T) * exp((self.kappa_Ea / self.R) * (1/self.kappa_Tref - 1/T))
        

        if cell.electrolyte.transferenceNumber['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.electrolyte.transferenceNumber['value']); type = cell.electrolyte.transferenceNumber['interpolation_type']

            if 'interpolation_options' in cell.electrolyte.transferenceNumber:
                opts = cell.electrolyte.transferenceNumber['interpolation_options']
            else:
                opts = None

            self.t_p = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.electrolyte.transferenceNumber['type'] == 'function':
            raise NameError('User defined functions not handled yet')
        elif cell.electrolyte.transferenceNumber['type'] == 'constant':
            self.t_p = Constant(cell.electrolyte.transferenceNumber['value'])
        else:
            raise NameError('Unknown variable type')

        if cell.negativeElectrode.composition[0].kineticConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.negativeElectrode.composition[0].kineticConstant['value']); type = cell.negativeElectrode.composition[0].kineticConstant['interpolation_type']

            if 'interpolation_options' in cell.negativeElectrode.composition[0].kineticConstant:
                opts = cell.negativeElectrode.composition[0].kineticConstant['interpolation_options']
            else:
                opts = None

            self.k_0_a = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.negativeElectrode.composition[0].kineticConstant['type'] == 'function':
            raise NameError('User defined functions not handled yet')
        elif cell.negativeElectrode.composition[0].kineticConstant['type'] == 'constant':
            self.k_0_a = Constant(cell.negativeElectrode.composition[0].kineticConstant['value'])
        else:
            raise NameError('Unknown variable type')

        if cell.positiveElectrode.composition[0].kineticConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.positiveElectrode.composition[0].kineticConstant['value']); type = cell.positiveElectrode.composition[0].kineticConstant['interpolation_type']

            if 'interpolation_options' in cell.positiveElectrode.composition[0].kineticConstant:
                opts = cell.positiveElectrode.composition[0].kineticConstant['interpolation_options']
            else:
                opts = None

            self.k_0_c = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.positiveElectrode.composition[0].kineticConstant['type'] == 'function':
            self.k_0_c = lambda x: eval(cell.positiveElectrode.composition[0].kineticConstant['value']) 
            #raise NameError('User defined functions not handled yet')
        elif cell.positiveElectrode.composition[0].kineticConstant['type'] == 'constant':
            self.k_0_c = Constant(cell.positiveElectrode.composition[0].kineticConstant['value'])
        else:
            raise NameError('Unknown variable type')

        # arrhenius
        self.k_0_a_Ea = Constant(cell.negativeElectrode.composition[0].kineticConstant_Ea['value'])
        self.k_0_c_Ea = Constant(cell.positiveElectrode.composition[0].kineticConstant_Ea['value'])
        self.k_0_a_Tref = Constant(cell.negativeElectrode.composition[0].kineticConstant_Tref['value'])
        self.k_0_c_Tref = Constant(cell.positiveElectrode.composition[0].kineticConstant_Tref['value'])

        self.c_s_a_max = Constant(cell.negativeElectrode.composition[0].maximumConcentration)
        self.c_s_c_max = Constant(cell.positiveElectrode.composition[0].maximumConcentration)

        self.c_s_a_ini = Constant(cell.negativeElectrode.composition[0].initialConcentration)
        self.c_s_c_ini = Constant(cell.positiveElectrode.composition[0].initialConcentration)

        if cell.negativeElectrode.composition[0].diffusionConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.negativeElectrode.composition[0].diffusionConstant['value']); type = cell.negativeElectrode.composition[0].diffusionConstant['interpolation_type']

            if 'interpolation_options' in cell.negativeElectrode.composition[0].diffusionConstant:
                opts = cell.negativeElectrode.composition[0].diffusionConstant['interpolation_options']
            else:
                opts = None

            self.D_s_a = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.negativeElectrode.composition[0].diffusionConstant['type'] == 'function':
            self.D_s_a = lambda x: eval(cell.negativeElectrode.composition[0].diffusionConstant['value'])
        elif cell.negativeElectrode.composition[0].diffusionConstant['type'] == 'constant':
            self.D_s_a = Constant(cell.negativeElectrode.composition[0].diffusionConstant['value'])
        else:
            raise NameError('Unknown variable type')

        if cell.positiveElectrode.composition[0].diffusionConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.positiveElectrode.composition[0].diffusionConstant['value']); type = cell.positiveElectrode.composition[0].diffusionConstant['interpolation_type']

            if 'interpolation_options' in cell.positiveElectrode.composition[0].diffusionConstant:
                opts = cell.positiveElectrode.composition[0].diffusionConstant['interpolation_options']
            else:
                opts = None

            self.D_s_c = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.positiveElectrode.composition[0].diffusionConstant['type'] == 'function':
            self.D_s_c = lambda x: eval(cell.positiveElectrode.composition[0].diffusionConstant['value'])
        elif cell.positiveElectrode.composition[0].diffusionConstant['type'] == 'constant':
            self.D_s_c = Constant(cell.positiveElectrode.composition[0].diffusionConstant['value'])
        else:
            raise NameError('Unknown variable type')

        # arrhenius
        self.D_s_a_Ea = Constant(cell.negativeElectrode.composition[0].diffusionConstant_Ea['value'])
        self.D_s_c_Ea = Constant(cell.positiveElectrode.composition[0].diffusionConstant_Ea['value'])
        self.D_s_a_Tref = Constant(cell.negativeElectrode.composition[0].diffusionConstant_Tref['value'])
        self.D_s_c_Tref = Constant(cell.positiveElectrode.composition[0].diffusionConstant_Tref['value'])

        self.R = Constant(8.7350e0)
        self.F = Constant(9.7700e4)

        self.T_ini = Constant(cell.initialTemperature); self.Q = Constant(cell.capacity); self.area = Constant(cell.area)
        self.T_ext = Constant(cell.exteriorTemperature)

        self.alpha = Constant(0.5)

        self.a_s_a = []
        self.a_s_c = []

        for i in range(len(self.eps_s_a)):
            self.a_s_a.append(3. * self.eps_s_a[i] / self.R_s_a[i])
        for i in range(len(self.eps_s_c)):
            self.a_s_c.append(3. * self.eps_s_c[i] / self.R_s_c[i])

        if cell.negativeElectrode.composition[0].openCircuitPotential['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.negativeElectrode.composition[0].openCircuitPotential['value']); type = cell.negativeElectrode.composition[0].openCircuitPotential['interpolation_type']

            if 'interpolation_options' in cell.negativeElectrode.composition[0].openCircuitPotential:
                opts = cell.negativeElectrode.composition[0].openCircuitPotential['interpolation_options']
            else:
                opts = None

            self.U_a = batFEM.class_battery_model.get_interpolation(xy, type, opts)
        else:
            self.U_a = lambda x: eval(cell.negativeElectrode.composition[0].openCircuitPotential['value'])

        if cell.positiveElectrode.composition[0].openCircuitPotential['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.positiveElectrode.composition[0].openCircuitPotential['value']); type = cell.positiveElectrode.composition[0].openCircuitPotential['interpolation_type']

            if 'interpolation_options' in cell.positiveElectrode.composition[0].openCircuitPotential:
                opts = cell.positiveElectrode.composition[0].openCircuitPotential['interpolation_options']
            else:
                opts = None

            self.U_c = batFEM.class_battery_model.get_interpolation(xy, type, opts)
        else:
            self.U_c = lambda x: eval(cell.positiveElectrode.composition[0].openCircuitPotential['value'])

    def initial_guess(self):
        pass

    def build_brg(self):

        self.D_e_a = self.get_brug_e_a(self.D_e(self.c_e_a_1, self.T_a_1))
        self.D_e_s = self.get_brug_e_s(self.D_e(self.c_e_s_1, self.T_s_1))
        self.D_e_c = self.get_brug_e_c(self.D_e(self.c_e_c_1, self.T_c_1))

        self.kappa_a = self.get_brug_e_a(self.kappa(self.c_e_a_1, self.T_a_1))
        self.kappa_s = self.get_brug_e_s(self.kappa(self.c_e_s_1, self.T_s_1))
        self.kappa_c = self.get_brug_e_c(self.kappa(self.c_e_c_1, self.T_c_1))

        self.t_p_a = self.t_p(self.c_e_a_1)
        self.t_p_s = self.t_p(self.c_e_s_1)
        self.t_p_c = self.t_p(self.c_e_c_1)

        if self.compute_sigma_a_eff:
            self.sigma_a = self.get_brug_s_a(self.sigma_a)
        if self.compute_sigma_c_eff:
            self.sigma_c = self.get_brug_s_c(self.sigma_c)
    
    def build_fs(self):
        pass

    def build_wf_0(self):
        pass

    def build_wf_ie(self):
        pass

    def build_wf_rk(self):
        pass
