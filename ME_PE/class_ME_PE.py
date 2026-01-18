from dolfin import *; import numpy; import matplotlib.pyplot as plt; import json#; from dolfin_adjoint import *

import batFEM.class_battery_model

class ME_PE(batFEM.class_battery_model.Model):

    def __init__(self, cell, simulation_options):

        self.N_x = simulation_options['space discretization']['elements']; self.FEM_order = simulation_options['space discretization']['FEM order']

        self.SGM_order_a = simulation_options['space discretization']['SGM order']
        self.SGM_order_c = simulation_options['space discretization']['SGM order']

        self.solve_thermal = simulation_options['multiphysics']['thermal']

        self.build_sgm(); self.build_mesh()

        self.t_list = []
        self.i_list = []
        self.v_list = []

        self.R_s_c = []

        self.eps_s_c = []

        for material in cell.positiveElectrode.composition:
            if material.active == 1:
                self.R_s_c.append(Constant(material.particleRadius))
                self.eps_s_c.append(Constant(material.volumeFraction))

        self.eps_e_s = Constant(cell.separator.porosity)
        self.eps_e_c = Constant(cell.positiveElectrode.porosity)

        self.bruggeman_s = Constant(cell.separator.bruggeman)
        self.bruggeman_c = Constant(cell.positiveElectrode.bruggeman)

        self.tortuosity_e_s = self.eps_e_s ** (1 - self.bruggeman_s)
        self.tortuosity_e_c = self.eps_e_c ** (1 - self.bruggeman_c)

        self.tortuosity_s_c = (1 - self.eps_e_c) ** (1 - self.bruggeman_c)

        self.sigma_c = Constant(cell.positiveElectrode.electronicConductivity)

        self.k_t_s = Constant(cell.separator.thermalConductivity)
        self.k_t_c = Constant(cell.positiveElectrode.thermalConductivity)

        self.rho_s = Constant(cell.separator.density)
        self.rho_c = Constant(cell.positiveElectrode.density)

        self.c_p_s = Constant(cell.separator.specificHeat)
        self.c_p_c = Constant(cell.positiveElectrode.specificHeat)

        self.h_t = Constant(cell.heatConvection)

        self.L_s = Constant(cell.separator.thickness)
        self.L_c = Constant(cell.positiveElectrode.thickness)

        self.c_e_ini = Constant(cell.electrolyte.initialConcentration)

        if cell.electrolyte.diffusionConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.electrolyte.diffusionConstant['value']); type = cell.electrolyte.diffusionConstant['interpolation_type']

            if 'interpolation_options' in cell.electrolyte.diffusionConstant:
                opts = cell.electrolyte.diffusionConstant['interpolation_options']
            else:
                opts = None

            self.D_e = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.electrolyte.diffusionConstant['type'] == 'function':
            raise NameError('User defined functions not handled yet')
        elif cell.electrolyte.diffusionConstant['type'] == 'constant':
            self.D_e = Constant(cell.electrolyte.diffusionConstant['value'])
        else:
            raise NameError('Unknown variable type')

        if cell.electrolyte.ionicConductivity['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.electrolyte.ionicConductivity['value']); type = cell.electrolyte.ionicConductivity['interpolation_type']

            if 'interpolation_options' in cell.electrolyte.ionicConductivity:
                opts = cell.electrolyte.ionicConductivity['interpolation_options']
            else:
                opts = None

            self.kappa = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.electrolyte.ionicConductivity['type'] == 'function':
            raise NameError('User defined functions not handled yet')
        elif cell.electrolyte.ionicConductivity['type'] == 'constant':
            self.kappa = Constant(cell.electrolyte.ionicConductivity['value'])
        else:
            raise NameError('Unknown variable type')

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

        self.k_0_a = Constant(cell.negativeElectrode.kineticConstant)

        if cell.positiveElectrode.composition[0].kineticConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.positiveElectrode.composition[0].kineticConstant['value']); type = cell.positiveElectrode.composition[0].kineticConstant['interpolation_type']

            if 'interpolation_options' in cell.positiveElectrode.composition[0].kineticConstant:
                opts = cell.positiveElectrode.composition[0].kineticConstant['interpolation_options']
            else:
                opts = None

            self.k_0_c = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.positiveElectrode.composition[0].kineticConstant['type'] == 'function':
            raise NameError('User defined functions not handled yet')
        elif cell.positiveElectrode.composition[0].kineticConstant['type'] == 'constant':
            self.k_0_c = Constant(cell.positiveElectrode.composition[0].kineticConstant['value'])
        else:
            raise NameError('Unknown variable type')

        # arrhenius
        self.k_0_a_Ea = Constant(cell.negativeElectrode.kineticConstant_Ea)
        self.k_0_c_Ea = Constant(cell.positiveElectrode.composition[0].kineticConstant_Ea)
        self.k_0_a_Tref = Constant(cell.negativeElectrode.kineticConstant_Tref)
        self.k_0_c_Tref = Constant(cell.positiveElectrode.composition[0].kineticConstant_Tref)

        self.c_s_c_max = Constant(cell.positiveElectrode.composition[0].maximumConcentration)

        self.c_s_c_ini = Constant(cell.positiveElectrode.composition[0].initialConcentration)

        if cell.positiveElectrode.composition[0].diffusionConstant['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.positiveElectrode.composition[0].diffusionConstant['value']); type = cell.positiveElectrode.composition[0].diffusionConstant['interpolation_type']

            if 'interpolation_options' in cell.positiveElectrode.composition[0].diffusionConstant:
                opts = cell.positiveElectrode.composition[0].diffusionConstant['interpolation_options']
            else:
                opts = None

            self.D_s_c = batFEM.class_battery_model.get_interpolation(xy, type, opts)

        elif cell.positiveElectrode.composition[0].diffusionConstant['type'] == 'function':
            raise NameError('User defined functions not handled yet')
        elif cell.positiveElectrode.composition[0].diffusionConstant['type'] == 'constant':
            self.D_s_c = Constant(cell.positiveElectrode.composition[0].diffusionConstant['value'])
        else:
            raise NameError('Unknown variable type')

        # arrhenius
        self.D_s_c_Ea = Constant(cell.positiveElectrode.composition[0].diffusionConstant_Ea)
        self.D_s_c_Tref = Constant(cell.positiveElectrode.composition[0].diffusionConstant_Tref)

        self.R = Constant(8.7350e0)
        self.F = Constant(9.7700e4)

        self.T_ini = Constant(cell.initialTemperature); self.Q = Constant(cell.capacity); self.area = Constant(min(cell.negativeElectrode.area, cell.positiveElectrode.area))
        self.T_ext = Constant(cell.exteriorTemperature)

        self.alpha = Constant(0.5)

        self.a_s_c = []

        for i in range(len(self.eps_s_c)):
            self.a_s_c.append(3. * self.eps_s_c[i] / self.R_s_c[i])

        if cell.positiveElectrode.composition[0].openCircuitPotential['type'] == 'interpolate':
            xy = numpy.loadtxt(cell.positiveElectrode.composition[0].openCircuitPotential['value']); type = cell.positiveElectrode.composition[0].openCircuitPotential['interpolation_type']

            if 'interpolation_options' in cell.positiveElectrode.composition[0].openCircuitPotential:
                opts = cell.positiveElectrode.composition[0].openCircuitPotential['interpolation_options']
            else:
                opts = None

            self.U_c = batFEM.class_battery_model.get_interpolation(xy, type, opts)

    def initial_guess(self):
        pass

    def build_brg(self):

        self.D_e_s = self.get_brug_e_s(self.D_e(self.c_e_s_1))
        self.D_e_c = self.get_brug_e_c(self.D_e(self.c_e_c_1))

        self.kappa_s = self.get_brug_e_s(self.kappa(self.c_e_s_1))
        self.kappa_c = self.get_brug_e_c(self.kappa(self.c_e_c_1))

        self.t_p_s = self.t_p(self.c_e_s_1)
        self.t_p_c = self.t_p(self.c_e_c_1)

        self.sigma_c = self.get_brug_s_c(self.sigma_c)

    def build_arr(self):

        self.k_0_a = self.get_arr_s(self.k_0_a, self.k_0_a_Ea, self.k_0_a_Tref)
        self.k_0_c = self.get_arr_c(self.k_0_c(self.c_s_c_sur[0]/self.c_s_c_max), self.k_0_c_Ea, self.k_0_c_Tref)

        self.D_s_c = self.get_arr_c(self.D_s_c(self.c_s_c_sur[0]/self.c_s_c_max), self.D_s_c_Ea, self.D_s_c_Tref)

        self.kappa_D_s = 2 * (self.R * self.T_s_1 / self.F) * (1 -  self.t_p_s) * self.kappa_s
        self.kappa_D_c = 2 * (self.R * self.T_c_1 / self.F) * (1 -  self.t_p_c) * self.kappa_c

    def build_fs(self):
        pass

    def build_wf_0(self):
        pass

    def build_wf_1(self):
        pass
