from unicodedata import name
from dolfin import *; import numpy; import matplotlib.pyplot as plt; import json

import scipy

class Material:

    def __init__(self, data):

        self.active = 0

        self.density = data['density']['value']; self.weightFraction = data['weightFraction']['value']
        self.volumeFraction = data['volumeFraction']['value']

        self.electronicConductivity = data['electronicConductivity']['value']
        self.thermalConducitivty = data['thermalConductivity']['value']
        self.heatCapacity = data['heatCapacity']['value']

class ActiveMaterial(Material):

    def __init__(self, data, SOC_ini):

        Material.__init__(self, data)

        self.active = 1

        self.theoreticalCapacity = data['theoreticalCapacity']['value']; self.particleRadius = data['particleRadius']['value']; self.diffusionConstant = data['diffusionConstant']
        self.filmRadius = data['filmRadius']['value']
        self.seiIonicConductivity = data['seiIonicConductivity']['value']
        self.seiElectronicConductivity = data['seiElectronicConductivity']['value']
        self.lplElectronicConductivity = data['lplElectronicConductivity']['value']
        self.alpha_c = data['alpha_c']['value']
        

        if 'seiOCP' in data:
            self.seiOCP = data['seiOCP']['value']
            self.seiKineticConstant = data['seiKineticConstant']['value']
            self.seiMass = data['seiMass']['value']
            self.seiDensity = data['seiDensity']['value']
            
        if 'seiAlpha_c' in data:
            self.seiAlpha_c = data['seiAlpha_c']['value']
        else:
            self.seiAlpha_c = 1.0
        
        if 'lplOCP' in data:
            self.lplOCP = data['lplOCP']['value']
            self.lplExchangeCurrent = data['lplExchangeCurrent']['value']
            self.lplStripExchangeCurrent = data['lplStripExchangeCurrent']['value'] if 'lplStripExchangeCurrent' in data else self.lplExchangeCurrent
            self.lplMass = data['lplMass']['value']
            self.lplDensity = data['lplDensity']['value']
        
        if 'lplAlpha_c' in data:
            self.lplAlpha_c = data['lplAlpha_c']['value']
        else:
            self.lplAlpha_c = 1.0

        if 'lplSigmoidCoefficient' in data:
            self.lplSigmoidCoefficient = data['lplSigmoidCoefficient']['value']
        else:
            self.lplSigmoidCoefficient = 1.0

        self.kineticConstant = data['kineticConstant']

        if 'arrhenius' in data['kineticConstant']:
            self.kineticConstant_Ea = data['kineticConstant']['arrhenius']['activationEnergy']
            self.kineticConstant_Tref = data['kineticConstant']['arrhenius']['referenceTemperature']
        else:
            self.kineticConstant_Ea = 0.
            self.kineticConstant_Tref = 298.15

        if 'arrhenius' in data['diffusionConstant']:
            self.diffusionConstant_Ea = data['diffusionConstant']['arrhenius']['activationEnergy']
            self.diffusionConstant_Tref = data['diffusionConstant']['arrhenius']['referenceTemperature']
        else:
            self.diffusionConstant_Ea = 0.
            self.diffusionConstant_Tref = 298.15

        if data['compute_maximumConcentration'] == 1:
            self.maximumConcentration = self.theoreticalCapacity * self.density * 3600. / 96485.3329
        else:
            self.maximumConcentration = data['maximumConcentration']['value']

        self.stoichiometry0 = data['stoichiometry0']['value']
        self.stoichiometry1 = data['stoichiometry1']['value']
        if data['compute_initialConcentration'] == 1:
            self.initialConcentration = self.maximumConcentration * (SOC_ini * (self.stoichiometry1 - self.stoichiometry0) + self.stoichiometry0)
        else:
            self.initialConcentration = data['initialConcentration']['value']

        self.openCircuitPotential = data['openCircuitPotential']

class Domain:

    def set_volume(self):
        self.volume = self.thickness * self.area

    def set_weight(self):
        self.weight = self.density * self.volume

class PorousDomain(Domain):

    def set_volume(self):
        self.volume = self.thickness * self.area

    def set_weight(self):
        self.weight = self.density * self.volume * (1 -  self.porosity)

def parse_json(cell_json, path='json_battery/'):

    cell_json = parse_field(cell_json, path, 'negativeCurrentCollector')
    cell_json = parse_field(cell_json, path, 'negativeElectrode')
    cell_json = parse_field(cell_json, path, 'separator'); cell_json = parse_field(cell_json, path, 'electrolyte')
    cell_json = parse_field(cell_json, path, 'positiveElectrode')
    cell_json = parse_field(cell_json, path, 'positiveCurrentCollector')

    return cell_json

def parse_field(cell_json, path, field):

    if 'file' in cell_json[field]:

        with open(path+cell_json[field]['file']) as data_file:

            field_json = json.load(data_file)

            for key in field_json:
                if key in cell_json[field]:
                    pass
                else:
                    cell_json[field][key] = field_json[key]
            
            if 'composition' in field_json:
                for i, material in enumerate(field_json['composition']):
                    for key in material:
                        if key in cell_json[field]['composition'][i]:
                            pass
                        else:
                            cell_json[field]['composition'][i][key] = material[key]

    return cell_json

def vf_formulation(cell_json, compute_stoichiometries=False, num_electrodes_n=1,num_electrodes_p=1):

    # Requirements
    epss_p = cell_json['positiveElectrode']['composition'][0]['volumeFraction']['value']
    epsi_p = cell_json['positiveElectrode']['composition'][1]['volumeFraction']['value']
    #epse_p = cell_json['positiveElectrode']['porosity']['value']
    rhos_p = cell_json['positiveElectrode']['composition'][0]['density']['value']
    rhoi_p = cell_json['positiveElectrode']['composition'][1]['density']['value']
    L_p = cell_json['positiveElectrode']['thickness']['value']

    epss_n = cell_json['negativeElectrode']['composition'][0]['volumeFraction']['value']
    epsi_n = cell_json['negativeElectrode']['composition'][1]['volumeFraction']['value']
    #epse_n = cell_json['negativeElectrode']['porosity']['value']
    rhos_n = cell_json['negativeElectrode']['composition'][0]['density']['value']
    rhoi_n = cell_json['negativeElectrode']['composition'][1]['density']['value']
    L_n = cell_json['negativeElectrode']['thickness']['value']

    # Computations
    epse_p = 1 - epss_p - epsi_p
    rho_p = epss_p*rhos_p + epsi_p*rhoi_p
    m_l_p = L_p * rho_p
    ws_p = epss_p*rhos_p/rho_p
    wi_p = epsi_p*rhoi_p/rho_p

    epse_n = 1 - epss_n - epsi_n
    rho_n = epss_n*rhos_n + epsi_n*rhoi_n
    m_l_n = L_n * rho_n
    ws_n = epss_n*rhos_n/rho_n
    wi_n = epsi_n*rhoi_n/rho_n

    # Assignations
    cell_json['positiveElectrode']['massLoading']['value'] = m_l_p
    cell_json['positiveElectrode']['density']['value'] = rho_p
    cell_json['positiveElectrode']['composition'][0]['weightFraction']['value'] = ws_p
    cell_json['positiveElectrode']['composition'][1]['weightFraction']['value'] = wi_p
    cell_json['positiveElectrode']['porosity']['value'] = epse_p

    cell_json['negativeElectrode']['massLoading']['value'] = m_l_n
    cell_json['negativeElectrode']['density']['value'] = rho_n
    cell_json['negativeElectrode']['composition'][0]['weightFraction']['value'] = ws_n
    cell_json['negativeElectrode']['composition'][1]['weightFraction']['value'] = wi_n
    cell_json['negativeElectrode']['porosity']['value'] = epse_n

    #print('neg. active wf:',cell_json['negativeElectrode']['composition'][0]['weightFraction']['value'])
    #print('neg. inactive wf:',cell_json['negativeElectrode']['composition'][1]['weightFraction']['value'])
    #print('neg. density:',cell_json['negativeElectrode']['density']['value'])
    #print('neg. mass loading:',cell_json['negativeElectrode']['massLoading']['value'])

    #print('pos. active wf:',cell_json['positiveElectrode']['composition'][0]['weightFraction']['value'])
    #print('pos. inactive wf:',cell_json['positiveElectrode']['composition'][1]['weightFraction']['value'])
    #print('pos. density:',cell_json['positiveElectrode']['density']['value'])
    #print('pos. mass loading:',cell_json['positiveElectrode']['massLoading']['value'])

    if compute_stoichiometries:
        cell_json = fcompute_stoichiometries(cell_json,num_electrodes_n,num_electrodes_p)
    else:
        cell_json = fcompute_preformlithium(cell_json,num_electrodes_n,num_electrodes_p)

    return cell_json


def wf_formulation(cell_json,compute_stoichiometries=False,num_electrodes_n=1,num_electrodes_p=1):

    # Requirements
    m_l_p = cell_json['positiveElectrode']['massLoading']['value']
    rho_p = cell_json['positiveElectrode']['density']['value']
    ws_p = cell_json['positiveElectrode']['composition'][0]['weightFraction']['value']
    rhos_p = cell_json['positiveElectrode']['composition'][0]['density']['value']
    rhoi_p = cell_json['positiveElectrode']['composition'][1]['density']['value']
    
    m_l_n = cell_json['negativeElectrode']['massLoading']['value']
    rho_n = cell_json['negativeElectrode']['density']['value']
    ws_n = cell_json['negativeElectrode']['composition'][0]['weightFraction']['value']
    rhos_n = cell_json['negativeElectrode']['composition'][0]['density']['value']
    rhoi_n = cell_json['negativeElectrode']['composition'][1]['density']['value']
    
    # Computations
    L_p = m_l_p / rho_p
    epss_p = ws_p * rho_p / rhos_p
    wi_p = 1 - ws_p
    epsi_p = wi_p * rho_p / rhoi_p
    #TODO eps_f
    epse_p = 1 - epss_p - epsi_p
    
    L_n = m_l_n / rho_n
    epss_n = ws_n * rho_n / rhos_n
    wi_n = 1 - ws_n
    epsi_n = wi_n * rho_n / rhoi_n
    #TODO eps_f
    epse_n = 1 - epss_n - epsi_n

    # Assignations
    cell_json['positiveElectrode']['thickness']['value'] = L_p
    cell_json['positiveElectrode']['composition'][0]['volumeFraction']['value'] = epss_p
    cell_json['positiveElectrode']['composition'][1]['weightFraction']['value'] = wi_p
    cell_json['positiveElectrode']['composition'][1]['volumeFraction']['value'] = epsi_p
    cell_json['positiveElectrode']['porosity']['value']=epse_p
   
    cell_json['negativeElectrode']['thickness']['value'] = L_n
    cell_json['negativeElectrode']['composition'][0]['volumeFraction']['value'] = epss_n
    cell_json['negativeElectrode']['composition'][1]['weightFraction']['value'] = wi_n
    cell_json['negativeElectrode']['composition'][1]['volumeFraction']['value'] = epsi_n
    cell_json['negativeElectrode']['porosity']['value']=epse_n

    #print('neg. active vf:',cell_json['negativeElectrode']['composition'][0]['volumeFraction']['value'])
    #print('neg. inactive vf:',cell_json['negativeElectrode']['composition'][1]['volumeFraction']['value'])
    #print('neg. porosity:',cell_json['negativeElectrode']['porosity']['value'])
    #print('neg. thickness:',cell_json['negativeElectrode']['thickness']['value'])

    #print('pos. active vf:',cell_json['positiveElectrode']['composition'][0]['volumeFraction']['value'])
    #print('pos. inactive vf:',cell_json['positiveElectrode']['composition'][1]['volumeFraction']['value'])
    #print('pos. porosity:',cell_json['positiveElectrode']['porosity']['value'])
    #print('pos. thickness:',cell_json['positiveElectrode']['thickness']['value'])

    if compute_stoichiometries:
        cell_json = fcompute_stoichiometries(cell_json,num_electrodes_n,num_electrodes_p)
    else:
        cell_json = fcompute_preformlithium(cell_json,num_electrodes_n,num_electrodes_p)

    return cell_json

def fcompute_preformlithium(cell_json,num_electrodes_n=1,num_electrodes_p=1):
    # Requirements
    cs_max_p = cell_json['positiveElectrode']['composition'][0]['maximumConcentration']['value']
    U_p=cell_json['positiveElectrode']['composition'][0]['openCircuitPotential']['value']
    L_p = cell_json['positiveElectrode']['thickness']['value']
    x_0_p = cell_json['positiveElectrode']['composition'][0]['stoichiometry0']['value']
    x_1_p = cell_json['positiveElectrode']['composition'][0]['stoichiometry1']['value']
    epss_p = cell_json['positiveElectrode']['composition'][0]['volumeFraction']['value']
    delta_film_p = cell_json['positiveElectrode']['composition'][0]['filmRadius']['value']
    M_sei_p = cell_json['positiveElectrode']['composition'][0]['seiMass']['value']
    rho_sei_p = cell_json['positiveElectrode']['composition'][0]['seiDensity']['value']
    R_p_p = cell_json['positiveElectrode']['composition'][0]['particleRadius']['value']

    cs_max_n = cell_json['negativeElectrode']['composition'][0]['maximumConcentration']['value']
    U_n=cell_json['negativeElectrode']['composition'][0]['openCircuitPotential']['value']
    L_n = cell_json['negativeElectrode']['thickness']['value']
    x_0_n = cell_json['negativeElectrode']['composition'][0]['stoichiometry0']['value']
    x_1_n = cell_json['negativeElectrode']['composition'][0]['stoichiometry1']['value']
    epss_n = cell_json['negativeElectrode']['composition'][0]['volumeFraction']['value']
    delta_film_n = cell_json['negativeElectrode']['composition'][0]['filmRadius']['value']
    M_sei_n = cell_json['negativeElectrode']['composition'][0]['seiMass']['value']
    rho_sei_n = cell_json['negativeElectrode']['composition'][0]['seiDensity']['value']
    R_p_n = cell_json['negativeElectrode']['composition'][0]['particleRadius']['value']

    #f_sei_n = cell_json['properties']['negativeFormationLithiumLoss']['value']
    #f_sei_p = cell_json['properties']['positiveFormationLithiumLoss']['value']
    
    A_n=cell_json['negativeElectrode']['area']['value']*num_electrodes_n
    A_p=cell_json['positiveElectrode']['area']['value']*num_electrodes_p

    # Computations
    Q_n=96485.3329*L_n*A_n*epss_n*cs_max_n*abs(x_1_n-x_0_n)/3600
    Q_p=96485.3329*L_p*A_p*epss_p*cs_max_p*abs(x_1_p-x_0_p)/3600

    a_s_n = 3*epss_n/R_p_n
    a_s_p = 3*epss_p/R_p_p

    r_np = Q_n/Q_p

    n_Li = L_n*epss_n*cs_max_n*x_0_n/r_np+L_p*epss_p*cs_max_p*x_0_p

    U_n_fun = lambda x: eval(U_n.replace('exp', 'numpy.exp').replace('tanh', 'numpy.tanh'))
    U_p_fun = lambda x: eval(U_p.replace('exp', 'numpy.exp').replace('tanh', 'numpy.tanh'))

    V_min = U_p_fun(x_0_p) - U_n_fun(x_0_n)
    V_max = U_p_fun(x_1_p) - U_n_fun(x_1_n)

    #print("neg. electrode capacity:",Q_n)
    #print("pos. electrode capacity:",Q_p)
    #print("neg./pos. ratio:",r_np)
    #print("V_min:",V_min)
    #print("V_max:",V_max)
    #print("number of moles:",n_Li)

    def f_1(x):
        n_Li_0=x[0]
        f_sei_n=x[1]
        f_sei_p=x[2]

        y = numpy.zeros(3)

        y[0] = n_Li_0 - n_Li/(1-f_sei_n-f_sei_p)
        y[1] = f_sei_n - (delta_film_n/n_Li_0) * (rho_sei_n*a_s_n*A_n*L_n)/(M_sei_n)
        y[2] = f_sei_p - (delta_film_p/n_Li_0) * (rho_sei_p*a_s_p*A_p*L_p)/(M_sei_p)

        return y
    
    f_sei = scipy.optimize.fsolve(f_1, [n_Li, 0.0, 0.0])

    n_Li_0 = f_sei[0]

    f_sei_n = f_sei[1]
    f_sei_p = f_sei[2]

    #print("number of moles (preform):",n_Li_0)
    #print("sei form n:",f_sei_n)
    #print("sei form p:",f_sei_p)

    # Assignations
    cell_json['properties']['preformationLithium']['value'] = n_Li_0
    cell_json['properties']['negativeFormationLithiumLoss']['value'] = f_sei_n
    cell_json['properties']['positiveFormationLithiumLoss']['value'] = f_sei_p
    cell_json['properties']['maxVoltage']['value'] = V_max
    cell_json['properties']['minVoltage']['value'] = V_min
    cell_json['properties']['negative2positiveCapacityRatio']['value'] = r_np

    return cell_json

def fcompute_stoichiometries(cell_json,num_electrodes_n=1,num_electrodes_p=1):
    # Requirements
    cs_max_p = cell_json['positiveElectrode']['composition'][0]['maximumConcentration']['value']
    U_p=cell_json['positiveElectrode']['composition'][0]['openCircuitPotential']['value']
    M_sei_p = cell_json['positiveElectrode']['composition'][0]['seiMass']['value']
    rho_sei_p = cell_json['positiveElectrode']['composition'][0]['seiDensity']['value']
    R_p_p = cell_json['positiveElectrode']['composition'][0]['particleRadius']['value']
    L_p = cell_json['positiveElectrode']['thickness']['value']
    epss_p = cell_json['positiveElectrode']['composition'][0]['volumeFraction']['value']

    cs_max_n = cell_json['negativeElectrode']['composition'][0]['maximumConcentration']['value']
    U_n=cell_json['negativeElectrode']['composition'][0]['openCircuitPotential']['value']
    M_sei_n = cell_json['negativeElectrode']['composition'][0]['seiMass']['value']
    rho_sei_n = cell_json['negativeElectrode']['composition'][0]['seiDensity']['value']
    R_p_n = cell_json['negativeElectrode']['composition'][0]['particleRadius']['value']
    L_n = cell_json['negativeElectrode']['thickness']['value']
    epss_n = cell_json['negativeElectrode']['composition'][0]['volumeFraction']['value']

    n_Li_0 = cell_json['properties']['preformationLithium']['value']
    f_sei_n = cell_json['properties']['negativeFormationLithiumLoss']['value']
    f_sei_p = cell_json['properties']['positiveFormationLithiumLoss']['value']
    V_min = cell_json['properties']['minVoltage']['value']
    V_max = cell_json['properties']['maxVoltage']['value']
    r_pn = cell_json['properties']['negative2positiveCapacityRatio']['value']

    A_n=cell_json['negativeElectrode']['area']['value']*num_electrodes_n
    A_p=cell_json['positiveElectrode']['area']['value']*num_electrodes_p

    # Computations
    f_sei = f_sei_n + f_sei_p
    
    U_n_fun = lambda x: eval(U_n.replace('exp', 'numpy.exp').replace('tanh', 'numpy.tanh'))
    U_p_fun = lambda x: eval(U_p.replace('exp', 'numpy.exp').replace('tanh', 'numpy.tanh'))

    n_Li = (1-f_sei)*n_Li_0

    def f_1(x):
        x_0_n=x[0]
        x_0_p=x[1]

        y = numpy.zeros(2)
        y[0]= L_n*epss_n*cs_max_n*x_0_n/r_pn+L_p*epss_p*cs_max_p*x_0_p - n_Li
        y[1]= U_p_fun(x_0_p)-U_n_fun(x_0_n) - V_min
        return y

    x_0 = scipy.optimize.fsolve(f_1, [0.01, 0.90])
    #print('  0% stoichiometries:',x_0)
    x_0_n = x_0[0]
    x_0_p = x_0[1]

    def f_2(x):
        x_1_n=x[0]
        x_1_p=x[1]

        y = numpy.zeros(2)
        y[0]= L_n*epss_n*cs_max_n*x_1_n/r_pn+L_p*epss_p*cs_max_p*x_1_p - n_Li
        y[1]= U_p_fun(x_1_p)-U_n_fun(x_1_n) - V_max
        return y

    x_1 = scipy.optimize.fsolve(f_2, [0.90, 0.01])
    #print('100% stoichiometries:',x_1,f_2(x_1))
    x_1_n = x_1[0]
    x_1_p = x_1[1]

    a_s_n = 3*epss_n/R_p_n
    a_s_p = 3*epss_p/R_p_p

    delta_film_n = f_sei_n*n_Li_0*M_sei_n/(rho_sei_n*a_s_n*A_n*L_n)
    delta_film_p = f_sei_p*n_Li_0*M_sei_p/(rho_sei_p*a_s_p*A_p*L_p)

    Q_n=96485.3329*L_n*A_n*epss_n*cs_max_n*abs(x_1_n-x_0_n)/3600
    Q_p=96485.3329*L_p*A_p*epss_p*cs_max_p*abs(x_1_p-x_0_p)/3600

    # Assignations
    cell_json['negativeElectrode']['composition'][0]['stoichiometry0']['value'] = x_0_n
    cell_json['negativeElectrode']['composition'][0]['stoichiometry1']['value'] = x_1_n
    cell_json['positiveElectrode']['composition'][0]['stoichiometry0']['value'] = x_0_p
    cell_json['positiveElectrode']['composition'][0]['stoichiometry1']['value'] = x_1_p

    cell_json['negativeElectrode']['composition'][0]['filmRadius']['value'] = delta_film_n
    cell_json['positiveElectrode']['composition'][0]['filmRadius']['value'] = delta_film_p

    #print('delta_film_n:', delta_film_n)
    #print('delta_film_p:', delta_film_p)

    #print("neg. electrode capacity:",Q_n)
    #print("pos. electrode capacity:",Q_p)
    #print("pos./neg. ratio:",Q_n/Q_p)

    return cell_json

class Cell:

    def __init__(self, json, temperature, SoC, formulation='vf', compute_stoichiometries=False):

        self.initialTemperature = temperature
        self.exteriorTemperature = temperature
        self.heatConvectionCoefficient = json['properties']['heatConvectionCoefficient']['value']
        self.heatConvectionArea = json['properties']['heatConvectionArea']['value']

        num_layers_n = json['properties']['negativeElectrodeLayers']['value']
        num_layers_p = json['properties']['positiveElectrodeLayers']['value']

        if num_layers_n == num_layers_p:
            num_electrodes_n = 2*(num_layers_n-1) + 1
            num_electrodes_p = 2*(num_layers_p-1) + 1
        elif num_layers_p < num_layers_n:
            num_electrodes_n = 2*(num_layers_n-2) + 2
            num_electrodes_p = 2*num_layers_p
        else:
            num_electrodes_p = 2*(num_layers_p-2) + 2
            num_electrodes_n = 2*num_layers_n

        num_separators = min(num_electrodes_n,num_electrodes_p)

        #TODO checks and vf formulation
        if formulation == 'wf':
            json = wf_formulation(json, compute_stoichiometries, num_electrodes_n,num_electrodes_p)
        if formulation == 'vf':
            json = vf_formulation(json, compute_stoichiometries, num_electrodes_n,num_electrodes_p)

        self.negativeCurrentCollector = CurrentCollector(json['negativeCurrentCollector'],num_layers=num_layers_n)

        self.SOC_ini = SoC

        if json['negativeElectrode']['type'] == 'PE':
            self.negativeElectrode = PorousElectrode(json['negativeElectrode'], self.SOC_ini, num_layers=num_electrodes_n)
        else:
            if json['negativeElectrode']['type'] == 'ME':
                self.negativeElectrode = FoilElectrode(json['negativeElectrode'], self.SOC_ini, num_layers=num_electrodes_n)
            else:
                raise NameError('Unknown type of negative electrode. Choose PE (porous electode) or ME (metal electrode)')

        self.separator = Separator(json['separator'],num_layers=num_separators); self.electrolyte = Electrolyte(json['electrolyte'])

        if json['positiveElectrode']['type'] == 'PE':
            self.positiveElectrode = PorousElectrode(json['positiveElectrode'], self.SOC_ini, num_layers=num_electrodes_p)
        else:
            if json['positiveElectrode']['type'] == 'ME':
                self.positiveElectrode = FoilElectrode(json['positiveElectrode'], self.SOC_ini, num_layers=num_electrodes_p)
            else:
                raise NameError('Unknown type of positive electrode. Choose PE (porous electode) or ME (metal electrode)')

        self.positiveCurrentCollector = CurrentCollector(json['positiveCurrentCollector'],num_layers=num_layers_p)

        self.fill_electrolyte()

        self.volume = self.negativeCurrentCollector.volume \
                    + self.negativeElectrode.volume * (1 - self.negativeElectrode.porosity) \
                    + self.separator.volume * (1 - self.separator.porosity) + self.electrolyte.volume \
                    + self.positiveElectrode.volume * (1 - self.positiveElectrode.porosity) \
                    + self.positiveCurrentCollector.volume

        self.weight = self.negativeCurrentCollector.weight \
                    + self.negativeElectrode.weight \
                    + self.separator.weight + self.electrolyte.weight \
                    + self.positiveElectrode.weight \
                    + self.positiveCurrentCollector.weight
        

        #volume_dict = {
        #"Neg CC": self.negativeCurrentCollector.volume,
        #"Neg electrode": self.negativeElectrode.volume * (1 - self.negativeElectrode.porosity),
        #"Separator": self.separator.volume * (1 - self.separator.porosity),
        #"Electrolyte": self.electrolyte.volume,
        #"Pos electrode": self.positiveElectrode.volume * (1 - self.positiveElectrode.porosity),
        #"Pos CC": self.positiveCurrentCollector.volume,
        #}

        #weight_dict = {
        #"Neg CC": self.negativeCurrentCollector.weight,
        #"Neg electrode": self.negativeElectrode.weight,
        #"Separator": self.separator.weight,
        #"Electrolyte": self.electrolyte.weight,
        #"Pos electrode": self.positiveElectrode.weight,
        #"Pos CC": self.positiveCurrentCollector.weight,
        #}

        #fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # --- Volume pie ---
        #axes[0].pie(
        #volume_dict.values(),
        #labels=volume_dict.keys(),
        #autopct='%1.1f%%',
        #startangle=90
        #)
        #axes[0].set_title(f"Marquis2019 — Volume")

        # --- Weight pie ---
        #axes[1].pie(
        #labels=weight_dict.keys(),
        #weight_dict.values(),
        #autopct='%1.1f%%',
        #startangle=90
        #)
        #axes[1].set_title(f"Marquis2019 — Weight")

        #plt.tight_layout()
        #plt.show()

        self.density = self.weight / self.volume

        self.heatCapacity = self.negativeCurrentCollector.heatCapacity * self.negativeCurrentCollector.weight \
                          + self.negativeElectrode.heatCapacity * self.negativeElectrode.weight \
                          + self.separator.heatCapacity * self.separator.weight + self.electrolyte.heatCapacity * self.electrolyte.weight \
                          + self.positiveElectrode.heatCapacity * self.positiveElectrode.weight \
                          + self.positiveCurrentCollector.heatCapacity * self.positiveCurrentCollector.weight
        self.heatCapacity = self.heatCapacity / self.weight

        if json['properties']['nominalCapacity']['value'] == 'compute':
            self.capacity = min(self.negativeElectrode.capacity, self.positiveElectrode.capacity)
        else:
            self.capacity = json['properties']['nominalCapacity']['value']
        
        if json['properties']['area']['value'] == 'compute':
            self.area = max(self.negativeElectrode.area, self.positiveElectrode.area)
        else:
            self.area = json['properties']['area']['value']
            
        
    def fill_electrolyte(self):

        self.electrolyte.volume = self.negativeElectrode.volume * self.negativeElectrode.porosity \
                                + self.separator.volume * self.separator.porosity \
                                + self.positiveElectrode.volume * self.positiveElectrode.porosity

        self.electrolyte.weight = self.electrolyte.volume * self.electrolyte.density

class CurrentCollector(Domain):

    def __init__(self, json_data, num_layers=1):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']*num_layers

        self.density = json_data['density']['value']
        self.heatCapacity = json_data['heatCapacity']['value']

        self.set_volume()
        self.set_weight()

class Separator(PorousDomain):

    def __init__(self, json_data, num_layers=1):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']*num_layers

        self.thermalConductivity = json_data['thermalConductivity']['value']
        self.heatCapacity = json_data['heatCapacity']['value']

        self.porosity = json_data['porosity']['value']
        self.bruggeman = json_data['bruggeman']['value']

        self.density = json_data['density']['value']

        self.set_volume()
        self.set_weight()

class Electrolyte:

    def __init__(self, json_data):

        self.name = json_data['name']

        self.density = json_data['density']['value']
        self.heatCapacity = json_data['heatCapacity']['value']

        self.diffusionConstant = json_data['diffusionConstant']; self.ionicConductivity = json_data['ionicConductivity']; self.transferenceNumber = json_data['transferenceNumber']

        if 'arrhenius' in self.diffusionConstant:
            self.diffusionConstant_Ea = json_data['diffusionConstant']['arrhenius']['activationEnergy']['value']
            self.diffusionConstant_Tref = json_data['diffusionConstant']['arrhenius']['referenceTemperature']['value']
        else:
            self.diffusionConstant_Ea = 0
            self.diffusionConstant_Tref = 298.15

        if 'arrhenius' in self.ionicConductivity:
            self.ionicConductivity_Ea = json_data['ionicConductivity']['arrhenius']['activationEnergy']['value']
            self.ionicConductivity_Tref = json_data['ionicConductivity']['arrhenius']['referenceTemperature']['value']
        else:
            self.ionicConductivity_Ea = 0
            self.ionicConductivity_Tref = 298.15

        self.initialConcentration = json_data['initialConcentration']['value']

        self.solventDiffusionConstant = json_data['solventDiffusionConstant']['value']
        self.solventConcentration = json_data['solventConcentration']['value']

class PorousElectrode(PorousDomain):

    def __init__(self, json_data, SOC_ini, capacityRatio=None, capacity=None, num_layers=1):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']*num_layers

        self.porosity = json_data['porosity']['value']
        self.electrolyteBruggeman = json_data['electrolyteBruggeman']['value']
        self.electrodeBruggeman = json_data['electrodeBruggeman']['value']

        self.composition = []

        for material in json_data['composition']:
            if material['active'] == 1:
                self.composition.append(ActiveMaterial(material, SOC_ini))
            else:
                self.composition.append(Material(material))

        self.set_density(json_data)


        for material in self.composition:
            if material.volumeFraction == 'compute':
                material.volumeFraction = material.weightFraction * self.density / material.density
            if material.weightFraction == 'compute':
                material.weightFraction = material.volumeFraction * material.density / self.density

        #if capacityRatio == None:
        #    pass
        #else:
        #    sum = 0
        #    for material in self.composition:
        #        if material.active == 1:
        #            sum += material.theoreticalCapacity * material.weightFraction
        #
        #    self.thickness = capacityRatio * capacity / (sum * self.density * (1 - self.porosity) * self.area )

        self.set_volume()
        self.set_weight()

        self.set_capacity()

        if json_data['electronicConductivity']['value']=='compute':
            self.electronicConductivity = 0
            for material in self.composition:
                self.electronicConductivity += material.volumeFraction * material.electronicConductivity
            self.electronicConductivity = self.electronicConductivity / (1-self.porosity)
            self.compute_effective_electronicConductivity = True
        else:
            self.electronicConductivity = json_data['electronicConductivity']['value']
            self.compute_effective_electronicConductivity = False

        if json_data['thermalConductivity']['value']=='compute':
            self.thermalConductivity = 0
            for material in self.composition:
                self.thermalConductivity += material.volumeFraction * material.thermalConductivity
            self.thermalConductivity = self.thermalConductivity / (1-self.porosity)
        else:
            self.thermalConductivity = json_data['thermalConductivity']['value']

        if json_data['heatCapacity']['value']=='compute':
            self.heatCapacity = 0
            for material in self.composition:
                self.heatCapacity += material.volumeFraction * material.heatCapacity
            self.heatCapacity = self.heatCapacity / (1-self.porosity)
        else:
            self.heatCapacity = json_data['heatCapacity']['value']


    def set_density(self, json_data):
        if json_data['density']['value'] == 'compute':
            self.density = (1 - self.porosity) / sum([material.weightFraction / material.density for material in self.composition])
        else:
            self.density = json_data['density']['value']

    def set_capacity(self):

        self.capacity = 0

        #for material in self.composition:
        #    if material.active == 1:
        #        self.capacity += material.practicalCapacity * material.weightFraction * self.weight
        #    else:
        #        pass

        for material in self.composition:
            
            if material.active == 1:
                self.capacity += self.area * 96485.3329 * self.thickness * material.volumeFraction * material.maximumConcentration * abs(material.stoichiometry1 - material.stoichiometry0) / 3600.

class FoilElectrode(Domain):

    def __init__(self, json_data, num_layers=1):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']*num_layers

        self.density = json_data['density']['value']

        self.porosity = 0

        self.set_volume()
        self.set_weight()

        self.capacity = json_data['theoreticalCapacity']['value'] * self.weight

        self.kineticConstant = json_data['kineticConstant']['value']

        if 'arrhenius' in json_data['kineticConstant']:
            self.kineticConstant_Ea = json_data['kineticConstant']['arrhenius']['activationEnergy']
            self.kineticConstant_Tref = json_data['kineticConstant']['arrhenius']['referenceTemperature']
        else:
            self.kineticConstant_Ea = 0.
            self.kineticConstant_Tref = 298.15
