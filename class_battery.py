from dolfin import *; import numpy; import matplotlib.pyplot as plt; import json#; from dolfin_adjoint import *

class Material:

    def __init__(self, data):

        self.active = 0

        self.density = data['density']['value']; self.weightFraction = data['weightFraction']['value']
        self.volumeFraction = data['volumeFraction']['value']

class ActiveMaterial(Material):

    def __init__(self, data, SOC_ini):

        Material.__init__(self, data)

        self.active = 1

        self.theoreticalCapacity = data['theoreticalCapacity']['value']; self.particleRadius = data['particleRadius']['value']; self.diffusionConstant = data['diffusionConstant']
        self.filmRadius = data['filmRadius']['value']
        self.seiVolumeFraction = data['seiVolumeFraction']['value']
        self.seiIonicConductivity = data['seiIonicConductivity']['value']

        if 'seiOCP' in data:
            self.seiOCP = data['seiOCP']['value']
            self.seiKineticConstant = data['seiKineticConstant']['value']
            self.seiMass = data['seiMass']['value']
            self.seiDensity = data['seiDensity']['value']
        if 'lplOCP' in data:
            self.lplOCP = data['lplOCP']['value']
            self.lplExchangeCurrent = data['lplExchangeCurrent']['value']
            self.lplMass = data['lplMass']['value']
            self.lplDensity = data['lplDensity']['value']

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

    return cell_json

class Cell:

    def __init__(self, json, temperature, SOC):

        self.initialTemperature = temperature
        self.exteriorTemperature = temperature
        self.heatConvectionCoefficient = json['properties']['heatConvectionCoefficient']['value']
        self.heatConvectionArea = json['properties']['heatConvectionArea']['value']

        self.negativeCurrentCollector = CurrentCollector(json['negativeCurrentCollector'])

        self.SOC_ini = SOC

        if json['negativeElectrode']['type'] == 'PE':
            if 'negative2positiveRatio' in json['properties']:
                self.positiveElectrode = PorousElectrode(json['positiveElectrode'], self.SOC_ini)
                self.negativeElectrode = PorousElectrode(json['negativeElectrode'], self.SOC_ini, json['properties']['negative2positiveRatio']['value'], self.positiveElectrode.capacity)
            else:
                self.negativeElectrode = PorousElectrode(json['negativeElectrode'], self.SOC_ini)
        else:
            if json['negativeElectrode']['type'] == 'ME':
                self.negativeElectrode = FoilElectrode(json['negativeElectrode'])
            else:
                raise NameError('Unknown type of negative electrode. Choose PE (porous electode) or ME (metal electrode)')

        self.separator = Separator(json['separator']); self.electrolyte = Electrolyte(json['electrolyte'])

        if json['positiveElectrode']['type'] == 'PE':
            if 'positive2negativeRatio' in json['properties']:
                self.positiveElectrode = PorousElectrode(json['positiveElectrode'], self.SOC_ini, json['properties']['positive2negativeRatio']['value'], self.negativeElectrode.capacity)
            else:
                self.positiveElectrode = PorousElectrode(json['positiveElectrode'], self.SOC_ini)
        else:
            if json['positiveElectrode']['type'] == 'ME':
                self.positiveElectrode = FoilElectrode(json['positiveElectrode'])
            else:
                raise NameError('Unknown type of positive electrode. Choose PE (porous electode) or ME (metal electrode)')

        self.positiveCurrentCollector = CurrentCollector(json['positiveCurrentCollector'])

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

        if "capacity" in json['properties']:
            self.capacity = json['properties']['capacity']['value']
        else:
            self.capacity = min(self.negativeElectrode.capacity, self.positiveElectrode.capacity)
        
        if "area" in json['properties']:
            self.area = json['properties']['area']['value']
        else:
            self.area = min(self.negativeElectrode.area, self.positiveElectrode.area)

        print(self.negativeElectrode.capacity)
        print(self.positiveElectrode.capacity)
    def fill_electrolyte(self):

        self.electrolyte.volume = self.negativeElectrode.volume * self.negativeElectrode.porosity \
                                + self.separator.volume * self.separator.porosity \
                                + self.positiveElectrode.volume * self.positiveElectrode.porosity

        self.electrolyte.weight = self.electrolyte.volume * self.electrolyte.density

class CurrentCollector(Domain):

    def __init__(self, json_data):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']

        self.density = json_data['density']['value']

        self.set_volume()
        self.set_weight()

class Separator(PorousDomain):

    def __init__(self, json_data):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']

        self.thermalConductivity = json_data['thermalConductivity']['value']
        self.specificHeat = json_data['specificHeat']['value']

        self.porosity = json_data['porosity']['value']
        self.bruggeman = json_data['bruggeman']['value']

        self.density = json_data['density']['value']

        self.set_volume()
        self.set_weight()

    def set_weight(self):
        self.weight = self.density * self.volume

class Electrolyte:

    def __init__(self, json_data):

        self.name = json_data['name']

        self.density = json_data['density']['value']

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

    def __init__(self, json_data, SOC_ini, capacityRatio=None, capacity=None):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']

        self.porosity = json_data['porosity']['value']
        self.electrolyteBruggeman = json_data['electrolyteBruggeman']['value']
        self.electrodeBruggeman = json_data['electrodeBruggeman']['value']
        
        self.electronicConductivity = json_data['electronicConductivity']['value']
        self.thermalConductivity = json_data['thermalConductivity']['value']
        self.specificHeat = json_data['heatCapacity']['value']

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

        if capacityRatio == None:
            pass
        else:
            sum = 0
            for material in self.composition:
                if material.active == 1:
                    sum += material.theoreticalCapacity * material.weightFraction

            self.thickness = capacityRatio * capacity / (sum * self.density * (1 - self.porosity) * self.area )

        self.set_volume()
        self.set_weight()

        self.set_capacity()

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

    def __init__(self, json_data):

        self.name = json_data['name']

        self.thickness = json_data['thickness']['value']; self.area = json_data['area']['value']

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
