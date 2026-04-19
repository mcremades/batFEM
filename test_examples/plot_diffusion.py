import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def get_diffusion_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Target the active material component
    am = next(item for item in data['composition'] if item.get('active') == 1)
    diff_block = am['diffusionConstant']
    arr_block = diff_block['arrhenius']
    
    return {
        "func_str": diff_block['value'],
        "Ea": arr_block['activationEnergy']['value'],
        "Tref": arr_block['referenceTemperature']['value'],
        "x0": am['stoichiometry0']['value'],
        "x1": am['stoichiometry1']['value']
    }

def plot_diffusion_dynamics(json_path):
    d_data = get_diffusion_data(json_path)
    R = 8.314  # Gas constant
    
    # 1. Setup Ranges
    soc = np.linspace(0, 1, 100)
    temps_c = np.array([-10, 0, 10, 25, 45]) # Degrees Celsius for 2D plot
    
    # Map SoC to x
    #x = d_data['x0'] + soc * (d_data['x1'] - d_data['x0'])
    x=soc
    # 2. Define Diffusion calculation
    def calc_diffusion(x_val, T_kelvin):
        # Base diffusion at T_ref
        D_ref = eval(d_data['func_str'], {"np": np, "exp": np.exp, "x": x_val})
        # Arrhenius multiplier
        arrhenius_corr = np.exp((d_data['Ea'] / R) * (1/d_data['Tref'] - 1/T_kelvin))
        return D_ref * arrhenius_corr

    # 3. Create Plots
    plt.figure()
    

    for T_c in temps_c:
        D = calc_diffusion(x, T_c + 273.15)
        plt.semilogy(soc, D, label=f'T = {T_c+273.15}°K', linewidth=2)
    
    plt.title("Ecker 2015. Positive Electrode")
    plt.xlabel("Degree-of-Lithiation [-]")
    plt.ylabel("$D_s$ [$m^2/s$] (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    plot_diffusion_dynamics('json_battery/electrodes/PE_Ecker2015.json')