import json
import numpy as np
import matplotlib.pyplot as plt

def solve_electrolyte_string(func_string, c_e, T):
    """
    Evaluates the JSON string function using numpy.
    Replaces 'c_e' and 'T' with the provided arrays.
    """
    # Create an evaluation environment with numpy and the variables
    eval_env = {
        "np": np, 
        "exp": np.exp, 
        "c_e": c_e, 
        "T": T
    }
    # Clean up common potential syntax issues from JSON (like ** to np.power if needed)
    # However, eval handles ** natively in Python
    return eval(func_string, eval_env)

def plot_electrolyte_from_json(json_path):
    # 1. Load the JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. Extract Data Blocks
    diff_block = data['diffusionConstant']
    cond_block = data['ionicConductivity']
    arr_cond = cond_block.get('arrhenius', None)
    
    # 3. Setup Ranges
    c_e_range = np.linspace(100, 3000, 500) # mol/m3
    temps_c = [-10, 0, 10, 25, 45]
    R = 8.314

    plt.figure()

    for T_c in temps_c:
        T_k = T_c + 273.15
        
        # --- Calculate Ionic Conductivity ---
        # Evaluate the base function
        kappa_base = solve_electrolyte_string(cond_block['value'], c_e_range, T_k)
        
        # Apply Arrhenius if present
        if arr_cond:
            Ea = arr_cond['activationEnergy']['value']
            Tref = arr_cond['referenceTemperature']['value']
            arr_factor = np.exp((Ea/R) * (1/Tref - 1/T_k))
            kappa = kappa_base * arr_factor
        else:
            kappa = kappa_base

        # 4. Plotting
        plt.plot(c_e_range, kappa, label=f'T = {T_c+273.15}°K', linewidth=2)

    plt.title(f"Ecker2015", fontsize=13)
    plt.xlabel("Concentration $c_e$ [mol/m$^3$]")
    plt.ylabel("$\kappa_e$ [S/m]")
    plt.grid(True, alpha=0.3)
    plt.legend()


    plt.figure()
    for T_c in temps_c:
        T_k = T_c + 273.15
        
        # --- Calculate Ionic Conductivity ---
        # Evaluate the base function
        kappa_base = solve_electrolyte_string(cond_block['value'], c_e_range, T_k)

        # --- Calculate Diffusion Constant ---
        # Note: Your diffusion JSON string already contains 'T' and 'c_e' 
        # so it handles its own temperature dependency internally.
        de = solve_electrolyte_string(diff_block['value'], c_e_range, T_k)

        # 4. Plotting
        plt.semilogy(c_e_range, de, label=f'T = {T_c+273.15}°K', linewidth=2)

    # Aesthetics
    
    plt.title(f"Ecker2015", fontsize=13)
    plt.xlabel("Concentration $c_e$ [mol/m$^3$]")
    plt.ylabel("$D_e$ (m$^2$/s) (Log Scale)")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    plt.show()

# Run with your file
if __name__ == "__main__":
    # Assuming your file is named 'electrolyte.json'
    plot_electrolyte_from_json('json_battery/electrolytes/LiPF6_EC:DMC_Ecker2015.json')

    path_Chen='json_battery/electrolytes/LiPF6_EC:EMC_Chen2020.json'
    path_Marquis='json_battery/electrolytes/LiPF6_EC:DMC_Marquis2019.json'
    path_Prada='json_battery/electrolytes/LiPF6_EC:EMC_Prada2013.json'
    with open(path_Chen, 'r') as f:
        data_Chen = json.load(f)
    with open(path_Marquis, 'r') as f:
        data_Marquis = json.load(f)
    with open(path_Prada, 'r') as f:
        data_Prada = json.load(f)

    x=np.linspace(0,3000,100)

    eval_env = {
        "np": np, 
        "exp": np.exp, 
        "c_e": x, 
        "T": 298.15
    }

    D_e_Chen=lambda c_e: eval(data_Chen['diffusionConstant']['value'],eval_env)
    D_e_Marquis=lambda c_e: eval(data_Marquis['diffusionConstant']['value'],eval_env)
    D_e_Prada=lambda c_e: data_Prada['diffusionConstant']['value']*(c_e/c_e)

    


    plt.figure()
    plt.semilogy(x,D_e_Chen(x),label='Chen2020', linewidth=2)
    plt.semilogy(x,D_e_Marquis(x),label='Marquis2019', linewidth=2)
    plt.semilogy(x,D_e_Prada(x),label='Prada2013', linewidth=2)
    plt.xlabel("Concentration $c_e$ [mol/m$^3$]")
    plt.ylabel("$D_e$ [m$^2$/s] (Log Scale)")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
   
    kappa_e_Chen=lambda c_e: eval(data_Chen['ionicConductivity']['value'],eval_env)
    kappa_e_Marquis=lambda c_e: eval(data_Marquis['ionicConductivity']['value'],eval_env)
    kappa_e_Prada=lambda c_e: eval(data_Prada['ionicConductivity']['value'],eval_env)

    plt.figure()
    plt.plot(x,kappa_e_Chen(x),label='Chen2020', linewidth=2)
    plt.plot(x,kappa_e_Marquis(x),label='Marquis2019', linewidth=2)
    plt.plot(x,kappa_e_Prada(x),label='Prada2013', linewidth=2)
    plt.xlabel("Concentration $c_e$ [mol/m$^3$]")
    plt.ylabel("$\kappa_e$ [S/m]")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
     
    plt.show()