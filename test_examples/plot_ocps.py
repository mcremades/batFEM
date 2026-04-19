import json
import numpy as np
import matplotlib.pyplot as plt

def load_electrode_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract the active material dictionary (first element in composition)
    active_material = next(item for item in data['composition'] if item.get('active') == 1)
    
    return {
        "name": data['name'],
        "ocp_func": active_material['openCircuitPotential']['value'],
        "x0": active_material['stoichiometry0']['value'],
        "x1": active_material['stoichiometry1']['value']
    }

def generate_ocv_plot(neg_json, pos_json):
    # 1. Load data from your specific JSON structures
    neg = load_electrode_data(neg_json)
    pos = load_electrode_data(pos_json)
    
    # 2. Setup SoC range
    soc = np.linspace(0, 1, 500)
    
    # 3. Map SoC to lithiation (x) for each electrode
    # Anode: x increases with SoC
    x_neg = neg['x0'] + soc * (neg['x1'] - neg['x0'])
    # Cathode: x decreases with SoC
    x_pos = pos['x0'] - soc * (pos['x0'] - pos['x1'])
    
    # 4. Evaluate OCP functions
    # Using a dictionary to provide 'np' and 'exp/tanh' for the eval() call
    eval_env = {"np": np, "exp": np.exp, "tanh": np.tanh}
    
    u_neg = eval(neg['ocp_func'], eval_env, {"x": x_neg})
    u_pos = eval(pos['ocp_func'], eval_env, {"x": x_pos})
    ocv = u_pos - u_neg
    
    # 5. Plotting
    plt.figure()
    
    # Primary axis for potentials
    plt.plot(soc, u_neg, color='blue', label=f'$U^-$', linewidth=1.5)
    plt.plot(soc, u_pos, color='red', label=f'$U^+$', linewidth=1.5)
    plt.plot(soc, ocv, color='green', label='$OCV$', linewidth=2.5)
    
    # Aesthetic adjustments for PhD-level plots
    plt.title("Marquis2019", fontsize=14)
    plt.xlabel("State-of-Charge[-]", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, loc='best')
    
    # Adding stoichiometry markers for clarity
    #plt.annotate(f'$x^-_{{0\%}}$={neg["x0"]}', (0, u_neg[0]), textcoords="offset points", xytext=(+20,10), ha='center', color='blue')
    #plt.annotate(f'$x^-_{{100\%}}$={neg["x1"]}', (1, u_neg[-1]), textcoords="offset points", xytext=(-20,10), ha='center', color='blue')
    #plt.annotate(f'$x^+_{{0\%}}$={pos["x0"]}', (0, u_pos[0]), textcoords="offset points", xytext=(+20,10), ha='center', color='red')
    #plt.annotate(f'$x^+_{{100\%}}$={pos["x1"]}', (1, u_pos[-1]), textcoords="offset points", xytext=(-20,-30), ha='center', color='red')
    
    plt.show()

# Run the generator
if __name__ == "__main__":
    # Update these paths to your actual file locations
    generate_ocv_plot('json_battery/electrodes/NE_Marquis2019.json', 'json_battery/electrodes/PE_Marquis2019.json')