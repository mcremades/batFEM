import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Optional imports (remove if unused)
# import xml.etree.ElementTree as ET
# import meshio
# import json
# import batFEM.class_battery

# ==========================================================
# CONFIGURATION
# ==========================================================

battery_name = "Ecker2015"
base_path = Path('test_examples/gradients_adjoint') / battery_name

# ==========================================================
# LOAD DATA
# ==========================================================

param_file = base_path / 'par.txt'
grad_sdirk_file_fd = base_path / 'fd_grad_gene_SDIRK3CP.txt'
grad_sdirk_file = base_path / 'grad_gene_SDIRK3CP.txt'
grad_row3p_file = base_path / 'grad_gene_ROW3P.txt'
grad_row4pw2_file = base_path / 'grad_gene_ROW4PW2.txt'


params = np.loadtxt(param_file)
grad_SDIRK3CP_fd = np.loadtxt(grad_sdirk_file_fd)
grad_SDIRK3CP = np.loadtxt(grad_sdirk_file)
grad_ROW3P = np.loadtxt(grad_row3p_file)
grad_ROW4PW2 = np.loadtxt(grad_row4pw2_file)

# ==========================================================
# CHECK SHAPES (important!)
# ==========================================================

print("Shapes:")
print("SDIRK3CP:", grad_SDIRK3CP.shape)
print("ROW3P:", grad_ROW3P.shape)
print("ROW4PW2:", grad_ROW4PW2.shape)

# ==========================================================
# PLOT
# ==========================================================

plt.figure()

params_list=['$L^-$','$\\varepsilon_s^-$','$\\varepsilon_i^-$','$R_p^-$','$L^+$','$\\varepsilon_s^+$','$\\varepsilon_i^+$','$R_p^+$']

plt.scatter(params_list, params*grad_SDIRK3CP_fd, color='r',marker='o', label='Finite differences')
plt.scatter(params_list, params*grad_SDIRK3CP, color='r',marker='x', label='SDIRK3CP')
plt.scatter(params_list, params*grad_ROW3P, color='g',marker='x', label='ROW3P')
plt.scatter(params_list, params*grad_ROW4PW2, color='b',marker='x', label='ROW4PW2')

plt.ylabel("$\\mathbf{u}* \\nabla \\Psi(\\mathbf{u})$")
plt.title(battery_name+' Gravimetric gradient (scaled)')
plt.legend()
plt.grid(True)

plt.figure()

params_list=['$L^-$','$\\varepsilon_s^-$','$\\varepsilon_i^-$','$R_p^-$','$L^+$','$\\varepsilon_s^+$','$\\varepsilon_i^+$','$R_p^+$']

plt.scatter(params_list, grad_SDIRK3CP_fd, color='r',marker='o', label='Finite differences')
plt.scatter(params_list, grad_SDIRK3CP, color='r',marker='x', label='SDIRK3CP')
plt.scatter(params_list, grad_ROW3P, color='g',marker='x', label='ROW3P')
plt.scatter(params_list, grad_ROW4PW2, color='b',marker='x', label='ROW4PW2')

plt.ylabel("$\\nabla \\Psi(\\mathbf{u})$")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title(battery_name+' Gravimetric gradient')
plt.legend()
plt.grid(True)


plt.show()



