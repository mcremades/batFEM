from dolfin import *; import numpy; import matplotlib.pyplot as plt#; from dolfin_adjoint import *

import batFEM.class_battery_model; import batFEM.ME_PE.class_ME_PE; import fatDAE.dolfin_interface.class_problem

class ME_PE_SPM(batFEM.ME_PE.class_ME_PE.ME_PE):
    pass

class RK_ME_PE_SPM(ME_PE_SPM, fatDAE.dolfin_interface.class_problem.UFL_Problem):
    pass

class ME_PE_SPME(ME_PE_SPM):
    pass

class RK_ME_PE_SPME(ME_PE_SPME, fatDAE.dolfin_interface.class_problem.UFL_Problem):
    pass
