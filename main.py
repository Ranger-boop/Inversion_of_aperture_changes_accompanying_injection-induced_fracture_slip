import numpy as np
import pandas as pd
from sko.GA import GA
from functions import *

# Import data to be utilized for inversion
# Replace ... with the excel data containing slip displacement, slip velocity,
# and aperture change * C (the product of aperture change and compressibility of the testing system)
dil_file = "..."
data_dil = pd.read_excel(dil_file)

# Adjust unreasonable slip displacement
for i in data_dil.index[1:]:
    if data_dil.loc[i, "fault slip"] - data_dil.loc[i - 1, "fault slip"] < 0:
        data_dil.loc[i, "fault slip"] = data_dil.loc[i - 1, "fault slip"]

# Adjust slip velocity to inverse correctly
data_dil.loc[data_dil[data_dil["slip rate"] < 0].index, "slip rate"] = 0
for i in data_dil.index:
    if data_dil.loc[i, "slip rate"] == 0:
        data_dil.loc[i, "slip rate"] = (
            data_dil.loc[i - 1, "slip rate"] + data_dil.loc[i + 1, "slip rate"]
        ) / 2

# Set parameters for inversion
# Permeability for computing initial hydraulic aperture, unit is m^2
perm = 1.3e-13

# Arguments needed to compute b_slip
b_0 = np.sqrt(12 * perm) * 1e3  # initial aperture, unit is mm
u0_ini = data_dil["fault slip"].values[0]  # initial slip displacement, unit is mm
u_end = data_dil["fault slip"].values[1:]  # slip displacement, unit is mm
dil_ang = 0  # dilation angle

# Arguments needed to compute dilation parameters
v = data_dil["slip rate"].values[1:]  # slip velocity

# Arguments needed to compute RMSE
# Normalized Delta b: (aperture-change * C) / max(aperture-change * C)
NDB_exp = data_dil["aperture change/K (aperture change*C)"].values[1:] / np.max(
    data_dil["aperture change/K (aperture change*C)"].values[1:]
)


def fun_obj(paras):
    """
    The objective function to be optimized with Genetic Algorithm.

    Parameters
    ----------
    paras :
        The set of constrained parameters to be computed.
    output :
        RMSE (root mean square error) of computed normalized aperture change
        and measured aperture change * C.
    """
    # Receive dilation factor and characteristic distance with their upper and
    # lower bounds constrained
    dil_fact, D_c = paras

    # Displacement-dependent aperture
    b_slip: np.ndarray = aperture_slip_disp(b_0, u_end, u0_ini, dil_ang)

    # Dilation parameters
    d_phi_2dim: list[np.ndarray] = [0] * len(b_slip)  # type: ignore
    for i in range(len(b_slip)):
        d_phi_2dim[i] = dil_para(dil_fact, u_end[: i + 1], v[: i + 1], D_c, dt_acq=0.01)

    # Modeled aperture change
    b_mod: np.ndarray = aperture_shear_dil(b_slip, d_phi_2dim)

    # Normalized aperture change
    if np.max(b_mod - b_0) == 0:
        rmse_obj = 1e5
    else:
        NDB_mod: np.ndarray = (b_mod - b_0) / np.max(b_mod - b_0)
        rmse_obj = rmse(NDB_mod, NDB_exp)

    return rmse_obj


ga = GA(
    func=fun_obj,
    n_dim=2,
    size_pop=200,
    max_iter=10000,
    # Lower bounds of dil_fact (dilation factor) and D_c (characteristic distance)
    lb=[0, 1e-4],
    # Upper bounds of dil_fact (dilation factor) and D_c (characteristic distance)
    ub=[5, 5],
    #     precision=1e-7,
)
best_x, best_y = ga.run()

print(f"Best fitted dilation factor: {best_x[0]},\n" + f"Best fitted D_c: {best_x[1]}")
print(f"Lowest value of objective function: {best_y[0]}")
