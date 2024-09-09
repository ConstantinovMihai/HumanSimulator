import scipy.signal as sg
import numpy as np
import control as ct
from simulate_f_star import convert_tf_ss_ccf
from simulate_f_star import cont2discrete_zoh

H = {'den': np.array([1, 0, 2]), 'num': np.array([1, 3, 2, 10]), "d0" : np.array([0])}

Gs = ct.tf2ss([1,0,1],[1,2,10])
Gc, T = ct.canonical_form(Gs,'reachable')

A, B, C, D = convert_tf_ss_ccf(H)
# check the discretization
dt = 0.01
print(sg.cont2discrete((A, B, C, D), dt))
(Ad, Bd, Cd, Dd, _) = sg.cont2discrete((A, B, C, D), dt)
Ad_own, Bd_own, Cd_own, Dd_own = cont2discrete_zoh(dt, A, B, C, D)

# check if the values are close:
print("-----------")
# Set tolerance for comparison
tolerance = 1e-6

# Check if values are not close and raise ValueError if they are not
if not (np.allclose(Ad, Ad_own, atol=tolerance) and
        np.allclose(Bd, Bd_own, atol=tolerance) and
        np.allclose(Cd, Cd_own, atol=tolerance) and
        np.allclose(Dd, Dd_own, atol=tolerance)):
    raise ValueError(f"The calculated matrices are not close to the expected ones:\n"
                     f"Ad: {np.isclose(Ad, Ad_own)}\n"
                     f"Bd: {np.isclose(Bd, Bd_own)}\n"
                     f"Cd: {np.isclose(Cd, Cd_own)}\n"
                     f"Dd: {np.isclose(Dd, Dd_own)}")


print(f"{Ad=},\n {Bd=},\n {Cd=},\n {Dd=}")
print("-----------")
print(f"{Ad_own=},\n {Bd_own=},\n {Cd_own=},\n {Dd_own=}")