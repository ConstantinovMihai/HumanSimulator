import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.control.lti import TransferFunction
from systems_functions import *
import constants

ct = constants.SimulationConstants()

# Define symbolic variables
a0, a1, a2, b0, b1 = sp.symbols('a0 a1 a2 b0 b1')

# Define the transfer function coefficients
H = {
    'num': [b1, b0],         # numerator coefficients
    'den': [a2, a1, a0],     # denominator coefficients
    'd0': 0                  # direct feedthrough term
}

# Convert to state space
A, B, C, D = convert_tf_ss_ccf_sympy(H)

print("shapes are")
print(A.shape)
print(B.shape)
print(C.shape)
print(D.shape)

print("A: ", A)
print("B: ", B)
print("C: ", C)
print("D: ", D)


# Example functions for time-varying parameters
def a_t(t):
    return 2 + 0.1 * np.sin(t)

def b_t(t):
    return 1 + 0.05 * np.cos(t)

def c_t(t):
    return 3 + 0.2 * np.sin(t/2)

def d_t(t):
    return 2 + 0.1 * np.cos(t/2)

for t in range(10):
   
    num_values = {a0 : a_t(t), a1 : a_t(t), a2 : b_t(t), b0 : c_t(t), b1 : d_t(t)}
    print(num_values)
    An = A.subs(num_values).evalf()
    Bn = B.subs(num_values).evalf()
    Cn = C.subs(num_values).evalf()
    Dn = D.subs(num_values).evalf()

    print("matrices are")
    print(An)
    print(Bn)
    print(Cn)
    print(Dn)

    Ad, Bd, Cd, Dd = cont2discrete_zoh(ct.dt, An, Bn, Cn, Dn)

    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Now at time {t}:")
    print("A: ", A.evalf(subs={a0: a_t(t), a1: a_t(t), a2: a_t(t)}))
    print("B: ", B.evalf(subs={b0: b_t(t), b1: b_t(t)}))
    print("C: ", C.evalf(subs={a2: a_t(t), b0: c_t(t), b1: c_t(t)}))
    print("D: ", D.evalf(subs={b0: d_t(t)}))


# # Convert symbolic matrices to functions of parameters
# A_func = sp.lambdify((a, b, c, d), A, 'numpy')
# B_func = sp.lambdify((a, b, c, d), B, 'numpy')
# C_func = sp.lambdify((a, b, c, d), C, 'numpy')
# D_func = sp.lambdify((a, b, c, d), D, 'numpy')


# # Define the state-space equations
# def state_space_equations(t, x, u_func):
#     # Substitute current parameter values
#     a_val = a_t(t)
#     b_val = b_t(t)
#     c_val = c_t(t)
#     d_val = d_t(t)
    
#     # Evaluate the state space matrices with current parameters
#     A_val = A_func(a_val, b_val, c_val, d_val)
#     B_val = B_func(a_val, b_val, c_val, d_val)
#     u_val = u_func(t)
#     dxdt = A_val @ x + B_val @ u_val
#     return dxdt

# # Define the input function, for example, a step input
# def u_func(t):
#     return np.array([1.0])  # constant input

# # Initial conditions and time span
# x0 = np.zeros(A.shape[0])
# t_span = (0, 10)  # from t=0 to t=10 seconds

# # Solve the state-space equations
# solution = solve_ivp(state_space_equations, t_span, x0, args=(u_func,), t_eval=np.linspace(0, 10, 1000))

# # Calculate the output
# def output_equation(t, x):
#     a_val = a_t(t)
#     b_val = b_t(t)
#     c_val = c_t(t)
#     d_val = d_t(t)
#     C_val = C_func(a_val, b_val, c_val, d_val)
#     D_val = D_func(a_val, b_val, c_val, d_val)
#     return C_val @ x + D_val @ u_func(t)

# y = np.array([output_equation(t, sol) for sol, t in zip(solution.y.T, solution.t)])

# # Plot the output
# plt.plot(solution.t, y)
# plt.xlabel('Time (s)')
# plt.ylabel('Output')
# plt.title('System Response with Time-Varying Parameters')
# plt.grid(True)
# plt.show()
