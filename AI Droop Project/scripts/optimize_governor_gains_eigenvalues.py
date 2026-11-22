"""Optimize governor gains using local linearized eigenvalue penalties.

Research implementation based on the uploaded original script.

Important: all numerical equations, initial conditions, control values, and
contingency settings are intentionally retained from the original implementation.
See the repository README for modeling limitations and validation notes.
"""

import numpy as np
import pandapower.networks as pn
import pandapower as pp
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import minimize


# ============================================================================
# 01  IEEE 9-BUS NETWORK
# ============================================================================
# Load the IEEE 9-bus system
net = pn.case9()
# remove existing slack and generators
net.ext_grid = net.ext_grid.drop(net.ext_grid.index)
net.gen = net.gen.drop(net.gen.index)
# add ext_grids as slack buses
new_vm_va_values = {0: (1.040, 0), 1: (1.025, 9.3), 2: (1.025, 4.7)}
for bus in new_vm_va_values:
    pp.create_ext_grid(net, bus=bus,
                       vm_pu=new_vm_va_values[bus][0],
                       va_degree=new_vm_va_values[bus][1])


# ============================================================================
# 02  BASE VALUES AND DYNAMIC PARAMETERS
# ============================================================================
# System base values
V_base = 345e3
S_base = 100e6
Z_base = V_base**2 / S_base
omega_s = 2 * np.pi * 60

# Machine & control parameters (example values)
H = [23.6, 6.4, 3.01]
D = [0.01, 0.01, 0.01]  # damping
# Transient reactances and time constants
Xd = [1.81 * Z_base, 1.76 * Z_base, 1.76 * Z_base]
Xq = [1.76 * Z_base, 1.71 * Z_base, 1.71 * Z_base]
Xd_prime = [0.3 * Z_base, 0.33 * Z_base, 0.33 * Z_base]
Xq_prime = [0.55 * Z_base, 0.62 * Z_base, 0.62 * Z_base]
Tdo_prime = [8.0, 6.0, 6.0]
Tqo_prime = [0.4, 0.5, 0.5]
# Governor / turbine
Tt = [7.0, 7.0, 7.0]
Tv = [0.05, 0.05, 0.05]
Pm_ref = [71.6, 163.0, 85.0]


# Function to compute eigenvalues of 7-state model for each generator
# State vector x = [δ, ω, Eq′, Ed′, Pm, v]ᵀ


# ============================================================================
# 03  LOCAL STATE-SPACE EIGENVALUES
# ============================================================================
def compute_eigenvalues_4th(Kg):
    eigenvalues = []
    for i in range(3):
        A = np.array([
        [0,                           1,                          0,                        0,                   0,                 0],
        [0,              -D[i]/(2*H[i]),                          0,                        0,          1/(2*H[i]),                 0],
        [0,                           0,            -1/Tdo_prime[i],                        0,                   0,                 0],
        [0,                           0,                          0,           -1/Tqo_prime[i],                   0,                 0],
        [0,                           0,                          0,                        0,          -1/Tt[i],        1/Tt[i]],
        [0,             -Kg[i]/Tv[i],                          0,                        0,                   0,        -1/Tv[i]]
        ])

        # Note: coupling via Id, Iq, Pe_int neglected in local model
        eig_vals, _ = eig(A)
        eigenvalues.append(eig_vals)
    return np.array(eigenvalues)

# Optimization penalty based on damping ratio

# ============================================================================
# 04  STABILITY AND DAMPING PENALTY
# ============================================================================
iteration_history = []
penalty_history = []
# def initial_guess_for_kg_4th(kg):
#     eig_vals = compute_eigenvalues_4th(kg)
#     damping_ratios = -np.real(eig_vals) / np.abs(eig_vals)
#     penalty = 0
#     threshold = 0.01
#     for i in range(3):
#         # penalty for any unstable mode
#         if np.any(np.real(eig_vals[i]) > -1e-3):
#             penalty += 1e6
#         # penalty for low damping
#         for dr in damping_ratios[i]:
#             if dr < threshold:
#                 penalty += (threshold - dr)**2 * 1e3
#     iteration_history.append(kg.copy())
#     penalty_history.append(penalty)
#     print("Panelty: ", penalty)
#     return penalty



def initial_guess_for_kg_4th(kg):
    eig_vals = compute_eigenvalues_4th(kg)
    penalty = 0
    threshold = 0.01  # Minimum damping ratio
    epsilon = 1e-12   # Small value to avoid division by zero
    
    for i in range(3):
        # Handle eigenvalues
        for ev in eig_vals[i]:
            mag = np.abs(ev)
            # Skip invalid eigenvalues (zero magnitude)
            if mag < 1e-6:
                penalty += 1e6  # Penalize invalid eigenvalues
                continue
            # Check stability (positive real part)
            if np.real(ev) > 0:
                penalty += 1e6  # High penalty for unstable modes
            # Compute damping ratio safely
            damping_ratio = -np.real(ev) / (mag + epsilon)
            # Apply penalty for low damping
            if damping_ratio < threshold:
                penalty += (threshold - damping_ratio)**2 * 1e3
    
    iteration_history.append(kg.copy())
    penalty_history.append(penalty)
    print("Penalty: ", penalty)
    return penalty


# Initial Kg guess
# initial_Kg = [22.99277813, 17.11845881 , 25.10358251]

# ============================================================================
# 05  OPTIMIZATION EXECUTION
# ============================================================================
initial_Kg = [90, 100, 110]

# Run optimization
result = minimize(initial_guess_for_kg_4th, initial_Kg,
                  bounds=[(1, 250)]*3, method='Nelder-Mead',
                  options={'maxiter': 500})
opt_Kg = result.x
print("Optimized initial Kg for 4th order model:", opt_Kg)

# Plot progress
it = np.array(iteration_history)
plt.figure()
plt.plot(it[:,0], label='Kg1')
plt.plot(it[:,1], label='Kg2')
plt.plot(it[:,2], label='Kg3')
plt.xlabel('Iteration')
plt.ylabel('Kg')
plt.legend()
plt.title('Kg Optimization Progress (4th Order)')
plt.grid(True)
plt.show()
