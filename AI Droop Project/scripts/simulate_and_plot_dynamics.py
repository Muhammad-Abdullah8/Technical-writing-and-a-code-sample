"""Simulate and plot synchronous-generator dynamics under a load change.

Research implementation based on the uploaded original script.

Important: all numerical equations, initial conditions, control values, and
contingency settings are intentionally retained from the original implementation.
See the repository README for modeling limitations and validation notes.
"""

import numpy as np 
import pandapower.networks as pn
import pandapower as pp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# -------------------------------

# ============================================================================
# 01  IEEE 9-BUS NETWORK AND INITIAL POWER FLOW
# ============================================================================
# Build the IEEE 9-bus system
# -------------------------------
net = pn.case9()
net.ext_grid = net.ext_grid.drop(net.ext_grid.index)
net.gen = net.gen.drop(net.gen.index)

load_flow_angle = [0, 9.3, 4.7]

# Define ext_grids (slack buses)
new_vm_va_values = {0: (1.040, load_flow_angle[0]),
                    1: (1.025, load_flow_angle[1]),
                    2: (1.025, load_flow_angle[2])}
for bus in new_vm_va_values:
    pp.create_ext_grid(net, bus=bus, vm_pu=new_vm_va_values[bus][0], va_degree=new_vm_va_values[bus][1])

initial_angles_rad = [np.radians(0), np.radians(9.3), np.radians(4.7)]
# print("Columns in net.ext_grid:\n", net.ext_grid)
  
pp.runpp(net, algorithm='nr')
print(net.line)
# Retrieve power outputs from network solution
Pe_1 = {net.ext_grid.at[idx, "bus"]: net.res_ext_grid.at[idx, "p_mw"] for idx in net.ext_grid.index}
Qe_1 = {net.ext_grid.at[idx, "bus"]: net.res_ext_grid.at[idx, "q_mvar"] for idx in net.ext_grid.index}
Pm_ref, Qm_ref = [], []
for i in range(3):
    Pe_initial, Qe_initial = Pe_1[i], Qe_1[i]
    Pm_ref.append(Pe_initial)
    Qm_ref.append(Qe_initial)
    
# print("Pe:", Pe_1, "\nQe:", Qe_1)

# -------------------------------

# ============================================================================
# 02  MACHINE AND CONTROL PARAMETERS
# ============================================================================
# Define System Parameters
# -------------------------------
V_base = 345e3
S_base = 100e6
Z_base = (V_base ** 2) / S_base

H = [23.64, 6.4, 3.01]         # Inertia constants
Tv = [0.05, 0.05, 0.05]         # Governor time constants
Tt = [7.0, 7.0, 7.0]            # Turbine time constants
omega_s = 2 * np.pi * 60        # Synchronous speed
Vout = [345, 345, 345]          # Terminal voltage 

# Kg = [(20 * (S_base/1e6)) / omega_s for _ in range(3)]
# Kg = [22.99277813, 17.11845881 , 25.10358251]

Kg = [5.3, 5.4, 5.3]




# Synchronous machine parameters (d-q parameters)
Xa = [0.0146, 0.08958, 0.13125]  # stator leakage reactance
Xd = [0.1460, 0.89580, 1.3125] 
Xd_prime = [0.060800, 0.119800, 0.181300]
Xd_double_prime = [0.06000, 0.11000, 0.18000]
Xq = [0.1000, 0.864500, 1.257800]
Xq_prime = [0.096900, 0.196900, 0.25000]
Xq_double_prime = [0.06000, 0.11000, 0.18000]
Ra = [0.000125, 0.000125, 0.000125]  # stator resistance 
Tdo_prime = [8.9600, 6.000, 5.89000]  # transient open time constant
Tdo_double_prime = [0.01000, 0.01000, 0.01000]
Tqo_prime = [0.3100, 0.53500, 0.6000]
Tqo_double_prime = [0.0100, 0.0100, 0.01000]
Ef = [1.082, 1.791, 1.406]

# PI Controller Parameters
Kp = [1, 1, 1]  # Proportional gains
Ki = [0.01, 0.01, 0.01]  # Integral gains
V_ref = [1.040, 1.025, 1.025]

Xq_prime_mult_base = [factor * Z_base for factor in Xq_prime]
Xq_double_prime_mult_base = [factor * Z_base for factor in Xq_double_prime]
Xq_mult_base = [factor * Z_base for factor in Xq]

D = []
for i in range(len(Xq)):
    D_i = (((Xq_prime_mult_base[i] - Xq_double_prime_mult_base[i]) / (Xq_mult_base[i] + Xq_prime_mult_base[i])) *
           (Xq_prime_mult_base[i] / Xq_double_prime_mult_base[i])) * (Tqo_double_prime[i] * Vout[i])
    D.append(D_i)
print("Damping values:", D)

# -------------------------------

# ============================================================================
# 03  SIMULATION TRACE COLLECTION
# ============================================================================
# Global Storage for Dynamic Quantities
# -------------------------------
time_list = []      # To store time stamps
Id_all = []         # List to store [Id for gen1, gen2, gen3] at each time step
Iq_all = []         # Similar for Iq
v_angles_all = []   # Voltage angles (in degrees)
Vt_all = []         # Terminal voltages
Pe_internal_all = []
# -------------------------------
# Functions for current and constraints
# -------------------------------

#In the following method, pi and qi should be subtracted but it works usulaly when we put + 
#If we chage - to + (for pi and qi) rember we need to cange Ed and Eq initial guesses that we are calcualting using 
# a seprate code. 


# ============================================================================
# 04  STATOR CURRENT CONSTRAINTS
# ============================================================================
def current_constr(Pi, Qi, Ed, Eq, Id, Iq, Xd_prime, Xq_prime, Ra):
    Vd = Ed - (Ra * Id - Xq_prime * Iq)*0
    Vq = Eq + (Xd_prime * Id - Ra * Iq)*0
    #print(Vd,Vq)
    pr = Pi/100 + (Vd * Id + Vq * Iq)
    qr = Qi/100 + (Vq * Id - Vd * Iq)
    return np.array([pr, qr])

def calc_current(p_g, q_g, ed, eq, xd_prime, xq_prime, ra):
    res = fsolve(lambda x: current_constr(p_g, q_g, ed, eq, x[0], x[1], xd_prime, xq_prime, ra),
                 x0=np.array([0, 0]))
    return res[0], res[1]

#for Pi and qi "-" 
# Eq_prime_inital_guess = [-0.04178, -0.02360, -0.06467]
# Ed_prime_inital_guess = [1.01486, 1.04970, 1.06506]

#for Pi and qi "+" 
Eq_prime_inital_guess = [0.04178, 0.35489, 0.23264]
Ed_prime_inital_guess = [1.06514, 0.97336, 0.97805]

# -------------------------------
# Differential Equations: swing governor dynamics
# -------------------------------

# ============================================================================
# 05  COUPLED ELECTROMECHANICAL DYNAMICS
# ============================================================================
def swing_governor(t, y):
    global Kg
    num_generators = 3  
    dydt = np.zeros(len(y))
    
        
    # Apply load increase at t=8 but for now let's not apply any contigency
    if t >=10:
        net.load.loc[net.load['bus'] == 6, 'p_mw'] = 200

    # if t>10: 
    # #     # Kg = [132.97490019,  62.48237466, 250]
    # #     Kg = [56.58778173, 31.6691488,  93.02271963]
    # #     # Kg = [103.33333333, 103.33333333 , 95.0]
    # # # Update network solution (if desired)

    #     # Kg =  [132.97490019 , 62.48237466 ,250.        ]
    #     #for max omega dev less than 2 
    #     # Kg = [54.03302861, 29.10137998, 92.88325529]
    #      #for max omega dev less than 1.5
    #     # Kg=[189.29833333 , 87.53833333, 100.965     ]
    #     # Kg= [118.33333333,  106.66666667, 113.33333333]

    #     Kg = [59.83, 28.76,  32.9]

    Id_list, Iq_list = [], []
    v_angles = []
    Vt_list = []

    for i in range(3):
        base = i * 7
        Eq_prime = y[base+2]
        Ed_prime = y[base+3]
        Vd = Ed_prime
        Vq = Eq_prime
        voltage_magnitude = np.sqrt(Vd**2 + Vq**2)
        voltage_angle = np.arctan2(Vq, Vd) + (y[base]-y[0])*1

        #print("  Vm: {:.4f} pu, angle: {:.4f} deg".format(voltage_magnitude, np.degrees(voltage_angle)))
        v_angles.append(np.degrees(voltage_angle))
        Vt_list.append(voltage_magnitude)

        
        # Update network voltages on the busses (0, 1, 2) /generator busses 
        net.ext_grid.at[i, "vm_pu"] = Vt_list[i] 
        net.ext_grid.at[i, "va_degree"] = v_angles[i]

    pp.runpp(net, algorithm='nr',numba = False)
    

    
    for i in range(num_generators):
        base = i * 7
        Eq_prime = y[base+2]
        Ed_prime = y[base+3]

        Pe = net.res_ext_grid.at[i, "p_mw"]
        Qe = net.res_ext_grid.at[i, "q_mvar"]

        # Debug print for current values
        #print("Time {:.4f} -- Gen {}: Pe={}, Qe={}".format(t, i, Pe, Qe))
        
        # Use previous solution for current calculation
        Id_i, Iq_i = calc_current(Pe, Qe, Ed_prime, Eq_prime, Xd_prime[i], Xq_prime[i], Ra[i])
        Id_list.append(Id_i)
        Iq_list.append(Iq_i)
    
        
        #print('='*10)
        #print(Vd,Vq)

        
        
        
        # Update network voltages (initally we were updating the voltages here now we are doing it after solving diffenrential equations)
        # net.ext_grid.at[i, "vm_pu"] = voltage_magnitude
        # net.ext_grid.at[i, "va_degree"] = np.degrees(voltage_angle)
        
    
    # Store the instantaneous values for post-simulation plotting.
    time_list.append(t)
    Id_all.append(Id_list.copy())
    Iq_all.append(Iq_list.copy())
    v_angles_all.append(v_angles.copy())
    Vt_all.append(Vt_list.copy())
    
    pe_interal_data = []
    # Differential equations with PI control for each generator
    for i in range(num_generators):
        base = i * 7
        delta, omega, Eq_prime, Ed_prime, Pm, v, int_V_err = y[base:base+7]
        Id = Id_list[i]
        Iq = Iq_list[i]
        Vt = Vt_list[i]
        
        error = V_ref[i] - Vt
        dydt[base+6] = error  # Update the integral term
        
        print("Time {:.4f} -- Gen {}: Vt={:.4f} pu, Error={:.6f}, Int_error={:.6f}".format(t, i, Vt, error,int_V_err ))
        # print("Int_error: ", int_V_err)
        # PI-controlled field voltage calculation
        Ef_i = 10*Kp[i]*error + Ki[i]*int_V_err*100 + Ef[i]
        
        # Swing equation and voltage regulation dynamics
        Pe_internal = (Eq_prime * Iq) + (Ed_prime * Id) + ((Xd_prime[i] - Xq_prime[i]) * Id * Iq)
        
        dydt[base]   = omega - omega_s
        dydt[base+1] = (Pm + Pe_internal*100 - D[i]*(omega - omega_s)) / (2 * H[i])
        #print(Pm,Pe_internal)

        #dydt[base+2] = Ef_i - Eq_prime#(Ef_i - Eq_prime + 0.0*Id*(Xd[i] - Xd_prime[i])) / Tdo_prime[i]
        #dydt[base+3] =  - Ed_prime#(-Ed_prime - 0.0*Iq*(Xq[i] - Xq_prime[i])) / Tqo_prime[i]
        
        dydt[base+2] = (Ef_i - Eq_prime +  1*Id*(Xd[i] - Xd_prime[i])) / Tdo_prime[i]#(Ef_i - Eq_prime + 0.0*Id*(Xd[i] - Xd_prime[i])) / Tdo_prime[i]
        dydt[base+3] = (     - Ed_prime -  1*Iq*(Xq[i] - Xq_prime[i]))# / Tqo_prime[i]#(-Ed_prime - 0.0*Iq*(Xq[i] - Xq_prime[i])) / Tqo_prime[i]


        #dydt[base+2] = (Ef_i - Eq_prime + 0.0*Id*(Xd[i] - Xd_prime[i])) / Tdo_prime[i]
        #dydt[base+3] =  (-Ed_prime - 0.0*Iq*(Xq[i] - Xq_prime[i])) / Tqo_prime[i]

        
        dydt[base+4] = (-Pm + v) / Tt[i]
        dydt[base+5] = (-v - Kg[i]*(omega - omega_s) + Pm_ref[i]) / Tv[i]

        pe_interal_data.append(Pe_internal)

    Pe_internal_all.append(pe_interal_data.copy())
    return dydt


# -------------------------------
# Initial State Vector
# (Each generator has 7 states: [delta, omega, Eq_prime, Ed_prime, Pm, v, integral_error])
# -------------------------------
# y0 = [
#     0, omega_s, Eq_prime_inital_guess[0], Ed_prime_inital_guess[0], Pm_ref[0], Pm_ref[0], 0,
#     0, omega_s, Eq_prime_inital_guess[1], Ed_prime_inital_guess[1], Pm_ref[1], Pm_ref[1], 0,
#     0, omega_s, Eq_prime_inital_guess[2], Ed_prime_inital_guess[2], Pm_ref[2], Pm_ref[2], 0
# ]

#inital state vector with the initial angles which are zero in 0 in the above case

# ============================================================================
# 06  INITIAL CONDITIONS AND NUMERICAL INTEGRATION
# ============================================================================
y0 = [
    initial_angles_rad[0], omega_s, Eq_prime_inital_guess[0]*1, Ed_prime_inital_guess[0]*1, Pm_ref[0], Pm_ref[0], 0,
    initial_angles_rad[1], omega_s, Eq_prime_inital_guess[1]*1, Ed_prime_inital_guess[1]*1, Pm_ref[1], Pm_ref[1], 0,
    initial_angles_rad[2], omega_s, Eq_prime_inital_guess[2]*1, Ed_prime_inital_guess[2]*1, Pm_ref[2], Pm_ref[2], 0
]

t_span = (0, 120)
t_eval = np.linspace(t_span[0], t_span[1], 200)

# -------------------------------
# Run the Simulation using BDF (stiff solver)
# -------------------------------
solution = solve_ivp(swing_governor, t_span, y0, method='RK23', t_eval=t_eval)

# -------------------------------

# ============================================================================
# 07  STATE EXTRACTION AND PERFORMANCE METRICS
# ============================================================================
# Extract State Variables from Solution
# -------------------------------
t = solution.t
delta = solution.y[0::7]
omega = solution.y[1::7]
eq_prime = solution.y[2::7]
ed_prime = solution.y[3::7]
Pm_sim = solution.y[4::7]
v_sim = solution.y[5::7]

omega_dev = np.abs(solution.y[1::7] - omega_s)
max_omega_dev = np.max(omega_dev)
print("Max_omega_dev: ", max_omega_dev)



# -------------------------------

# ============================================================================
# 08  GENERATOR STATE PLOTS
# ============================================================================
# Plot Set 01: Mechanical Power, Ed_prime, Rotor Speed Deviation, and Eq_prime
# -------------------------------
# Mechanical Power (Reference):
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(t, Pm_sim[i], label=f'Gen {i+1}')
plt.title('Mechanical Power (Reference)')
plt.xlabel('Time [s]')
plt.ylabel('Power')
plt.legend()
plt.tight_layout()


# Transient Voltage Ed_prime:
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(t, ed_prime[i], label=f'Gen {i+1}')
plt.title('Transient Voltage Ed_Prime')
plt.xlabel('Time [s]')
plt.ylabel('Ed_prime')
plt.legend()
plt.tight_layout()


# Rotor Speed Deviation:
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(t, omega[i] - omega_s, label=f'Gen {i+1}')
plt.title('Rotor Speed Deviation')
plt.xlabel('Time [s]')
plt.ylabel('Speed Deviation')
plt.legend()
plt.tight_layout()


# Transient Voltage Eq_prime:
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(t, eq_prime[i], label=f'Gen {i+1}')
plt.title('Transient Voltage Eq_Prime')
plt.xlabel('Time [s]')
plt.ylabel('Eq_prime')
plt.legend()
plt.tight_layout()



# -------------------------------

# ============================================================================
# 09  CURRENT, ANGLE, POWER, AND VOLTAGE PLOTS
# ============================================================================
# Plot Set 2: Dynamics of I_d, I_q, Voltage Angle, and Terminal Voltage
# -------------------------------
# Convert stored lists into arrays.
time_arr = np.array(time_list)
Id_arr = np.array(Id_all)      # Shape: (n_time, 3)
Iq_arr = np.array(Iq_all)
v_angles_arr = np.array(v_angles_all)
Vt_arr = np.array(Vt_all)
Pe_internal_arr = np.array(Pe_internal_all)

# Stator Current I_d:
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(time_arr, Id_arr[:, i], label=f'Gen {i+1}')
plt.title('Stator Current I_d')
plt.xlabel('Time [s]')
plt.ylabel('I_d')
plt.legend()
plt.tight_layout()


# Stator Current I_q:
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(time_arr, Iq_arr[:, i], label=f'Gen {i+1}')
plt.title('Stator Current I_q')
plt.xlabel('Time [s]')
plt.ylabel('I_q')
plt.legend()
plt.tight_layout()

# Voltage Angle (deg):
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(time_arr, v_angles_arr[:, i], label=f'Gen {i+1}')
plt.title('Voltage Angle (deg)')
plt.xlabel('Time [s]')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.tight_layout()


# Terminal Voltage V:
plt.figure(figsize=(8, 6))
for i in range(3):
    # Multiply each point in Pe_internal_arr by -100
    plt.plot(time_arr, Pe_internal_arr[:, i] * -100, label=f'Gen {i+1}')
plt.title('Pe Internal V')
plt.xlabel('Time [s]')
plt.ylabel('Voltage (pu)')  # Check if this y-label is still accurate after scaling!
plt.legend()
plt.tight_layout()
plt.show()



# Terminal Voltage V:
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(time_arr, Vt_arr[:, i], label=f'Gen {i+1}')
plt.title('Terminal Voltage V')
plt.xlabel('Time [s]')
plt.ylabel('Voltage (pu)')
plt.legend()
plt.tight_layout()
plt.show()
