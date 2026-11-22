"""Optimize governor gains under a generator-disconnection contingency.

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
from scipy.linalg import eig
from scipy.optimize import minimize


# -------------------------------
# Define System Parameters
# -------------------------------

# ============================================================================
# 01  SYSTEM BASE VALUES AND MACHINE PARAMETERS
# ============================================================================
V_base = 345e3
S_base = 100e6
Z_base = (V_base ** 2) / S_base

H = [23.64, 6.4, 3.01]         # Inertia constants
Tv = [0.05, 0.05, 0.05]         # Governor time constants
Tt = [7.0, 7.0, 7.0]            # Turbine time constants
omega_s = 2 * np.pi * 60        # Synchronous speed
Vout = [345, 345, 345]          # Terminal voltage 

# Kg = [(20 * (S_base/1e6)) / omega_s for _ in range(3)]
# Kg = [5.3, 5.4, 5.3]

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
Kg = [5.3, 5.3, 5.3]


# ============================================================================
# 02  EXCITATION AND VOLTAGE-CONTROL PARAMETERS
# ============================================================================
# PI Controller Parameters
Kp = [1, 1, 1]  # Proportional gains
Ki = [0.01, 0.01, 0.01]  # Integral gains
V_ref = [1.040, 1.025, 1.025]


# ============================================================================
# 03  DERIVED DAMPING COEFFICIENTS
# ============================================================================
Xq_prime_mult_base = [factor * Z_base for factor in Xq_prime]
Xq_double_prime_mult_base = [factor * Z_base for factor in Xq_double_prime]
Xq_mult_base = [factor * Z_base for factor in Xq]

D = []
for i in range(len(Xq)):
    D_i = (((Xq_prime_mult_base[i] - Xq_double_prime_mult_base[i]) / (Xq_mult_base[i] + Xq_prime_mult_base[i])) *
        (Xq_prime_mult_base[i] / Xq_double_prime_mult_base[i])) * (Tqo_double_prime[i] * Vout[i])
    D.append(D_i)
# print("Damping values:", D)
# -------------------------------



# ============================================================================
# 04  OPTIMIZATION STATE
# ============================================================================
call_count =0
contingency_applied = False  # Global flag
# Optimization function


# ============================================================================
# 05  DYNAMIC SIMULATION AND CONTINGENCY MODEL
# ============================================================================
def run_dynamic_simulation(Kg):
    global call_count, contingency_applied
    call_count += 1    
    print(f"Iteration {call_count}: Kg = {Kg}")  

    # -------------------------------
    # Build the IEEE 9-bus system
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
    # Retrieve power outputs from network solution
    Pe_1 = {net.ext_grid.at[idx, "bus"]: net.res_ext_grid.at[idx, "p_mw"] for idx in net.ext_grid.index}
    Qe_1 = {net.ext_grid.at[idx, "bus"]: net.res_ext_grid.at[idx, "q_mvar"] for idx in net.ext_grid.index}
    Pm_ref, Qm_ref = [], []
    for i in range(3):
        Pe_initial, Qe_initial = Pe_1[i], Qe_1[i]
        Pm_ref.append(Pe_initial)
        Qm_ref.append(Qe_initial)

    # -------------------------------
    # Global Storage for Dynamic Quantities
    # -------------------------------
# -------------------------------
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


    # Differential Equations: swing governor dynamics
    # -------------------------------
    def swing_governor(t, y):
        gen_tripping_time =8
        load_increment_time = 80
        num_generators = 3  
        dydt = np.zeros(len(y))
        
        # # Apply load increase at t=8
        if t >= load_increment_time:
            net.load.loc[net.load['bus'] == 8, 'p_mw'] = 190

        
        disconnect_gen_idx = 2
        # Disconnect Generator 3 at t=15s
        if t >= gen_tripping_time and disconnect_gen_idx in net.ext_grid.bus.values:
            # Remove Generator 3 from the ext_grid
            gen3_idx = net.ext_grid[net.ext_grid.bus == disconnect_gen_idx].index
            net.ext_grid = net.ext_grid.drop(gen3_idx)
            Pm_ref[disconnect_gen_idx] = 0  # No mechanical power
            Ef[disconnect_gen_idx] = 0      # No excitation voltage
            Kg[disconnect_gen_idx] = 0
            # print(f"Disconnected Generator 3 at t={t:.2f}s")

        Id_list, Iq_list = [], []
        v_angles = []
        Vt_list = []

        # First pass: Calculate voltages and angles for connected generators
        connected_gens = []
        for i in range(3):
            base = i * 7
            if t >= gen_tripping_time and i == disconnect_gen_idx:
                # Generator 3 is disconnected
                Id_list.append(0)
                Iq_list.append(0)
                v_angles.append(0)
                Vt_list.append(0)
                continue
                
            connected_gens.append(i)
            Eq_prime = y[base+2]
            Ed_prime = y[base+3]
            Vd = Ed_prime
            Vq = Eq_prime
            voltage_magnitude = np.sqrt(Vd**2 + Vq**2)
            voltage_angle = np.arctan2(Vq, Vd) + (y[base]-y[0])*1

            v_angles.append(np.degrees(voltage_angle))
            Vt_list.append(voltage_magnitude)

            # Update network voltages
            if i in net.ext_grid.bus.values:
                ext_grid_idx = net.ext_grid[net.ext_grid.bus == i].index[0]
                net.ext_grid.at[ext_grid_idx, "vm_pu"] = voltage_magnitude
                net.ext_grid.at[ext_grid_idx, "va_degree"] = np.degrees(voltage_angle)

        # Run power flow only if there are connected generators
        if connected_gens:
            pp.runpp(net, algorithm='nr', numba=False)
        
        # Second pass: Calculate currents and derivatives
        pe_interal_data = []
        for i in range(3):
            base = i * 7
            if t >= gen_tripping_time and i == disconnect_gen_idx:
                # Generator 3 is disconnected - zero out derivatives
                dydt[base:base+7] = 0
                pe_interal_data.append(0)
                continue
                
            # Get power values for connected generators
            if i in net.ext_grid.bus.values:
                ext_grid_idx = net.ext_grid[net.ext_grid.bus == i].index[0]
                Pe = net.res_ext_grid.at[ext_grid_idx, "p_mw"]
                Qe = net.res_ext_grid.at[ext_grid_idx, "q_mvar"]
            else:
                Pe, Qe = 0, 0  # Fallback if generator not found

            Eq_prime = y[base+2]
            Ed_prime = y[base+3]
            
            # Calculate currents
            Id_i, Iq_i = calc_current(Pe, Qe, Ed_prime, Eq_prime, Xd_prime[i], Xq_prime[i], Ra[i])
            Id_list.append(Id_i)
            Iq_list.append(Iq_i)
            
            # Differential equations
            delta, omega, Eq_prime, Ed_prime, Pm, v, int_V_err = y[base:base+7]
            Vt = Vt_list[i]
            
            error = V_ref[i] - Vt
            dydt[base+6] = error
            
            Ef_i = 10*Kp[i]*error + Ki[i]*int_V_err*100 + Ef[i]
            Pe_internal = (Eq_prime * Iq_i) + (Ed_prime * Id_i) + ((Xd_prime[i] - Xq_prime[i]) * Id_i * Iq_i)
            
            dydt[base] = omega - omega_s
            dydt[base+1] = (Pm + Pe_internal*100 - D[i]*(omega - omega_s)) / (2 * H[i])
            dydt[base+2] = (Ef_i - Eq_prime + Id_i*(Xd[i] - Xd_prime[i])) / Tdo_prime[i]
            dydt[base+3] = (-Ed_prime - Iq_i*(Xq[i] - Xq_prime[i]))
            dydt[base+4] = (-Pm + v) / Tt[i]
            dydt[base+5] = (-v - Kg[i]*(omega - omega_s) + Pm_ref[i]) / Tv[i]

            pe_interal_data.append(Pe_internal)

        # Store results
        time_list.append(t)
        Id_all.append(Id_list.copy())
        Iq_all.append(Iq_list.copy())
        v_angles_all.append(v_angles.copy())
        Vt_all.append(Vt_list.copy())
        Pe_internal_all.append(pe_interal_data.copy())
        
        return dydt


    # -------------------------------
    # Initial State Vector
    # (Each generator has 7 states: [delta, omega, Eq_prime, Ed_prime, Pm, v, integral_error])
    # -------------------------------
    y0 = [
        initial_angles_rad[0], omega_s, Eq_prime_inital_guess[0]*1, Ed_prime_inital_guess[0]*1, Pm_ref[0], Pm_ref[0], 0,
        initial_angles_rad[1], omega_s, Eq_prime_inital_guess[1]*1, Ed_prime_inital_guess[1]*1, Pm_ref[1], Pm_ref[1], 0,
        initial_angles_rad[2], omega_s, Eq_prime_inital_guess[2]*1, Ed_prime_inital_guess[2]*1, Pm_ref[2], Pm_ref[2], 0
    ]

    t_span = (0, 22)
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    # -------------------------------
    # Run the Simulation using BDF (stiff solver)
    # -------------------------------
    solution = solve_ivp(swing_governor, t_span, y0, method='RK23', t_eval=t_eval)

    omega_dev = np.abs(solution.y[1::7] - omega_s)
    max_omega_dev = np.max(omega_dev)
    print("Max_omega_dev: ", max_omega_dev)

    return max_omega_dev




# ============================================================================
# 06  LOCAL EIGENVALUE PENALTY
# ============================================================================
iteration_history = []
penalty_history = []
# revised eigen‑penalty
def compute_local_eigen_penalty(Kg):
    """
    Returns a penalty that is:
      • very large if any mode has Re(λ) > -eps (violates stability margin)
      • adds a smaller penalty if any mode has damping ratio < damping_thresh
    """
    penalty = 0.0
    eps = 1e-3              # stability margin (constraint (5): max Re(λ) ≤ -eps) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    damping_thresh = 0.02   # minimum acceptable damping ratio
    ignore_tol = 1e-4       # ignore eigenvalues |λ| < ignore_tol (trivial integrator mode)

    for i in range(3):
        # build local A‑matrix for machine i exactly as in (3a–3e)
        A = np.array([
            [0,                    1,                       0,                      0,                 0,              0],
            [0,           -D[i]/(2*H[i]),                   0,                      0,          1/(2*H[i]),         0],
            [0,                    0,             -1.0/Tdo_prime[i],                0,                 0,              0],
            [0,                    0,                       0,             -1.0/Tqo_prime[i],         0,              0],
            [0,                    0,                       0,                      0,          -1.0/Tt[i],     1.0/Tt[i]],
            [0,            -Kg[i]/Tv[i],                   0,                      0,                 0,         -1.0/Tv[i]]
        ])
        evs, _ = eig(A)
        # print(i," : " , evs)
        for ev in evs:
            re = np.real(ev)
            mag = np.abs(ev)

            # skip the trivial zero eigenvalue (angle reference mode)
            if mag < ignore_tol:
                continue

            # 1) stability margin penalty
            if re > -eps:
                # quadratic penalty ∝ (Re(λ)+eps)²
                penalty += 1e6 * (re + eps)**2

            # 2) damping ratio penalty
            zeta = -re/(mag + 1e-12)
            if zeta < damping_thresh:
                penalty += 1e4 * (damping_thresh - zeta)**2

    iteration_history.append(Kg.copy())
    penalty_history.append(penalty)
    return penalty

# Custom exception

# ============================================================================
# 07  EARLY STOPPING AND OBJECTIVE
# ============================================================================
class EarlyStop(Exception):
    pass

# Global to hold the most recent Kg
last_Kg = None

def objective(Kg):
    global last_Kg
    last_Kg = np.array(Kg, copy=True)  # store a copy each call

    dev = run_dynamic_simulation(Kg)
    # Early-stop check
    if dev < 1.75:
        print(f"Early stop: max ω-deviation = {dev:.3f} < 2")
        raise EarlyStop(f"dev={dev:.3f}")

    pen = compute_local_eigen_penalty(Kg)
    obj = dev + pen
    print(f"Kg={Kg}, dev={dev:.3f}, pen={pen:.1f}, obj={obj:.3f}")
    return obj


# ============================================================================
# 08  OPTIMIZATION EXECUTION
# ============================================================================
initial_Kg = [5.3, 5.3, 5.3]

try:
    result = minimize(
        objective,
        initial_Kg,
        bounds=[(1, 250)]*3,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3}
    )
except EarlyStop as e:
    print("Optimization halted early:", e)
    print("Optimized Kg at stop:", last_Kg)
else:
    print("Optimization completed normally.")
    print("Optimized Kg:", result.x)
