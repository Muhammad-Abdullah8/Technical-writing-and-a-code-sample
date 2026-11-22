import numpy as np
import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1) Initialize IEEE 9-bus system ---
def create_network():
    net = pn.case9()
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    net.gen.drop(net.gen.index, inplace=True)
    for bus, (vm, va) in {0:(1.040,0), 1:(1.025,9.3), 2:(1.025,4.7)}.items():
        pp.create_ext_grid(net, bus=bus, vm_pu=vm, va_degree=va)
    net.bus["vn_kv"] = 345
    return net

net = create_network()

# --- 2) Base power flow to get reference Pe, Qe ---
pp.runpp(net, algorithm='nr')
P_e_ref = np.array([net.res_ext_grid.at[i, "p_mw"] for i in range(3)])
Q_e_ref = np.array([net.res_ext_grid.at[i, "q_mvar"] for i in range(3)])

# --- 3) Reference current calculation ---
bus_indices = [net.ext_grid.at[i, "bus"] for i in range(3)]
V_nominal_V = net.bus.loc[bus_indices, "vn_kv"].values * 1e3
S_ref_VA = np.sqrt(P_e_ref**2 + Q_e_ref**2) * 1e6
I_ref = S_ref_VA / (np.sqrt(3) * V_nominal_V)

# --- 4) System & machine constants ---
S_base = 100
omega_s = 2 * np.pi * 60
Pm_ref = [71.6, 163.0, 85.0]
Tg = [0.05]*3  # Governor time constants

# --- 5) Disturbance setup ---
load_idx = net.load[net.load['bus']==6].index[0]
original_p_mw = net.load.at[load_idx, 'p_mw']
power_incr = 72  # Load increase magnitude in MW

# --- 6) Initial state vector ---
initial_angles = [np.radians(va) for va in (0,9.3,4.7)]
y0 = []
for ang, Pm in zip(initial_angles, Pm_ref):
    y0 += [ang, omega_s, Pm]

def objective(x):
    # Decision variables: H[0:3], D[3:6], Kg[6:9]
    H = x[0:3]
    D = x[3:6]
    Kg = x[6:9]
    
    # Fresh network for simulation
    net = create_network()
    counter['n'] += 1

    def swing_governor(t, y):
        # Update load disturbance
        net.load.at[load_idx, 'p_mw'] = original_p_mw + power_incr if t >= 10 else original_p_mw
        # Set generator angles
        for i in range(3):
            net.ext_grid.at[i, 'va_degree'] = np.degrees(y[3*i])
        # Power flow
        pp.runpp(net, algorithm='nr', max_iteration=50)
        Pe = np.array([net.res_ext_grid.at[i,'p_mw'] for i in range(3)])
        
        dydt = np.zeros_like(y)
        for i in range(3):
            idx = 3*i
            delta, omega, Pm = y[idx], y[idx+1], y[idx+2]
            omega_dev = omega - omega_s
            dydt[idx] = omega_dev
            dydt[idx+1] = (1/(2*H[i]))*(Pm - Pe[i] - D[i]*omega_dev)
            dydt[idx+2] = -(1/Tg[i])*(Pm - Pm_ref[i] + Kg[i]*omega_dev)
        return dydt

    # Solve dynamic simulation
    sol = solve_ivp(swing_governor, (0,40), y0, method='Radau', t_eval=np.linspace(0,40,200))
    if not sol.success:
        return 1e6

    # Compute performance metrics
    t = sol.t
    y = sol.y
    omega = y[1::3]
    max_dom = np.max(np.abs(omega - omega_s))
    dt = np.diff(t)
    droc = np.diff(omega - omega_s, axis=1) / dt
    max_rocof = np.max(np.abs(droc))

    # Compute violations in electrical power and current
    max_Pe = np.zeros(3)
    max_I = np.zeros(3)
    for k, tk in enumerate(t):
        net.load.at[load_idx,'p_mw'] = original_p_mw + power_incr if tk >= 10 else original_p_mw
        for i in range(3):
            net.ext_grid.at[i,'va_degree'] = np.degrees(y[3*i,k])
        pp.runpp(net, algorithm='nr', max_iteration=50)
        for i in range(3):
            Pe_k = net.res_ext_grid.at[i,'p_mw']
            Qe_k = net.res_ext_grid.at[i,'q_mvar']
            S_va = np.sqrt(Pe_k**2 + Qe_k**2)*1e6
            V_act = net.bus.at[net.ext_grid.at[i,'bus'], 'vn_kv']*1e3
            I_k = S_va/(np.sqrt(3)*V_act)
            max_Pe[i] = max(max_Pe[i], Pe_k)
            max_I[i] = max(max_I[i], I_k)

    # Penalty for exceeding 15% limits
    pe_viol = np.maximum(0, max_Pe - (1+limit/100)*P_e_ref)
    i_viol = np.maximum(0, max_I - (1+limit/100)*I_ref)
    penalty = 2*(np.sum(pe_viol) + np.sum(i_viol))

    # Objective: speed dev + RoCoF + penalty
    obj = alpha_1*max_dom + alpha_2*max_rocof + penalty

    # Track best
    if best['sol'] is None or max_dom < np.max(np.abs(best['sol'].y[1::3] - omega_s)):
        best['sol'], best['x'] = sol, x

    # Early stopping
    pe_dev = 100*(max_Pe / P_e_ref - 1)
    i_dev = 100*(max_I / I_ref - 1)
    if (max_dom <= 0.5) and np.all(pe_dev <= limit) and np.all(i_dev <= limit):
        best['sol'], best['x'] = sol, x
        print(f"Early stop @ Iter {counter['n']}: Δω={max_dom:.4f}, Pe_dev={pe_dev.round(1)}%, I_dev={i_dev.round(1)}%")
        raise Exception("Early stopping: all criteria met")

    # Progress print
    print(f"{counter['n']:2d}: Δω={max_dom:.3f} | RoCoF={max_rocof:.3f} | Penalty={penalty:.1f} | "
          f"Pe_dev={[f'{d:.1f}%' for d in pe_dev]} | I_dev={[f'{d:.1f}%' for d in i_dev]} | "
          f"H={H} | D={D} | Kg={Kg}")

    return obj

# --- 7) Optimization setup ---
counter = {'n': 0}
best = {'sol': None, 'x': None}
limit = 25  # percent limit on P and I
alpha_1 = 1
alpha_2 = 1
bounds = [(1,20)]*3 + [(0.2,20)]*3 + [(1,20)]*3  # H, D, Kg bounds

# Run optimizer
try:
    res = differential_evolution(objective, bounds, maxiter=50, popsize=5,
                                 tol=1e-5, disp=False, polish=True, updating='deferred', workers=1)
except Exception as e:
    print(f"Optimization stopped: {e}")

# --- 8) Result plotting ---
if best['sol'] is not None:
    sol = best['sol']
    x_opt = best['x']
    H_opt, D_opt, Kg_opt = x_opt[:3], x_opt[3:6], x_opt[6:9]
    print("Optimal parameters:")
    print(f"H = {H_opt}\nD = {D_opt}\nKg = {Kg_opt}")

    t = sol.t
    y = sol.y
    dom = y[1::3] - omega_s
    # Plot Δω
    plt.figure(figsize=(8,4))
    for i in range(3): plt.plot(t, dom[i], label=f'Gen{i+1}')
    plt.axhline(0.1, linestyle='--')
    plt.axhline(-0.1, linestyle='--')
    plt.title('Speed Deviation Δω')
    plt.xlabel('Time (s)')
    plt.ylabel('rad/s')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()

    # Plot Pe
    net = create_network()
    Pe_log = []
    for k, tk in enumerate(t):
        net.load.at[load_idx,'p_mw'] = original_p_mw + power_incr if tk >= 10 else original_p_mw
        for i in range(3): net.ext_grid.at[i,'va_degree'] = np.degrees(y[3*i,k])
        pp.runpp(net, algorithm='nr', max_iteration=50)
        Pe_log.append([net.res_ext_grid.at[i,'p_mw'] for i in range(3)])
    Pe_log = np.array(Pe_log).T
    plt.figure(figsize=(8,4))
    for i in range(3): plt.plot(t, Pe_log[i], label=f'Gen{i+1}')
    plt.title('Electrical Power Pe (MW)')
    plt.xlabel('Time (s)')
    plt.ylabel('MW')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()
else:
    print("No feasible solution found.")
