import numpy as np
import pandapower as pp
import pandapower.networks as pn
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing, basinhopping
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1) Define network and base values ---
def create_network():
    net = pn.case9()
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    net.gen.drop(net.gen.index, inplace=True)
    for bus, (vm, va) in {0:(1.040,0), 1:(1.025,9.3), 2:(1.025,4.7)}.items():
        pp.create_ext_grid(net, bus=bus, vm_pu=vm, va_degree=va)
    net.bus["vn_kv"] = 345
    return net

net = create_network()
pp.runpp(net, algorithm='nr')
P_e_ref = np.array([net.res_ext_grid.at[i,"p_mw"] for i in range(3)])
Q_e_ref = np.array([net.res_ext_grid.at[i,"q_mvar"] for i in range(3)])
bus_indices = [net.ext_grid.at[i,"bus"] for i in range(3)]
V_nominal_V = net.bus.loc[bus_indices,"vn_kv"].values * 1e3
S_ref_VA = np.sqrt(P_e_ref**2 + Q_e_ref**2)*1e6
I_ref = S_ref_VA / (np.sqrt(3)*V_nominal_V)

S_base = 100
omega_s = 2*np.pi*60
Pm_ref = [71.6, 163.0, 85.0]
Tg = [0.05]*3  # governor time constants

# 2) Disturbance setup (50 MW step at t=10s)
load_idx = net.load[net.load['bus']==6].index[0]
orig_p_mw = net.load.at[load_idx,'p_mw']
power_incr = 72

# 3) Initial state vector [δ1, ω1, Pm1, δ2, ω2, Pm2, ...]
initial_angles = [np.radians(va) for va in (0,9.3,4.7)]
y0 = []
for ang, Pm in zip(initial_angles, Pm_ref):
    y0 += [ang, omega_s, Pm]

# 4) Simulation function: solve swing-governor ODEs
def simulate_response(x):
    H = x[0:3]; D = x[3:6]; Kg = x[6:9]
    net_sim = create_network()
    net_sim.load.at[load_idx, 'p_mw'] = orig_p_mw

    def swing_governor(t, y):
        # Apply load step at t>=10s
        if t >= 10.0:
            net_sim.load.at[load_idx,'p_mw'] = orig_p_mw + power_incr
        # Set generator angles for power flow
        for i in range(3):
            net_sim.ext_grid.at[i,'va_degree'] = np.degrees(y[3*i])
        pp.runpp(net_sim, algorithm='nr', max_iteration=50)
        Pe = np.array([net_sim.res_ext_grid.at[i,'p_mw'] for i in range(3)])
        # State derivatives
        dydt = np.zeros_like(y)
        for i in range(3):
            idx = 3*i
            delta, omega, Pm = y[idx], y[idx+1], y[idx+2]
            omega_dev = omega - omega_s
            dydt[idx]   = omega_dev
            dydt[idx+1] = (1/(2*H[i]))*(Pm - Pe[i] - D[i]*omega_dev)
            dydt[idx+2] = -(1/Tg[i])*(Pm - Pm_ref[i] + Kg[i]*omega_dev)
        return dydt

    sol = solve_ivp(swing_governor, (0,25), y0, method='RK23',
                    t_eval=np.linspace(0,25,201))
    if not sol.success:
        return None
    return sol

counter = {'n': 0}
limit = 25  # percent limit on P and I
alpha_1 = 1
alpha_2 = 1
best = {'sol': None, 'x': None}

# 5) Objective function: sum of nadirs + RoCoF + constraint penalties
def compute_objective(x):
    counter['n'] += 1
    H = x[0:3]
    D = x[3:6]
    Kg = x[6:9]

    sol = simulate_response(x)
    if sol is None:
        return 1e6
    t = sol.t
    y = sol.y
    omega_dev = y[1::3] - omega_s  # speed deviations for each gen

    # Compute nadirs (most negative deviation) and sum them
    nadirs = np.min(omega_dev, axis=1)  # (negative or zero)
    sum_nadir = np.sum(np.abs(nadirs))
    max_nadir = np.max(np.abs(nadirs))

    # Compute RoCoF (max slope of freq deviation)
    droc = np.diff(omega_dev, axis=1) / np.diff(t)
    max_rocof = np.max(np.abs(droc))

    # Check power and current constraints (25% above reference)
    max_Pe = np.zeros(3); max_I = np.zeros(3)
    net_sim = create_network()
    for k, tk in enumerate(t):
        if tk >= 10.0:
            net_sim.load.at[load_idx,'p_mw'] = orig_p_mw + power_incr
        else:
            net_sim.load.at[load_idx,'p_mw'] = orig_p_mw
        for i in range(3):
            net_sim.ext_grid.at[i,'va_degree'] = np.degrees(y[3*i,k])
        pp.runpp(net_sim, algorithm='nr', max_iteration=50)
        for i in range(3):
            Pe_k = net_sim.res_ext_grid.at[i,'p_mw']
            Qe_k = net_sim.res_ext_grid.at[i,'q_mvar']
            S_va = np.sqrt(Pe_k**2 + Qe_k**2)*1e6
            V_act = net_sim.bus.at[net_sim.ext_grid.at[i,'bus'],'vn_kv']*1e3
            I_k = S_va/(np.sqrt(3)*V_act)
            max_Pe[i] = max(max_Pe[i], Pe_k)
            max_I[i]  = max(max_I[i], I_k)
   
    pe_viol = np.maximum(0, max_Pe - (1+limit/100)*P_e_ref)
    i_viol = np.maximum(0, max_I - (1+limit/100)*I_ref)
    penalty = 1e3*(np.sum(pe_viol) + np.sum(i_viol))

    # Combined objective: sum_nadir + weighted RoCoF + penalties
    obj = alpha_1*max_nadir + alpha_2*max_rocof + penalty
    
    # Track best
    if best['sol'] is None or max_nadir < np.max(np.abs(best['sol'].y[1::3] - omega_s)):
        best['sol'], best['x'] = sol, x

    pe_dev = 100*(max_Pe / P_e_ref - 1)
    i_dev = 100*(max_I / I_ref - 1)
    

    if (max_nadir <= 0.7) and np.all(pe_dev <= limit) and np.all(i_dev <= limit):
        best['sol'], best['x'] = sol, x
        print(f"Early stop @ Iter {counter['n']}: Δω={max_nadir:.4f}, Pe_dev={pe_dev.round(1)}%, I_dev={i_dev.round(1)}%")
        raise Exception("Early stopping: all criteria met")

    
    # Progress print
    print(f"{counter['n']:2d}: Δω={max_nadir:.3f} | RoCoF={max_rocof:.3f} | Penalty={penalty:.1f} | "
          f"Pe_dev={[f'{d:.1f}%' for d in pe_dev]} | I_dev={[f'{d:.1f}%' for d in i_dev]} | "
          f"H={H} | D={D} | Kg={Kg}")

    return obj

# 6) Run global optimization (e.g. Dual Annealing or Basin-Hopping)
bounds = [(1,20)]*3 + [(0.2,20)]*3 + [(1,20)]*3  # H, D, Kg bounds
try:
    result = dual_annealing(compute_objective, bounds, maxiter=100)
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
        net.load.at[load_idx,'p_mw'] = orig_p_mw + power_incr if tk >= 10 else orig_p_mw
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
