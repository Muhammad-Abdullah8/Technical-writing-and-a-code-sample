import numpy as np
import pandapower as pp
import pandapower.networks as pn
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
import warnings

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

omega_s = 2*np.pi*60
Pm_ref = [71.6, 163.0, 85.0]
Tt = [0.5]*3    # turbine time constants
Tv = [0.08]*3   # governor servo time constants

# 2) Disturbance setup (50 MW step at t=10s)
load_idx   = net.load[net.load['bus']==6].index[0]
orig_p_mw  = net.load.at[load_idx,'p_mw']
power_incr = 100

# 3) Initial state vector [δ, ω, Pm, v] for each generator
initial_angles = [np.radians(va) for va in (0, 9.3, 4.7)]
y0 = []
for ang, Pm in zip(initial_angles, Pm_ref):
    y0 += [ang, omega_s, Pm, Pm]  # start valve pos = Pm_ref

# 4) Simulation with 2nd-order governor
def simulate_2nd_order(x):
    H, D, Kg = x[:3], x[3:6], x[6:9]
    net_sim = create_network()
    net_sim.load.at[load_idx,'p_mw'] = orig_p_mw

    def swing_governor_2nd(t, y):
        if t >= 8.0:
            net_sim.load.at[load_idx,'p_mw'] = orig_p_mw + power_incr

        # update angles
        for ext_idx in net_sim.ext_grid.index:
            bus_idx = net_sim.ext_grid.at[ext_idx,"bus"]
            delta = y[bus_idx*4]
            net_sim.ext_grid.at[ext_idx,"va_degree"] = np.degrees(delta)

        pp.runpp(net_sim, algorithm='nr', max_iteration=50)
        Pe = np.array([net_sim.res_ext_grid.at[i,'p_mw'] for i in range(3)])

        dydt = np.zeros_like(y)
        for i in range(3):
            base = 4*i
            delta, omega, Pm, v = y[base:base+4]
            omega_dev = omega - omega_s

            # swing
            dydt[base]   = omega_dev
            dydt[base+1] = (Pm - Pe[i] - D[i]*omega_dev) / (2*H[i])
            # turbine & governor
            dydt[base+2] = (-Pm + v) / Tt[i]
            dydt[base+3] = (-v - Kg[i]*omega_dev + Pm_ref[i]) / Tv[i]
        return dydt

    sol = solve_ivp(swing_governor_2nd, (0,25), y0, method='RK23',
                    t_eval=np.linspace(0,25,201))
    omega = sol.y[1::4]
    t = sol.t
    # plt.figure(figsize=(8,5))
    # for i in range(3): plt.plot(t, omega[i]-omega_s, label=f'Gen{i+1}')
    # plt.title('Rotor Speed Deviation'); plt.xlabel('Time [s]'); plt.ylabel('\u0394\omega [rad/s]'); plt.legend(); plt.tight_layout()
    # plt.show()
    return sol if sol.success else None

# 5) Optimization setup and objective
counter = {'n': 0}
limit   = 35      # percent limit on P and I
alpha_1 = 1       # weight for nadir
alpha_2 = 1       # weight for RoCoF
best    = {'sol': None, 'x': None}

def compute_objective(x):
    counter['n'] += 1
    H, D, Kg = x[:3], x[3:6], x[6:9]

    sol = simulate_2nd_order(x)
    if sol is None:
        return 1e6

    t = sol.t
    y = sol.y
    omega_dev = np.vstack([y[4*i+1] - omega_s for i in range(3)])

    # nadir metrics
    nadirs     = np.min(omega_dev, axis=1)
    max_nadir = np.max(np.abs(nadirs))
    sum_nadir = np.sum(np.abs(nadirs))

    # RoCoF
    droc      = np.diff(omega_dev, axis=1) / np.diff(t)
    max_rocof = np.max(np.abs(droc))

    # power/current constraints
    max_Pe = np.zeros(3); max_I = np.zeros(3)
    net_chk = create_network()
    for k, tk in enumerate(t):
        net_chk.load.at[load_idx,'p_mw'] = orig_p_mw + (power_incr if tk>=10 else 0)
        for i in range(3):
            net_chk.ext_grid.at[i,'va_degree'] = np.degrees(y[4*i, k])
        pp.runpp(net_chk, algorithm='nr', max_iteration=50)
        for i in range(3):
            Pe_k = net_chk.res_ext_grid.at[i,'p_mw']
            Qe_k = net_chk.res_ext_grid.at[i,'q_mvar']
            S_va = np.sqrt(Pe_k**2 + Qe_k**2)*1e6
            V   = net_chk.bus.at[net_chk.ext_grid.at[i,'bus'],'vn_kv']*1e3
            I_k = S_va/(np.sqrt(3)*V)
            max_Pe[i] = max(max_Pe[i], Pe_k)
            max_I[i]  = max(max_I[i], I_k)

    pe_dev = 100*(max_Pe / P_e_ref - 1)
    i_dev  = 100*(max_I  / I_ref   - 1)
    pe_viol = np.maximum(0, pe_dev - limit)
    i_viol  = np.maximum(0, i_dev  - limit)
    penalty = 1e3*(pe_viol.sum() + i_viol.sum())

    # combined objective
    obj = alpha_1*max_nadir + alpha_2*max_rocof + penalty

    # track best
    if best['sol'] is None or max_nadir < np.max(np.abs(best['sol'].y[1::4] - omega_s)):
        best['sol'], best['x'] = sol, x

    # early stopping
    if (max_nadir <= 0.7) and np.all(pe_dev <= limit) and np.all(i_dev <= limit):
        print(f"Early stop @ Iter {counter['n']}: Δω={max_nadir:.4f}, Pe_dev={pe_dev.round(1)}%, I_dev={i_dev.round(1)}%")
        raise Exception("Early stopping: criteria met")

    # progress print
    print(f"{counter['n']:3d}: Δω={max_nadir:.3f} | RoCoF={max_rocof:.3f} | Penalty={penalty:.1f} | "
          f"Pe_dev={[f'{d:.1f}%' for d in pe_dev]} | I_dev={[f'{d:.1f}%' for d in i_dev]} | "
          f"H={H} | D={D} | Kg={Kg}")

    return obj

# 6) Run optimizer
bounds = [(1,20)]*3 + [(0.2,20)]*3 + [(1,20)]*3
try:
    res = dual_annealing(compute_objective, bounds, maxiter=100)
except Exception as e:
    print(f"Optimization stopped: {e}")

# 7) Plot results if available
if best['sol'] is not None:
    sol   = best['sol']
    x_opt = best['x']
    H_opt, D_opt, Kg_opt = x_opt[:3], x_opt[3:6], x_opt[6:9]

    print("\nOptimal parameters:")
    print(f"  H  = {H_opt}")
    print(f"  D  = {D_opt}")
    print(f"  Kg = {Kg_opt}\n")

    t = sol.t
    y = sol.y
    dom = np.vstack([y[4*i+1] - omega_s for i in range(3)])

    # Plot Δω
    plt.figure(figsize=(8,4))
    for i in range(3):
        plt.plot(t, dom[i], label=f'Gen{i+1}')
    plt.axhline( 0.1, linestyle='--')
    plt.axhline(-0.1, linestyle='--')
    plt.title('Speed Deviation Δω')
    plt.xlabel('Time (s)'); plt.ylabel('rad/s')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()

    # Plot Pe
    net_plot = create_network()
    Pe_log = []
    for k, tk in enumerate(t):
        net_plot.load.at[load_idx,'p_mw'] = orig_p_mw + (power_incr if tk>=10 else 0)
        for i in range(3):
            net_plot.ext_grid.at[i,'va_degree'] = np.degrees(y[4*i,k])
        pp.runpp(net_plot, algorithm='nr', max_iteration=50)
        Pe_log.append([net_plot.res_ext_grid.at[i,'p_mw'] for i in range(3)])
    Pe_log = np.array(Pe_log).T

    plt.figure(figsize=(8,4))
    for i in range(3):
        plt.plot(t, Pe_log[i], label=f'Gen{i+1}')
    plt.title('Electrical Power Pe (MW)')
    plt.xlabel('Time (s)'); plt.ylabel('MW')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()
else:
    print("No feasible solution found.")
