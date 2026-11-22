"""
IEEE 14-bus: AC OPF, generator trips and a sampled physical load-attack/protection game.

Research reproduction script. Original equations and parameter settings are retained.
See README.md for methodological cautions and setup.
"""

import pandapower as pp
import pandapower.networks as nw
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
logging.getLogger("pandapower").setLevel(logging.ERROR)

# ================================================================
# 1) LOAD IEEE-14 + PREPARE OPF
# ================================================================
BASE = nw.case14()
pp.replace_ext_grid_by_gen(BASE)
BASE.gen["controllable"] = True

# Relax limits first
BASE.bus["max_vm_pu"] = 1.10
BASE.bus["min_vm_pu"] = 0.90
BASE.line["max_loading_percent"] = 100

print("IEEE-14 OPF-ready.")

# ================================================================
# 2) GENERATOR COST FUNCTIONS (Quadratic)
# ================================================================
BASE.poly_cost.drop(BASE.poly_cost.index, inplace=True)

C1 = [10, 20, 30, 40, 50]
C2 = [0.01, 0.02, 0.025, 0.03, 0.035]

for g, (c1, c2) in enumerate(zip(C1, C2)):
    pp.create_poly_cost(
        BASE, g, et="gen",
        cp1_eur_per_mw=c1,
        cp2_eur_per_mw2=c2
    )

print("Generator cost curves added.")

# ================================================================
# 3) ACTIVE LOAD SHEDDING ONLY (sP)
# ================================================================
VoLL = 100000.0     # €/MWh
alpha = 0.0         # reactive shedding disabled

print("Adding active shedding...")

for _, row in BASE.load.iterrows():
    bus_i = row.bus

    shedP = pp.create_sgen(
        BASE,
        bus=bus_i,
        p_mw=0.0,
        q_mvar=0.0,
        controllable=True,
        name=f"ShedP@Bus{bus_i}"
    )
    pp.create_poly_cost(BASE, shedP, et="sgen", cp1_eur_per_mw=VoLL)

    BASE.sgen.at[shedP, "min_p_mw"] = 0.0
    BASE.sgen.at[shedP, "max_p_mw"] = row.p_mw + 20
    BASE.sgen.at[shedP, "min_q_mvar"] = 0.0
    BASE.sgen.at[shedP, "max_q_mvar"] = 0.0

print("Load shedding added.")

# ================================================================
# 4) FORCE SYSTEM STRESS (to ensure shedding)
# ================================================================
print("Applying forced-stress conditions...")

BASE.gen["max_p_mw"] *= 0.50     # reduce generator limits
if "max_i_ka" in BASE.line.columns:
    BASE.line["max_i_ka"] *= 0.30

BASE.bus["max_vm_pu"] = 1.03
BASE.bus["min_vm_pu"] = 0.97

print("System stressed.")

# ================================================================
# 5) SLACK GENERATOR
# ================================================================
BASE.gen.loc[:, "slack"] = False
BASE.gen.loc[0, "slack"] = True

print("Slack assigned.")

# ================================================================
# 6) HELPER FUNCTIONS
# ================================================================
def safe_runopp(net):
    try:
        pp.runopp(net, calculate_voltage_angles=True)
        return True
    except Exception as e:
        # print(f"OPF failed: {e}")
        return False

def compute_total_cost(net):
    rc = net.res_cost
    return float(rc.cost.sum()) if hasattr(rc, "cost") else float(rc)

def apply_load_attack_by_busindex(net, bus_idx, dP, dQ):
    """
    Physical load attack: changes the actual p_mw, q_mvar at a bus.
    Used in tri-level OPF game (not in SE-based FDI).
    """
    attacked = copy.deepcopy(net)
    loads = attacked.load.index[attacked.load.bus == bus_idx]
    for ld in loads:
        attacked.load.at[ld, "p_mw"] += dP
        attacked.load.at[ld, "q_mvar"] += dQ
    return attacked

# ================================================================
# 7) SCENARIO A — GENERATOR TRIPPING
# ================================================================
print("\n========== SCENARIO A: Generator Tripping ==========\n")

results_trip = []

slack_gen_idx = 0
gens = [g for g in BASE.gen.index if g != slack_gen_idx]

for g in gens:
    before = copy.deepcopy(BASE)
    after = copy.deepcopy(BASE)

    after.gen.at[g, "in_service"] = False
    after.gen.at[g, "min_p_mw"] = 0.0
    after.gen.at[g, "max_p_mw"] = 0.0

    if not safe_runopp(before) or not safe_runopp(after):
        print(f"OPF failed for generator {g} trip.")
        continue

    cost_before = compute_total_cost(before)
    cost_after = compute_total_cost(after)

    shedP = after.res_sgen.p_mw[after.sgen.name.str.contains("ShedP")].sum()
    dV = after.res_bus.vm_pu.values - before.res_bus.vm_pu.values

    results_trip.append({
        "gen": g,
        "bus": int(BASE.gen.bus.at[g]) + 1,
        "before": cost_before,
        "after": cost_after,
        "delta": cost_after - cost_before,
        "shedP": shedP,
        "dV": dV
    })

    print(f"Generator {g} at Bus {int(BASE.gen.bus.at[g]) + 1}")
    print(f"  Before: {cost_before:.2f} €")
    print(f"  After : {cost_after:.2f} €")
    print(f"  ΔCost : {cost_after - cost_before:.2f} €")
    print(f"  Shedding: {shedP:.4f} MW")
    print("-"*40)

# (Optional) Plot generator-tripping voltage deviations
plt.figure(figsize=(10, 5))
for r in results_trip:
    plt.plot(range(1, len(r["dV"]) + 1), r["dV"], marker="o", label=f"G{r['gen']} tripped")
plt.axhline(0, color='black')
plt.grid(True)
plt.xlabel("Bus")
plt.ylabel("ΔV (p.u.)")
plt.title("Voltage Deviations due to Generator Tripping")
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
# 8) TRI-LEVEL GAME:
#    Multi-Bus Attacker + Convex Protection Cost at Bus k
# ================================================================
print("\n========== TRI-LEVEL GAME (Physical Load Attack + Convex Protection) ==========\n")

k_bus = 9         # protected bus number (1..14)
k_idx = k_bus - 1 # internal index

# Attack capability (physical load distortion, not SE-FDI)
U_P_bar = 500.0
U_Q_bar = 100.0
beta_k  = 0.8     # protection effectiveness

# Convex protection cost: C(x) = c1 x + c2 x^2
c1 = 1000.0
c2 = 1500.0

def protection_cost(x):
    return c1 * x + c2 * (x ** 2)

# Nominal cost J_nominal (no attack)
nominal = copy.deepcopy(BASE)
safe_runopp(nominal)
J_nominal = compute_total_cost(nominal)
print(f"Nominal OPF cost = {J_nominal:.2f} €\n")

# ---------------- MIDDLE LEVEL: worst physical attack for given x_k ----------------
def worst_attack_cost_for_x(x_k):
    """
    For a given level of physical protection x_k at bus k, attacker can:
      - choose ANY bus i
      - apply (uP, uQ) in its admissible rectangle U_i(x_k)
    Here, U_i(x_k) shrinks only for the protected bus k.
    """
    worst_J = -1e12
    best_tuple = (None, 0.0, 0.0)

    for bus_idx in BASE.bus.index:

        # Shrink capability only at protected bus
        if bus_idx == k_idx:
            U_P = U_P_bar * (1 - beta_k * x_k)
            U_Q = U_Q_bar * (1 - beta_k * x_k)
        else:
            U_P = U_P_bar
            U_Q = U_Q_bar

        if U_P < 1e-6 and U_Q < 1e-6:
            continue

        # Four corners of the attack box
        UP = [+U_P, +U_P, -U_P, -U_P]
        UQ = [+U_Q, -U_Q, +U_Q, -U_Q]

        for uP, uQ in zip(UP, UQ):
            attacked = apply_load_attack_by_busindex(BASE, bus_idx, uP, uQ)

            if not safe_runopp(attacked):
                continue

            J = compute_total_cost(attacked)

            if J > worst_J:
                worst_J = J
                best_tuple = (bus_idx, uP, uQ)

    if worst_J < -1e10:
        return J_nominal, (None, 0.0, 0.0)

    return worst_J, best_tuple

# ---------------- UPPER LEVEL: defender chooses x_k ----------------
x_values = np.linspace(0.0, 1.0, 11)
best_x = None
best_obj = 1e12
best_attack_bus = None
best_attack = None
best_J_attack = None

print("Searching defender protection level x_k...\n")

for x_k in x_values:
    J_worst, (bus_idx_star, uP_star, uQ_star) = worst_attack_cost_for_x(x_k)
    total_obj = protection_cost(x_k) + J_worst

    print(f"x={x_k:.2f} → attack @ Bus {bus_idx_star+1 if bus_idx_star is not None else 'None'}, "
          f"uP={uP_star:.1f}, uQ={uQ_star:.1f}, "
          f"J={J_worst:.2f}, C(x)={protection_cost(x_k):.2f}, Total={total_obj:.2f}")

    if total_obj < best_obj:
        best_obj = total_obj
        best_x = x_k
        best_attack_bus = bus_idx_star
        best_attack = (uP_star, uQ_star)
        best_J_attack = J_worst

print("\n========== FINAL TRI-LEVEL RESULT ==========\n")
print(f"Optimal protection x* = {best_x:.2f}")
print(f"Attack bus            = {best_attack_bus + 1}")
print(f"Attack magnitudes     = uP={best_attack[0]:.1f} MW, uQ={best_attack[1]:.1f} Mvar")
print(f"Nominal cost          = {J_nominal:.2f} €")
print(f"Worst-case cost       = {best_J_attack:.2f} €")
print(f"Cost increase         = {best_J_attack - J_nominal:.2f} €")
print(f"Total Objective       = {best_obj:.2f} €")

# Re-simulate best physical attack scenario for voltages & shedding
best_nom = copy.deepcopy(BASE)
safe_runopp(best_nom)

best_att = apply_load_attack_by_busindex(BASE, best_attack_bus,
                                         best_attack[0], best_attack[1])
safe_runopp(best_att)

V0 = best_nom.res_bus.vm_pu.values
V1 = best_att.res_bus.vm_pu.values
dV = V1 - V0

shedP = best_att.res_sgen.p_mw[best_att.sgen.name.str.contains("ShedP")].sum()
print(f"\nTotal active shedding under worst-case physical attack = {shedP:.4f} MW")

plt.figure(figsize=(10,5))
plt.plot(range(1, len(V0)+1), V0, marker='o', label="Before attack")
plt.plot(range(1, len(V1)+1), V1, marker='s', label="After False Data Injection")
plt.grid(True)
plt.xlabel("Bus")
plt.ylabel("Voltage (p.u.)")
plt.title("Voltages: Before vs After FDI Attack")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.bar(range(1, len(dV)+1), dV, color="purple")
plt.axhline(0, color="black")
plt.grid(axis='y')
plt.xlabel("Bus")
plt.ylabel("ΔV (p.u.)")
plt.title("Voltage Deviations due to FDI Attack")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.bar(["Nominal", "Attacked"], [J_nominal, best_J_attack],
        color=["steelblue", "red"])
plt.grid(axis='y')
plt.ylabel("Cost (€)")
plt.title("Cost Before vs After Worst-Case Physical Attack")
plt.tight_layout()
plt.show()

# # =================================================================
# # 9) NEW SECTION: DC STATE ESTIMATION + STEALTHY FDI PIPELINE
# # =================================================================
# print("\n========== DC STATE ESTIMATION + STEALTHY FDI (Measurement Attack) ==========\n")

# # For SE, we use the nominal OPF solution as the "true" operating point
# SE_NET = copy.deepcopy(BASE)
# if not safe_runopp(SE_NET):
#     raise RuntimeError("Nominal OPF for SE did not converge.")

# # ---------------- Build DC nodal susceptance matrix B ----------------
# n_bus = len(SE_NET.bus)
# B = np.zeros((n_bus, n_bus))

# for _, line in SE_NET.line.iterrows():
#     i = int(line.from_bus)
#     j = int(line.to_bus)

#     # approximate total reactance from x_ohm_per_km * length_km
#     x = float(line.x_ohm_per_km * line.length_km)
#     if abs(x) < 1e-6:
#         continue

#     bij = 1.0 / x
#     B[i, i] += bij
#     B[j, j] += bij
#     B[i, j] -= bij
#     B[j, i] -= bij

# # Slack bus (same as OPF slack)
# slack_bus = int(SE_NET.gen.bus.at[0])

# mask = np.ones(n_bus, dtype=bool)
# mask[slack_bus] = False
# n_state = n_bus - 1  # angles of non-slack buses
# B_red = B[np.ix_(mask, mask)]

# # ---------------- True state angles (θ_true) from DC power flow ----------------
# # Use net injections from OPF as P = B θ
# P_inj_full = SE_NET.res_bus.p_mw.values  # net injection at each bus
# P_red = P_inj_full[mask]

# theta_red = np.linalg.solve(B_red, P_red)
# theta_full = np.zeros(n_bus)
# theta_full[mask] = theta_red  # slack angle = 0

# x_true = theta_red.copy()     # state vector (non-slack angles)

# # ---------------- Build linear measurement model z = H x ----------------
# H_rows = []
# z_true = []

# # 1) Bus injection measurements at all NON-slack buses
# for i in range(n_bus):
#     if i == slack_bus:
#         continue
#     # Row for P_i = sum_k B[i,k] * θ_k
#     row = B[i, mask]  # only non-slack angles are in the state
#     H_rows.append(row)

#     # True measurement value P_i from OPF
#     z_true.append(P_inj_full[i])

# # 2) Line flow measurements P_ij for every line
# for _, line in SE_NET.line.iterrows():
#     i = int(line.from_bus)
#     j = int(line.to_bus)

#     x = float(line.x_ohm_per_km * line.length_km)
#     if abs(x) < 1e-6:
#         continue
#     bij = 1.0 / x

#     row = np.zeros(n_state)

#     # mapping: bus index -> state index
#     # state index is position of bus in mask array, skipping slack
#     if i != slack_bus:
#         idx_i = np.where(mask & (np.arange(n_bus) == i))[0][0]
#         row[idx_i] += bij
#     if j != slack_bus:
#         idx_j = np.where(mask & (np.arange(n_bus) == j))[0][0]
#         row[idx_j] -= bij

#     H_rows.append(row)

#     # true line flow P_ij from DC formula bij*(θ_i - θ_j)
#     P_ij_true = bij * (theta_full[i] - theta_full[j])
#     z_true.append(P_ij_true)

# H = np.vstack(H_rows)
# z_true = np.array(z_true)

# print(f"Number of states  (angles)   : {n_state}")
# print(f"Number of measurements (z)   : {len(z_true)}")

# # ---------------- WLS State Estimation (linear DC) ----------------
# # With DC and these measurements, model is linear: z = H x + e
# # WLS solution: x_hat = (H^T H)^-1 H^T z   (assuming W = I)
# G = H.T @ H
# x_hat = np.linalg.solve(G, H.T @ z_true)

# print("\nState estimation (no attack):")
# for idx, bus in enumerate(SE_NET.bus.index[mask]):
#     print(f"  Bus {bus+1}: θ_true = {theta_full[bus]: .4f} rad, θ_hat = {x_hat[idx]: .4f} rad")

# # =================================================================
# # STEALTHY FDI ATTACK:  z_a = z + a, with a = H c
# # =================================================================
# ATTACK_BUS = 9   # choose any non-slack bus to target in state space

# if ATTACK_BUS == slack_bus:
#     raise ValueError("Choose ATTACK_BUS different from slack bus for FDI example.")

# # Find index of ATTACK_BUS in state vector
# state_idx_attack = np.where(mask & (np.arange(n_bus) == ATTACK_BUS))[0][0]

# # Choose a state perturbation direction c (only one bus angle is shifted)
# c = np.zeros(n_state)
# c[state_idx_attack] = 0.1  # ~ 5.7 degrees shift in estimated angle

# # Build stealthy FDI vector a = H c
# a = H @ c

# # Attacked measurements
# z_attack = z_true + a

# # New SE with attacked measurements
# x_hat_attack = np.linalg.solve(G, H.T @ z_attack)

# print("\nStealthy FDI attack applied:")
# print(f"  Target bus (in state)      : Bus {ATTACK_BUS}")
# print(f"  Angle perturbation c[bus]  : {c[state_idx_attack]: .4f} rad")

# print("\nState estimation under FDI:")
# for idx, bus in enumerate(SE_NET.bus.index[mask]):
#     print(f"  Bus {bus+1}: θ_hat (no attack) = {x_hat[idx]: .4f} rad, "
#           f"θ_hat (FDI) = {x_hat_attack[idx]: .4f} rad, "
#           f"Δθ = {x_hat_attack[idx] - x_hat[idx]: .4f} rad")

# # Compare estimated bus injections before/after attack
# theta_est_no_attack = np.zeros(n_bus)
# theta_est_no_attack[mask] = x_hat

# theta_est_attack = np.zeros(n_bus)
# theta_est_attack[mask] = x_hat_attack

# P_est_no_attack = B @ theta_est_no_attack
# P_est_attack    = B @ theta_est_attack
# dP_est          = P_est_attack - P_est_no_attack

# print("\nEstimated active injections (DC) before vs after FDI:")
# for bus in SE_NET.bus.index:
#     print(f"  Bus {bus+1}: P_est(no attack) = {P_est_no_attack[bus]: 8.4f} MW, "
#           f"P_est(FDI) = {P_est_attack[bus]: 8.4f} MW, "
#           f"ΔP = {dP_est[bus]: 7.4f} MW")

# plt.figure(figsize=(10,4))
# plt.bar(range(1, n_bus+1), dP_est)
# plt.axhline(0, color="black")
# plt.grid(axis='y')
# plt.xlabel("Bus")
# plt.ylabel("ΔP_est (MW)")
# plt.title("Change in Estimated Bus Injections due to Stealthy FDI")
# plt.tight_layout()
# plt.show()

# print("\nNOTE:")
# print("  • The tri-level game above uses PHYSICAL load changes (uP, uQ).")
# print("  • This SE + FDI block models a PURE MEASUREMENT attack (no physical change).")
# print("  • The FDI is stealthy because a = H c, so residuals in the linear DC model do not change.")
