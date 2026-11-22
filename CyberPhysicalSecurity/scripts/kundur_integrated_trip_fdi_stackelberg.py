"""
Kundur two-area: generator-trip criticality, defender–attacker protection games and FDI-inspired EMS/truth evaluations.

Research reproduction script. Original equations and parameter settings are retained.
See README.md for methodological cautions and setup.
"""

import copy

import itertools

import warnings

import logging

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pandapower as pp



warnings.filterwarnings("ignore")

logging.getLogger("pandapower").setLevel(logging.ERROR)



# ============================================================

# GLOBAL SETTINGS

# ============================================================

SBASE = 100.0

VN_KV = 230.0

F_HZ = 50.0



GEN_CP1 = 30.0

VoLL = 1000.0

BIG_PEN = 1e6

VMIN_COLLAPSE = 0.75



# Generator-trip game settings

TRIP_MAX_PROTECT = 2

TRIP_PROTECT_COST = 5000.0



# FDI game settings

FDI_LEVELS = np.arange(0.0, 275.0, 25.0)   # 0,25,...,250 MW

FDI_ATTACK_BUSES = {7: "Area 1", 9: "Area 2"}

FDI_MAX_PROTECT = 1

FDI_PROTECT_COST = 3000.0

FDI_PROTECT_EFFECT = 0.70



# Criticality weights

W = {"LS": 1.0, "VV": 0.8, "OL": 0.8, "TF": 0.8, "IN": 0.6}





# ============================================================

# BASIC HELPERS

# ============================================================

def pu_to_ohm(pu, vn_kv=VN_KV, sbase_mva=SBASE):

    zbase_ohm = (vn_kv ** 2) / sbase_mva

    return pu * zbase_ohm





def b_pu_to_c_nf_per_km(b_pu, vn_kv=VN_KV, sbase_mva=SBASE, f_hz=F_HZ, length_km=1.0):

    zbase = (vn_kv ** 2) / sbase_mva

    b_si = b_pu / zbase

    c_total = b_si / (2 * np.pi * f_hz)

    return (c_total / length_km) * 1e9





def set_standard_line_limits(net):

    if "max_loading_percent" not in net.line.columns:

        net.line["max_loading_percent"] = 100.0

    net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(100.0)





# ============================================================

# 1) BUILD KUNDUR NETWORK FOR GENERATOR TRIP CASE

#    ALL 4 MACHINES AS "gen", slack assigned dynamically

# ============================================================

def build_kundur_net_trip():

    net = pp.create_empty_network(sn_mva=SBASE, f_hz=F_HZ)



    bus = {}

    for b in range(1, 12):

        bus[b] = pp.create_bus(net, vn_kv=VN_KV, name=f"Bus {b}")



    net.bus["min_vm_pu"] = 0.90

    net.bus["max_vm_pu"] = 1.10



    x_tr_100 = 0.15 * (SBASE / 900.0)

    r_tr_100 = 0.0

    z_tr = np.sqrt(r_tr_100**2 + x_tr_100**2)

    vk_percent = z_tr * 100.0

    vkr_percent = r_tr_100 * 100.0



    for (hv, lv) in [(1, 5), (2, 6), (3, 11), (4, 10)]:

        pp.create_transformer_from_parameters(

            net, hv_bus=bus[hv], lv_bus=bus[lv],

            sn_mva=900.0, vn_hv_kv=VN_KV, vn_lv_kv=VN_KV,

            vk_percent=vk_percent, vkr_percent=vkr_percent,

            pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,

            name=f"TR {hv}-{lv}"

        )



    line_specs = [

        (5, 6, 0.0025, 0.025, 0.04375, 1),

        (6, 7, 0.0010, 0.010, 0.01750, 1),

        (7, 8, 0.0110, 0.110, 0.19250, 2),

        (8, 9, 0.0110, 0.110, 0.19250, 2),

        (9, 10, 0.0010, 0.010, 0.01750, 1),

        (10, 11, 0.0025, 0.025, 0.04375, 1),

    ]



    corridor_line_ids = []

    for (fb, tb, rpu, xpu, bpu, npar) in line_specs:

        for k in range(npar):

            lid = pp.create_line_from_parameters(

                net,

                from_bus=bus[fb], to_bus=bus[tb], length_km=1.0,

                r_ohm_per_km=pu_to_ohm(rpu), x_ohm_per_km=pu_to_ohm(xpu),

                c_nf_per_km=b_pu_to_c_nf_per_km(bpu),

                max_i_ka=3.0, name=f"Line {fb}-{tb} #{k+1}"

            )

            if (fb, tb) in [(7, 8), (8, 9)]:

                corridor_line_ids.append(lid)



    pp.create_load(net, bus=bus[7], p_mw=967.0, q_mvar=100.0, name="Load7")

    pp.create_load(net, bus=bus[9], p_mw=1767.0, q_mvar=100.0, name="Load9")

    pp.create_shunt(net, bus=bus[7], p_mw=0.0, q_mvar=-200.0, name="QSH7")

    pp.create_shunt(net, bus=bus[9], p_mw=0.0, q_mvar=-350.0, name="QSH9")



    gen_specs = [

        (1, "G1", 1.03, 700.0),

        (2, "G2", 1.01, 700.0),

        (3, "G3", 1.03, 719.0),

        (4, "G4", 1.01, 700.0),

    ]

    for (b, name, vm, p0) in gen_specs:

        gid = pp.create_gen(net, bus=bus[b], p_mw=p0, vm_pu=vm, name=name, controllable=True)

        net.gen.loc[gid, ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]] = [0.0, 3000.0, -2000.0, 2000.0]



    net.gen["slack"] = False



    if len(net.poly_cost) == 0:

        for gid in net.gen.index:

            pp.create_poly_cost(net, element=gid, et="gen", cp1_eur_per_mw=GEN_CP1)



    set_standard_line_limits(net)

    return net, corridor_line_ids





# ============================================================

# 2) BUILD KUNDUR NETWORK FOR FDI CASE

#    G3 modeled as ext_grid for robust OPF / PF convergence

# ============================================================

def build_kundur_net_fdi():

    net = pp.create_empty_network(sn_mva=SBASE, f_hz=F_HZ)



    bus = {}

    for b in range(1, 12):

        bus[b] = pp.create_bus(net, vn_kv=VN_KV, name=f"Bus {b}")



    net.bus["min_vm_pu"] = 0.90

    net.bus["max_vm_pu"] = 1.10



    x_tr_100 = 0.15 * (SBASE / 900.0)

    r_tr_100 = 0.0

    z_tr = np.sqrt(r_tr_100**2 + x_tr_100**2)

    vk_percent = z_tr * 100.0

    vkr_percent = r_tr_100 * 100.0



    for (hv, lv) in [(1, 5), (2, 6), (3, 11), (4, 10)]:

        pp.create_transformer_from_parameters(

            net, hv_bus=bus[hv], lv_bus=bus[lv],

            sn_mva=900.0, vn_hv_kv=VN_KV, vn_lv_kv=VN_KV,

            vk_percent=vk_percent, vkr_percent=vkr_percent,

            pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,

            name=f"TR {hv}-{lv}"

        )



    line_specs = [

        (5, 6, 0.0025, 0.025, 0.04375, 1),

        (6, 7, 0.0010, 0.010, 0.01750, 1),

        (7, 8, 0.0110, 0.110, 0.19250, 2),

        (8, 9, 0.0110, 0.110, 0.19250, 2),

        (9, 10, 0.0010, 0.010, 0.01750, 1),

        (10, 11, 0.0025, 0.025, 0.04375, 1),

    ]



    corridor_line_ids = []

    for (fb, tb, rpu, xpu, bpu, npar) in line_specs:

        for k in range(npar):

            lid = pp.create_line_from_parameters(

                net,

                from_bus=bus[fb], to_bus=bus[tb], length_km=1.0,

                r_ohm_per_km=pu_to_ohm(rpu), x_ohm_per_km=pu_to_ohm(xpu),

                c_nf_per_km=b_pu_to_c_nf_per_km(bpu),

                max_i_ka=3.0, name=f"Line {fb}-{tb} #{k+1}"

            )

            if (fb, tb) in [(7, 8), (8, 9)]:

                corridor_line_ids.append(lid)



    pp.create_load(net, bus=bus[7], p_mw=967.0, q_mvar=100.0, name="Load7")

    pp.create_load(net, bus=bus[9], p_mw=1767.0, q_mvar=100.0, name="Load9")

    pp.create_shunt(net, bus=bus[7], p_mw=0.0, q_mvar=-200.0, name="QSH7")

    pp.create_shunt(net, bus=bus[9], p_mw=0.0, q_mvar=-350.0, name="QSH9")



    eg = pp.create_ext_grid(net, bus=bus[3], vm_pu=1.03, name="G3_ref")

    net.ext_grid.loc[eg, ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]] = [-3000.0, 3000.0, -2000.0, 2000.0]

    pp.create_poly_cost(net, element=eg, et="ext_grid", cp1_eur_per_mw=GEN_CP1)



    gen_specs = [

        (1, "G1", 1.03, 700.0),

        (2, "G2", 1.01, 700.0),

        (4, "G4", 1.01, 700.0),

    ]

    for (b, name, vm, p0) in gen_specs:

        gid = pp.create_gen(net, bus=bus[b], p_mw=p0, vm_pu=vm, name=name, controllable=True)

        net.gen.loc[gid, ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]] = [0.0, 3000.0, -2000.0, 2000.0]

        pp.create_poly_cost(net, element=gid, et="gen", cp1_eur_per_mw=GEN_CP1)



    set_standard_line_limits(net)

    return net, corridor_line_ids





# ============================================================

# 3) OPF CONFIGURATION

# ============================================================

def enable_load_shedding_opf(net, voll=VoLL):

    if "p_nom_mw" not in net.load.columns:

        net.load["p_nom_mw"] = net.load["p_mw"].values

    if "q_nom_mvar" not in net.load.columns:

        net.load["q_nom_mvar"] = net.load["q_mvar"].values



    net.load["controllable"] = True

    net.load["min_p_mw"] = 0.0

    net.load["max_p_mw"] = net.load["p_nom_mw"].values

    net.load["min_q_mvar"] = 0.0

    net.load["max_q_mvar"] = net.load["q_nom_mvar"].values



    existing = set()

    if len(net.poly_cost):

        existing = set(net.poly_cost.loc[net.poly_cost["et"] == "load", "element"].values)

    for ld in net.load.index:

        if ld not in existing:

            pp.create_poly_cost(net, element=ld, et="load", cp1_eur_per_mw=-voll)





def prepare_fixed_load_opf(net):

    net.load["controllable"] = False





def assign_single_slack_gen(net, preferred_name="G3"):

    net.gen["slack"] = False

    online = net.gen.index[net.gen["in_service"].fillna(True)].tolist()

    if len(online) == 0:

        raise RuntimeError("No online generators available.")



    preferred = None

    for gid in online:

        if str(net.gen.at[gid, "name"]) == preferred_name:

            preferred = gid

            break



    slack_gid = preferred if preferred is not None else online[0]

    net.gen.at[slack_gid, "slack"] = True

    return slack_gid





# ============================================================

# 4) CORE METRICS

# ============================================================

def run_opf_and_metrics(net, corridor_line_ids):

    try:

        if len(net.ext_grid) == 0:

            assign_single_slack_gen(net, preferred_name="G3")

        pp.runopp(net, calculate_voltage_angles=True, suppress_warnings=True)

    except Exception:

        return {

            "opf_ok": False,

            "cost_paper": BIG_PEN,

            "LS": BIG_PEN,

            "VV": BIG_PEN,

            "OL": BIG_PEN,

            "TF": BIG_PEN,

            "vmin": 0.0,

            "max_loading": BIG_PEN,

            "collapsed_proxy": True,

        }



    p_nom = float(net.load["p_nom_mw"].sum()) if "p_nom_mw" in net.load.columns else float(net.load["p_mw"].sum())

    p_serv = float(net.res_load["p_mw"].sum()) if len(net.load) else 0.0

    shed = max(0.0, p_nom - p_serv)



    gen_cost = 0.0

    if len(net.gen):

        gen_cost += float(net.res_gen["p_mw"].clip(lower=0.0).sum()) * GEN_CP1

    if len(net.ext_grid):

        gen_cost += float(net.res_ext_grid["p_mw"].clip(lower=0.0).sum()) * GEN_CP1



    cost_paper = gen_cost + VoLL * shed



    v = net.res_bus["vm_pu"].values

    vmin = float(np.min(v))

    vlims_min = net.bus["min_vm_pu"].values

    vlims_max = net.bus["max_vm_pu"].values

    VV = float(np.maximum(0.0, vlims_min - v).sum() + np.maximum(0.0, v - vlims_max).sum())



    loading = net.res_line["loading_percent"].values if len(net.line) else np.array([0.0])

    max_loading = float(np.max(loading)) if len(loading) else 0.0

    lim = net.line["max_loading_percent"].values

    OL = float(np.maximum(0.0, loading - lim).sum()) if len(net.line) else 0.0



    TF = float(net.res_line.loc[corridor_line_ids, "p_from_mw"].abs().sum()) if corridor_line_ids else 0.0

    collapsed_proxy = (vmin < VMIN_COLLAPSE)



    return {

        "opf_ok": True,

        "cost_paper": float(cost_paper),

        "LS": float(shed),

        "VV": float(VV),

        "OL": float(OL),

        "TF": float(TF),

        "vmin": vmin,

        "max_loading": max_loading,

        "collapsed_proxy": collapsed_proxy,

    }





def run_opf_fixed_loads(net, corridor_line_ids):

    try:

        pp.runopp(net, calculate_voltage_angles=True, suppress_warnings=True)

    except Exception:

        return {

            "opf_ok": False,

            "ems_cost": BIG_PEN,

            "VV": BIG_PEN,

            "OL": BIG_PEN,

            "TF": BIG_PEN,

            "vmin": 0.0,

            "max_loading": BIG_PEN,

            "collapsed_proxy": True,

        }



    v = net.res_bus["vm_pu"].values

    vmin = float(np.min(v))

    vlims_min = net.bus["min_vm_pu"].values

    vlims_max = net.bus["max_vm_pu"].values

    VV = float(np.maximum(0.0, vlims_min - v).sum() + np.maximum(0.0, v - vlims_max).sum())



    loading = net.res_line["loading_percent"].values if len(net.line) else np.array([0.0])

    max_loading = float(np.max(loading)) if len(loading) else 0.0

    lim = net.line["max_loading_percent"].values

    OL = float(np.maximum(0.0, loading - lim).sum()) if len(net.line) else 0.0



    TF = float(net.res_line.loc[corridor_line_ids, "p_from_mw"].abs().sum()) if corridor_line_ids else 0.0

    collapsed_proxy = (vmin < VMIN_COLLAPSE)



    ems_cost = 0.0

    if len(net.gen):

        ems_cost += float(net.res_gen["p_mw"].clip(lower=0.0).sum()) * GEN_CP1

    if len(net.ext_grid):

        ems_cost += float(net.res_ext_grid["p_mw"].clip(lower=0.0).sum()) * GEN_CP1



    return {

        "opf_ok": True,

        "ems_cost": float(ems_cost),

        "VV": float(VV),

        "OL": float(OL),

        "TF": float(TF),

        "vmin": vmin,

        "max_loading": max_loading,

        "collapsed_proxy": collapsed_proxy,

    }





def run_pf_truth_metrics(net, corridor_line_ids):

    try:

        pp.runpp(net, calculate_voltage_angles=True, init="flat")

    except Exception:

        return {

            "pf_ok": False,

            "mismatch_mw": BIG_PEN,

            "VV": BIG_PEN,

            "OL": BIG_PEN,

            "TF": BIG_PEN,

            "vmin": 0.0,

            "max_loading": BIG_PEN,

            "collapsed_proxy": True,

        }



    total_gen = 0.0

    if len(net.gen):

        total_gen += float(net.res_gen["p_mw"].sum())

    if len(net.ext_grid):

        total_gen += float(net.res_ext_grid["p_mw"].sum())

    total_load = float(net.res_load["p_mw"].sum()) if len(net.load) else 0.0

    mismatch_mw = abs(total_gen - total_load)



    v = net.res_bus["vm_pu"].values

    vmin = float(np.min(v))

    vlims_min = net.bus["min_vm_pu"].values

    vlims_max = net.bus["max_vm_pu"].values

    VV = float(np.maximum(0.0, vlims_min - v).sum() + np.maximum(0.0, v - vlims_max).sum())



    loading = net.res_line["loading_percent"].values if len(net.line) else np.array([0.0])

    max_loading = float(np.max(loading)) if len(loading) else 0.0

    lim = net.line["max_loading_percent"].values

    OL = float(np.maximum(0.0, loading - lim).sum()) if len(net.line) else 0.0



    TF = float(net.res_line.loc[corridor_line_ids, "p_from_mw"].abs().sum()) if corridor_line_ids else 0.0

    collapsed_proxy = (vmin < VMIN_COLLAPSE)



    return {

        "pf_ok": True,

        "mismatch_mw": float(mismatch_mw),

        "VV": float(VV),

        "OL": float(OL),

        "TF": float(TF),

        "vmin": vmin,

        "max_loading": max_loading,

        "collapsed_proxy": collapsed_proxy,

    }





# ============================================================

# 5) GENERATOR TRIP CASE

# ============================================================

def get_inertia_data(net):

    S_rated = 900.0

    H = {}

    S = {}

    for gid in net.gen.index:

        nm = str(net.gen.at[gid, "name"])

        H[gid] = 6.5 if nm in ["G1", "G2"] else 6.175

        S[gid] = S_rated

    return H, S





def system_inertia(H, S, online_gids):

    denom = sum(S[g] for g in online_gids) + 1e-12

    num = sum(H[g] * S[g] for g in online_gids)

    return num / denom





def trip_gen(net, gid):

    net.gen.at[gid, "in_service"] = False





def compute_criticality_ranking(net_base, corridor_line_ids):

    H, S = get_inertia_data(net_base)

    base_online = net_base.gen.index[net_base.gen["in_service"].fillna(True)].tolist()

    Hsys_base = system_inertia(H, S, base_online)



    rows = []

    for gid in net_base.gen.index:

        net = copy.deepcopy(net_base)

        enable_load_shedding_opf(net, VoLL)

        if bool(net.gen.at[gid, "in_service"]):

            trip_gen(net, gid)



        m = run_opf_and_metrics(net, corridor_line_ids)

        online_minus = net.gen.index[net.gen["in_service"].fillna(True)].tolist()

        Hsys_minus = system_inertia(H, S, online_minus) if len(online_minus) else 0.0

        IN = float(1.0 / (Hsys_minus + 1e-12))



        rows.append({

            "gen_id": gid,

            "name": str(net_base.gen.at[gid, "name"]),

            "cost_paper": m["cost_paper"],

            "LS": m["LS"],

            "VV": m["VV"],

            "OL": m["OL"],

            "TF": m["TF"],

            "IN": IN,

            "opf_ok": m["opf_ok"],

            "vmin": m["vmin"],

            "max_loading": m["max_loading"],

            "collapsed_proxy": m["collapsed_proxy"],

            "Hsys_base": Hsys_base,

            "Hsys_minus": Hsys_minus,

        })



    df = pd.DataFrame(rows)

    eps = 1e-9

    for col in ["LS", "VV", "OL", "TF", "IN"]:

        x = df[col].astype(float).values

        xmin, xmax = float(np.min(x)), float(np.max(x))

        df[f"n_{col}"] = (df[col] - xmin) / (xmax - xmin + eps)



    df["CI"] = (

        W["LS"] * df["n_LS"] +

        W["VV"] * df["n_VV"] +

        W["OL"] * df["n_OL"] +

        W["TF"] * df["n_TF"] +

        W["IN"] * df["n_IN"]

    )

    return df.sort_values("CI", ascending=False).reset_index(drop=True)





def generate_defender_actions_gen(gen_ids, max_protect=TRIP_MAX_PROTECT):

    actions = [tuple()]

    for k in range(1, max_protect + 1):

        actions.extend(list(itertools.combinations(gen_ids, k)))

    return actions





def evaluate_trip_given_defense(net_base, corridor_line_ids, protected_set, attacked_gid):

    net = copy.deepcopy(net_base)

    enable_load_shedding_opf(net, VoLL)



    blocked = attacked_gid in protected_set

    if not blocked:

        trip_gen(net, attacked_gid)



    metrics = run_opf_and_metrics(net, corridor_line_ids)

    return blocked, metrics, net





def solve_stackelberg_kundur_trip(net_base, corridor_line_ids, protect_cost=TRIP_PROTECT_COST, max_protect=TRIP_MAX_PROTECT):

    gen_ids = list(net_base.gen.index)

    defender_actions = generate_defender_actions_gen(gen_ids, max_protect=max_protect)



    rows = []

    best_def_action = None

    best_att_gid = None

    best_metrics = None

    best_net = None

    best_obj = 1e18

    best_blocked = False



    for def_action in defender_actions:

        protected = set(def_action)



        worst_loss = -1.0

        worst_gid = None

        worst_metrics = None

        worst_net = None

        worst_blocked = None



        for gid in gen_ids:

            blocked, metrics, scenario_net = evaluate_trip_given_defense(net_base, corridor_line_ids, protected, gid)

            attack_loss = float(metrics["cost_paper"])

            if attack_loss > worst_loss:

                worst_loss = attack_loss

                worst_gid = gid

                worst_metrics = metrics

                worst_net = scenario_net

                worst_blocked = blocked



        defender_cost = protect_cost * len(protected)

        defender_obj = defender_cost + worst_loss



        rows.append({

            "protected_generators": "None" if len(def_action) == 0 else ",".join([net_base.gen.at[g, "name"] for g in def_action]),

            "attacker_best_response": net_base.gen.at[worst_gid, "name"],

            "attack_blocked": worst_blocked,

            "worst_loss": worst_loss,

            "defender_cost": defender_cost,

            "defender_objective": defender_obj,

            "LS": worst_metrics["LS"],

            "vmin": worst_metrics["vmin"],

            "max_loading": worst_metrics["max_loading"],

        })



        if defender_obj < best_obj:

            best_obj = defender_obj

            best_def_action = def_action

            best_att_gid = worst_gid

            best_metrics = worst_metrics

            best_net = worst_net

            best_blocked = worst_blocked



    df_game = pd.DataFrame(rows).sort_values("defender_objective").reset_index(drop=True)

    return {

        "df_game": df_game,

        "best_def_action": best_def_action,

        "best_att_gid": best_att_gid,

        "best_metrics": best_metrics,

        "best_net": best_net,

        "best_obj": best_obj,

        "best_blocked": best_blocked,

    }





# ============================================================

# 6) FDI CASE

# ============================================================

def get_load_index_by_bus(net, bus_number):

    pp_bus_idx = bus_number - 1

    candidates = net.load.index[net.load["bus"] == pp_bus_idx].tolist()

    if len(candidates) == 0:

        raise ValueError(f"No load found at bus {bus_number}")

    return candidates[0]





def apply_fdi_load_shift(net, attacked_bus_number, delta_p_mw, maintain_pf=True):

    ld = get_load_index_by_bus(net, attacked_bus_number)

    p0 = float(net.load.at[ld, "p_mw"])

    q0 = float(net.load.at[ld, "q_mvar"])



    net.load.at[ld, "p_mw"] = p0 + delta_p_mw

    if maintain_pf and abs(p0) > 1e-9:

        net.load.at[ld, "q_mvar"] = q0 * (p0 + delta_p_mw) / p0

    return ld





def apply_ems_dispatch_to_truth_net(net_truth, net_ems):

    for gid in net_truth.gen.index:

        net_truth.gen.at[gid, "p_mw"] = float(net_ems.res_gen.at[gid, "p_mw"])

        net_truth.gen.at[gid, "vm_pu"] = float(net_ems.gen.at[gid, "vm_pu"])



    for eg in net_truth.ext_grid.index:

        net_truth.ext_grid.at[eg, "vm_pu"] = float(net_ems.ext_grid.at[eg, "vm_pu"])





def get_dispatch_vector_from_results(net):

    vals = []

    if len(net.gen):

        vals.extend(list(net.res_gen["p_mw"].values))

    if len(net.ext_grid):

        vals.extend(list(net.res_ext_grid["p_mw"].values))

    return np.array(vals, dtype=float)





def get_base_dispatch_vector(net):

    vals = []

    if len(net.gen):

        vals.extend(list(net.gen["p_mw"].values))

    if len(net.ext_grid):

        vals.extend([0.0] * len(net.ext_grid))

    return np.array(vals, dtype=float)





def run_fdi_case_with_net(net_base, corridor_line_ids, attacked_bus_number, delta_p_mw):

    net_ems = copy.deepcopy(net_base)

    prepare_fixed_load_opf(net_ems)

    apply_fdi_load_shift(net_ems, attacked_bus_number, delta_p_mw, maintain_pf=True)



    ems_metrics = run_opf_fixed_loads(net_ems, corridor_line_ids)

    if not ems_metrics["opf_ok"]:

        metrics = {

            "bus": attacked_bus_number,

            "area": FDI_ATTACK_BUSES.get(attacked_bus_number, "Unknown"),

            "delta_p_mw": delta_p_mw,

            "ems_cost": BIG_PEN,

            "dispatch_dev": BIG_PEN,

            "mismatch_mw": BIG_PEN,

            "VV": BIG_PEN,

            "OL": BIG_PEN,

            "TF": BIG_PEN,

            "opf_ok": False,

            "pf_ok": False,

            "vmin": 0.0,

            "max_loading": BIG_PEN,

            "collapsed_proxy": True,

        }

        return metrics, net_ems, None



    net_truth = copy.deepcopy(net_base)

    apply_ems_dispatch_to_truth_net(net_truth, net_ems)

    truth_metrics = run_pf_truth_metrics(net_truth, corridor_line_ids)



    base_dispatch = get_base_dispatch_vector(net_base)

    ems_dispatch = get_dispatch_vector_from_results(net_ems)

    n = min(len(base_dispatch), len(ems_dispatch))

    dispatch_dev = float(np.abs(ems_dispatch[:n] - base_dispatch[:n]).sum())



    metrics = {

        "bus": attacked_bus_number,

        "area": FDI_ATTACK_BUSES.get(attacked_bus_number, "Unknown"),

        "delta_p_mw": delta_p_mw,

        "ems_cost": ems_metrics["ems_cost"],

        "dispatch_dev": dispatch_dev,

        "mismatch_mw": truth_metrics["mismatch_mw"],

        "VV": truth_metrics["VV"],

        "OL": truth_metrics["OL"],

        "TF": truth_metrics["TF"],

        "opf_ok": True,

        "pf_ok": truth_metrics["pf_ok"],

        "vmin": truth_metrics["vmin"],

        "max_loading": truth_metrics["max_loading"],

        "collapsed_proxy": truth_metrics["collapsed_proxy"],

    }

    return metrics, net_ems, net_truth





def run_fdi_sweep(net_base, corridor_line_ids, fdi_levels):

    rows = []

    for attacked_bus in FDI_ATTACK_BUSES.keys():

        for dP in fdi_levels:

            metrics, _, _ = run_fdi_case_with_net(net_base, corridor_line_ids, attacked_bus, dP)

            rows.append(metrics)

    return pd.DataFrame(rows)





def compute_graph_scores(detail_df):

    rows = []

    for bus in sorted(detail_df["bus"].unique()):

        dfb = detail_df[

            (detail_df["bus"] == bus) &

            (detail_df["opf_ok"]) &

            (detail_df["pf_ok"])

        ].sort_values("delta_p_mw")



        if len(dfb) < 2:

            rows.append({

                "bus": int(bus),

                "area": FDI_ATTACK_BUSES.get(bus, "Unknown"),

                "score_tieline": BIG_PEN,

                "score_dispatch_dev": BIG_PEN,

                "score_max_loading": BIG_PEN,

                "score_voltage_drop": BIG_PEN,

            })

            continue



        x = dfb["delta_p_mw"].values

        rows.append({

            "bus": int(bus),

            "area": dfb["area"].iloc[0],

            "score_tieline": float(np.trapz(dfb["TF"].values, x)),

            "score_dispatch_dev": float(np.trapz(dfb["dispatch_dev"].values, x)),

            "score_max_loading": float(np.trapz(dfb["max_loading"].values, x)),

            "score_voltage_drop": float(np.trapz(1.1 - dfb["vmin"].values, x)),

        })

    return pd.DataFrame(rows).sort_values(

        ["score_tieline", "score_dispatch_dev", "score_max_loading", "score_voltage_drop"],

        ascending=False

    ).reset_index(drop=True)





def compute_fdi_total_impact(metrics):

    if (not metrics["opf_ok"]) or (not metrics["pf_ok"]):

        return BIG_PEN



    v_pen = max(0.0, 0.95 - metrics["vmin"]) * 1e5

    total = (

        metrics["ems_cost"]

        + 20.0 * metrics["dispatch_dev"]

        + 10.0 * metrics["TF"]

        + 5.0 * metrics["max_loading"]

        + v_pen

    )

    return float(total)





def generate_fdi_defender_actions():

    actions = [tuple()]

    for b in sorted(FDI_ATTACK_BUSES.keys()):

        actions.append((b,))

    return actions





def evaluate_fdi_attack_given_defense(net_base, corridor_line_ids, protected_buses, attacked_bus, delta_p_mw):

    effective_delta = delta_p_mw

    if attacked_bus in protected_buses:

        effective_delta = (1.0 - FDI_PROTECT_EFFECT) * delta_p_mw



    metrics, net_ems, net_truth = run_fdi_case_with_net(

        net_base=net_base,

        corridor_line_ids=corridor_line_ids,

        attacked_bus_number=attacked_bus,

        delta_p_mw=effective_delta

    )



    total_impact = compute_fdi_total_impact(metrics)

    return {

        "attacked_bus": attacked_bus,

        "requested_delta_p_mw": delta_p_mw,

        "effective_delta_p_mw": effective_delta,

        "impact": total_impact,

        "metrics": metrics,

        "net_ems": net_ems,

        "net_truth": net_truth,

    }





def solve_stackelberg_kundur_fdi(net_base, corridor_line_ids):

    defender_actions = generate_fdi_defender_actions()



    rows = []

    best_def_action = None

    best_attack_bus = None

    best_attack_level = None

    best_result = None

    best_obj = 1e18



    for def_action in defender_actions:

        protected = set(def_action)

        worst_impact = -1.0

        worst_result = None



        for attacked_bus in sorted(FDI_ATTACK_BUSES.keys()):

            for dP in FDI_LEVELS[1:]:

                result = evaluate_fdi_attack_given_defense(

                    net_base, corridor_line_ids, protected, attacked_bus, float(dP)

                )

                if result["impact"] > worst_impact:

                    worst_impact = result["impact"]

                    worst_result = result



        defender_cost = FDI_PROTECT_COST * len(protected)

        defender_obj = defender_cost + worst_impact



        rows.append({

            "protected_buses": "None" if len(protected) == 0 else ",".join([f"Bus {b}" for b in sorted(protected)]),

            "attacker_best_bus": f"Bus {worst_result['attacked_bus']}",

            "requested_delta_p_mw": worst_result["requested_delta_p_mw"],

            "effective_delta_p_mw": worst_result["effective_delta_p_mw"],

            "worst_impact": worst_impact,

            "defender_cost": defender_cost,

            "defender_objective": defender_obj,

            "TF": worst_result["metrics"]["TF"],

            "vmin": worst_result["metrics"]["vmin"],

            "dispatch_dev": worst_result["metrics"]["dispatch_dev"],

            "max_loading": worst_result["metrics"]["max_loading"],

        })



        if defender_obj < best_obj:

            best_obj = defender_obj

            best_def_action = def_action

            best_attack_bus = worst_result["attacked_bus"]

            best_attack_level = worst_result["requested_delta_p_mw"]

            best_result = worst_result



    df_game = pd.DataFrame(rows).sort_values("defender_objective").reset_index(drop=True)

    return {

        "df_game": df_game,

        "best_def_action": best_def_action,

        "best_attack_bus": best_attack_bus,

        "best_attack_level": best_attack_level,

        "best_result": best_result,

        "best_obj": best_obj,

    }





# ============================================================

# 7) TABLES

# ============================================================

def load_shedding_by_bus(net):

    if net is None or not hasattr(net, "res_load") or len(net.res_load) == 0:

        return pd.DataFrame(columns=["bus_1based", "p_nom_mw", "p_served_mw", "shed_mw"])



    df = net.load[["bus", "p_nom_mw"]].copy()

    df["p_served_mw"] = net.res_load["p_mw"].values

    df["shed_mw"] = np.maximum(0.0, df["p_nom_mw"] - df["p_served_mw"])

    df["bus_1based"] = df["bus"] + 1

    return df.groupby("bus_1based")[["p_nom_mw", "p_served_mw", "shed_mw"]].sum().reset_index()





# ============================================================

# 8) PLOTS

# ============================================================

def plot_trip_case_study_results(ranking_df, df_game, net_nom, net_eq, equilibrium_trip_name):

    plt.figure(figsize=(8, 5))

    plt.bar(ranking_df["name"], ranking_df["CI"])

    plt.xlabel("Generator")

    plt.ylabel("Criticality Index (CI)")

    plt.title("Generator Criticality Index Under N-1 Tripping Scenarios")

    plt.grid(True, axis="y")

    plt.tight_layout()



    x = np.arange(len(df_game))

    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(x - width/2, df_game["worst_loss"], width, label=r"Worst-case impact $**\P**hi(y,a^*)$")

    plt.bar(x + width/2, df_game["defender_objective"], width, label=r"Defender objective $C(y)+**\P**hi(y,a^*)$")

    plt.xticks(x, df_game["protected_generators"], rotation=25)

    plt.xlabel("Defender protection strategy")

    plt.ylabel("Cost (€)")

    plt.title("Stackelberg Malicious Trip Game on the Kundur Two-Area System")

    plt.legend()

    plt.grid(True, axis="y")

    plt.tight_layout()



    if net_eq is not None and hasattr(net_eq, "res_bus") and len(net_eq.res_bus) > 0:

        V_nom = net_nom.res_bus.vm_pu.values

        V_eq = net_eq.res_bus.vm_pu.values

        buses = np.arange(1, len(V_nom) + 1)

        plt.figure(figsize=(10, 5))

        plt.bar(buses - width/2, V_nom, width, label=r"Nominal voltage $V_i^{nom}$")

        plt.bar(buses + width/2, V_eq, width, label=rf"Equilibrium trip voltage $V_i^{{eq}}$ ({equilibrium_trip_name})")

        plt.xlabel(r"Bus Index $i$")

        plt.ylabel("Voltage Magnitude (p.u.)")

        plt.title("Bus Voltage Profile Under Nominal Operation and Stackelberg Equilibrium Trip Scenario")

        plt.legend()

        plt.grid(True, axis="y")

        plt.tight_layout()





def plot_fdi_stackelberg_case(score_df, df_game, net_nom, net_eq_truth, best_attack_bus, best_attack_level):

    score_df_plot = score_df.copy()

    score_df_plot["combined_score"] = (

        score_df_plot["score_tieline"]

        + score_df_plot["score_dispatch_dev"]

        + score_df_plot["score_max_loading"]

        + score_df_plot["score_voltage_drop"]

    )



    plt.figure(figsize=(8, 5))

    plt.bar(score_df_plot["bus"].astype(str), score_df_plot["combined_score"])

    plt.xlabel("Attacked Bus")

    plt.ylabel("Combined Graph-Based Criticality Score")

    plt.title("Graph-Based Criticality Ranking for False Data Injection Buses")

    plt.grid(True, axis="y")

    plt.tight_layout()



    x = np.arange(len(df_game))

    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(x - width/2, df_game["worst_impact"], width, label=r"Worst-case impact $**\P**hi(y,a^*)$")

    plt.bar(x + width/2, df_game["defender_objective"], width, label=r"Defender objective $C(y)+**\P**hi(y,a^*)$")

    plt.xticks(x, df_game["protected_buses"], rotation=20)

    plt.xlabel("Defender protection strategy")

    plt.ylabel("Cost / Impact")

    plt.title("Stackelberg False Data Injection Game on the Kundur Two-Area System")

    plt.legend()

    plt.grid(True, axis="y")

    plt.tight_layout()



    if net_eq_truth is not None and hasattr(net_eq_truth, "res_bus") and len(net_eq_truth.res_bus) > 0:

        V_nom = net_nom.res_bus.vm_pu.values

        V_eq = net_eq_truth.res_bus.vm_pu.values

        buses = np.arange(1, len(V_nom) + 1)

        plt.figure(figsize=(10, 5))

        plt.bar(buses - width/2, V_nom, width, label=r"Nominal voltage $V_i^{nom}$")

        plt.bar(

            buses + width/2, V_eq, width,

            label=rf"Equilibrium FDI voltage $V_i^{{eq}}$ (Bus {best_attack_bus}, {best_attack_level:.0f} MW)"

        )

        plt.xlabel(r"Bus Index $i$")

        plt.ylabel("Voltage Magnitude (p.u.)")

        plt.title("Bus Voltage Profile Under Nominal Operation and Stackelberg Equilibrium FDI Scenario")

        plt.legend()

        plt.grid(True, axis="y")

        plt.tight_layout()





# ============================================================

# MAIN

# ============================================================

def main():

    pd.set_option("display.width", 220)

    pd.set_option("display.max_columns", 60)



    # ---------------- BASE NETWORK FOR TRIP CASE ----------------

    net_trip, corridor_trip = build_kundur_net_trip()

    enable_load_shedding_opf(net_trip, VoLL)



    net_trip_nom = copy.deepcopy(net_trip)

    base_trip_metrics = run_opf_and_metrics(net_trip_nom, corridor_trip)

    print("\n========== BASE CASE: KUNDUR TRIP MODEL ==========")

    print(base_trip_metrics)



    # ---------------- CASE STUDY I: MALICIOUS TRIP ----------------

    ranking = compute_criticality_ranking(net_trip, corridor_trip)

    print("\n========== CASE STUDY I: GENERATOR CRITICALITY RANKING ==========")

    print(ranking[

        ["name", "CI", "cost_paper", "LS", "VV", "OL", "TF", "IN", "opf_ok", "vmin", "max_loading"]

    ].to_string(index=False))



    trip_game_res = solve_stackelberg_kundur_trip(

        net_base=net_trip,

        corridor_line_ids=corridor_trip,

        protect_cost=TRIP_PROTECT_COST,

        max_protect=TRIP_MAX_PROTECT

    )



    df_trip_game = trip_game_res["df_game"]

    trip_protected_names = [net_trip.gen.at[g, "name"] for g in trip_game_res["best_def_action"]] if len(trip_game_res["best_def_action"]) else ["None"]

    trip_attacked_name = net_trip.gen.at[trip_game_res["best_att_gid"], "name"]



    print("\n========== STACKELBERG MALICIOUS TRIP GAME RESULTS ==========")

    print(df_trip_game.to_string(index=False))



    print("\n========== MALICIOUS TRIP EQUILIBRIUM ==========")

    print("Defender protected:", trip_protected_names)

    print("Attacker tripped  :", trip_attacked_name)

    print("Attack blocked    :", trip_game_res["best_blocked"])

    print("Equilibrium metrics:", trip_game_res["best_metrics"])

    print("Defender objective:", trip_game_res["best_obj"])



    if trip_game_res["best_metrics"]["opf_ok"]:

        shed_eq_trip = load_shedding_by_bus(trip_game_res["best_net"])

        shed_eq_trip = shed_eq_trip[shed_eq_trip["shed_mw"] > 1e-6]

        print("\n========== MALICIOUS TRIP EQUILIBRIUM LOAD SHEDDING ==========")

        if len(shed_eq_trip) == 0:

            print("No load shedding in malicious trip equilibrium scenario.")

        else:

            print(shed_eq_trip.to_string(index=False))



    # ---------------- BASE NETWORK FOR FDI CASE ----------------

    net_fdi, corridor_fdi = build_kundur_net_fdi()



    net_fdi_nom = copy.deepcopy(net_fdi)

    pp.runpp(net_fdi_nom, calculate_voltage_angles=True, init="flat")



    # ---------------- CASE STUDY II: FDI GRAPH + GAME ----------------

    detail_df = run_fdi_sweep(net_fdi, corridor_fdi, FDI_LEVELS)

    print("\n========== CASE STUDY II: FDI SWEEP DATA ==========")

    print(detail_df[

        ["bus", "area", "delta_p_mw", "ems_cost", "dispatch_dev", "mismatch_mw", "VV", "OL", "TF", "opf_ok", "pf_ok", "vmin", "max_loading"]

    ].to_string(index=False))



    score_df = compute_graph_scores(detail_df)

    print("\n========== CASE STUDY II: GRAPH-BASED FDI CRITICALITY ==========")

    print(score_df.to_string(index=False))



    fdi_game_res = solve_stackelberg_kundur_fdi(net_fdi, corridor_fdi)

    df_fdi_game = fdi_game_res["df_game"]

    best_fdi_result = fdi_game_res["best_result"]



    fdi_protected_names = (

        "None" if len(fdi_game_res["best_def_action"]) == 0

        else ",".join([f"Bus {b}" for b in fdi_game_res["best_def_action"]])

    )



    print("\n========== STACKELBERG FDI GAME RESULTS ==========")

    print(df_fdi_game.to_string(index=False))



    print("\n========== FDI EQUILIBRIUM ==========")

    print("Defender protected :", fdi_protected_names)

    print("Attacker best bus  :", f"Bus {fdi_game_res['best_attack_bus']}")

    print("Requested attack   :", f"{fdi_game_res['best_attack_level']:.2f} MW")

    print("Effective attack   :", f"{best_fdi_result['effective_delta_p_mw']:.2f} MW")

    print("Equilibrium impact :", f"{best_fdi_result['impact']:.2f}")

    print("Defender objective :", f"{fdi_game_res['best_obj']:.2f}")

    print("Equilibrium metrics:", best_fdi_result["metrics"])



    # ---------------- PLOTS ----------------

    plot_trip_case_study_results(

        ranking_df=ranking,

        df_game=df_trip_game,

        net_nom=net_trip_nom,

        net_eq=trip_game_res["best_net"],

        equilibrium_trip_name=trip_attacked_name,

    )



    plot_fdi_stackelberg_case(

        score_df=score_df,

        df_game=df_fdi_game,

        net_nom=net_fdi_nom,

        net_eq_truth=best_fdi_result["net_truth"],

        best_attack_bus=fdi_game_res["best_attack_bus"],

        best_attack_level=fdi_game_res["best_attack_level"],

    )



    plt.show()





if __name__ == "__main__":

    main()
