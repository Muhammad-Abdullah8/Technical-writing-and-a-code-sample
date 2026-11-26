# Lead project author and repository maintainer: Muhammad Abdullah Anwar
"""Generate supervised AC-OPF examples for IEEE case9."""
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import random
import argparse
from pathlib import Path

def build_ieee9_with_quadratic_costs():
    # 1) Load IEEE 9-bus test case
    net = pn.case9()

    # 2) Remove any existing cost definitions to avoid duplicates
    if hasattr(net, "poly_cost"):
        net.poly_cost = net.poly_cost.iloc[0:0]

    # 3) Make generators and ext_grid controllable for OPF
    if "controllable" not in net.gen.columns:
        net.gen["controllable"] = True
    else:
        net.gen["controllable"] = True

    if "controllable" not in net.ext_grid.columns:
        net.ext_grid["controllable"] = True
    else:
        net.ext_grid["controllable"] = True

    # 4) Ensure min/max bounds exist for generators
    if "min_p_mw" not in net.gen.columns:
        net.gen["min_p_mw"] = 0.0
    if "max_p_mw" not in net.gen.columns:
        net.gen["max_p_mw"] = net.gen["p_mw"] * 1.5

    # Bounds for slack/ext_grid
    if "min_p_mw" not in net.ext_grid.columns:
        net.ext_grid["min_p_mw"] = -9999.0
    if "max_p_mw" not in net.ext_grid.columns:
        net.ext_grid["max_p_mw"] = 9999.0

    # 5) QUADRATIC COSTS
    # ---- Generator at bus 2 (index 0 in net.gen) ----
    pp.create_poly_cost(
        net,
        element=0,
        et="gen",
        cp2_eur_per_mw2=0.085,   # c2
        cp1_eur_per_mw=1.2,      # c1
        cp0_eur=600.0            # c0
    )

    # ---- Generator at bus 3 (index 1 in net.gen) ----
    pp.create_poly_cost(
        net,
        element=1,
        et="gen",
        cp2_eur_per_mw2=0.1225,
        cp1_eur_per_mw=1.0,
        cp0_eur=335.0
    )

    # ---- Slack / ext_grid ----
    pp.create_poly_cost(
        net,
        element=0,
        et="ext_grid",
        cp2_eur_per_mw2=0.11,
        cp1_eur_per_mw=5.0,
        cp0_eur=150.0
    )

    return net

def generate(output_folder, num_datasets=1, samples_per_dataset=500, variation_range=0.2, seed=0):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    # Use one reference network to define sizes and column names
    net_ref = build_ieee9_with_quadratic_costs()
    bus_indices = sorted(list(net_ref.bus.index.values))
    n_buses = len(bus_indices)
    n_gens = len(net_ref.gen)
    n_ext = len(net_ref.ext_grid)   # usually 1 for IEEE9

    # Build column names
    columns = []

    # Per-bus data: loads and voltages
    for b in range(1, n_buses + 1):
        columns.extend([
            f"P_D_bus{b}",
            f"Q_D_bus{b}",
            f"V_bus{b}",
            f"d_bus{b}_deg",
        ])

    # Generator outputs (P and Q)
    for g in range(1, n_gens + 1):
        columns.append(f"P_G_gen{g}")
    for g in range(1, n_gens + 1):
        columns.append(f"Q_G_gen{g}")

    # Ext_grid (slack) outputs
    for eg in range(1, n_ext + 1):
        columns.append(f"P_G_ext{eg}")
    for eg in range(1, n_ext + 1):
        columns.append(f"Q_G_ext{eg}")

    # Totals, losses, cost
    columns.extend([
        "P_D_total",
        "Q_D_total",
        "P_G_total",
        "Q_G_total",
        "P_loss",
        "Q_loss",
        "Total_Cost_EUR",
    ])

    # Helper: map bus index → position 0..n_buses-1
    bus_index_to_pos = {idx: pos for pos, idx in enumerate(bus_indices)}

    # ===============================
    # 3. Generate datasets
    # ===============================

    for dataset_n in range(1, num_datasets + 1):
        print(f"Generating IEEE 9-bus OPF dataset {dataset_n}...")

        data = []

        for sample in range(1, samples_per_dataset + 1):

            # -------------------------------
            # 3.1 Build network and randomize loads
            # -------------------------------
            net = build_ieee9_with_quadratic_costs()

            # Store base loads
            base_p = net.load["p_mw"].copy()
            base_q = net.load["q_mvar"].copy()

            # Randomly vary all loads
            for i in net.load.index:
                factor = 1.0 + random.uniform(-variation_range, variation_range)
                net.load.at[i, "p_mw"] = base_p[i] * factor
                net.load.at[i, "q_mvar"] = base_q[i] * factor


            # -------------------------------
            # 3.2 Solve OPF
            # -------------------------------
            try:
                pp.runopp(net, verbose=False, calculate_voltage_angles=True)
            except Exception as e:
                print(f"OPF error at sample {sample} in dataset {dataset_n}, skipping. Error: {e}")
                continue

            if not net["OPF_converged"]:
                print(f"OPF did not converge at sample {sample} in dataset {dataset_n}, skipping.")
                continue

            # -------------------------------
            # 3.3 Collect per-bus loads
            # -------------------------------
            P_D = np.zeros(n_buses)
            Q_D = np.zeros(n_buses)

            for _, load_row in net.load.iterrows():
                bus_idx = load_row["bus"]
                pos = bus_index_to_pos[bus_idx]
                P_D[pos] += load_row["p_mw"]
                Q_D[pos] += load_row["q_mvar"]

            P_D_total = P_D.sum()
            Q_D_total = Q_D.sum()

            # -------------------------------
            # 3.4 Collect bus voltages
            # -------------------------------
            V_mag = np.zeros(n_buses)
            V_ang = np.zeros(n_buses)

            for bus_idx in bus_indices:
                pos = bus_index_to_pos[bus_idx]
                V_mag[pos] = net.res_bus.at[bus_idx, "vm_pu"]
                V_ang[pos] = net.res_bus.at[bus_idx, "va_degree"]

            # -------------------------------
            # 3.5 Generator and slack outputs
            # -------------------------------
            # Generators
            P_G_gens = net.res_gen["p_mw"].values  # length n_gens
            Q_G_gens = net.res_gen["q_mvar"].values

            # Ext_grid (slack)
            P_G_ext = net.res_ext_grid["p_mw"].values  # length n_ext
            Q_G_ext = net.res_ext_grid["q_mvar"].values

            P_G_total = P_G_gens.sum() + P_G_ext.sum()
            Q_G_total = Q_G_gens.sum() + Q_G_ext.sum()

            # -------------------------------
            # 3.6 Losses
            # -------------------------------
            P_loss = P_G_total - P_D_total
            Q_loss = Q_G_total - Q_D_total

            # -------------------------------
            # 3.7 Total cost
            # -------------------------------
            res_cost = net.res_cost

            # Case 1: res_cost is just a scalar (numpy.float64, float, int)
            if isinstance(res_cost, (float, int, np.floating)):
                total_cost = float(res_cost)

            # Case 2: res_cost is a pandas object (DataFrame/Series) with a 'cost' column
            elif hasattr(res_cost, "columns") and "cost" in res_cost.columns:
                total_cost = float(res_cost["cost"].sum())

            # Case 3: some other structured object – fallback to summing all numeric values
            else:
                total_cost = float(np.array(res_cost).astype(float).sum())

            # -------------------------------
            # 3.8 Build one row of dataset
            # -------------------------------
            row = []

            # Per-bus data
            for pos in range(n_buses):
                row.extend([
                    P_D[pos],       # P_D_busX
                    Q_D[pos],       # Q_D_busX
                    V_mag[pos],     # V_busX
                    V_ang[pos],     # d_busX_deg
                ])

            # Generators
            row.extend(list(P_G_gens))
            row.extend(list(Q_G_gens))

            # Ext_grid(s)
            row.extend(list(P_G_ext))
            row.extend(list(Q_G_ext))

            # Totals, losses, cost
            row.extend([
                P_D_total,
                Q_D_total,
                P_G_total,
                Q_G_total,
                P_loss,
                Q_loss,
                total_cost,
            ])

            data.append(row)

        # ===============================
        # 4. Save to Excel
        # ===============================
        if not data:
            raise RuntimeError("No OPF scenarios converged.")
        df = pd.DataFrame(data, columns=columns)
        df.insert(0, "Data", [f"Data {i + 1}" for i in range(len(df))])
        df.insert(0, "Dataset", [f"IEEE9_OPF_Dataset_{dataset_n}"] + [""] * (len(df) - 1))

        output_filename = output_folder / f"IEEE9_OPF_Dataset_{dataset_n}.xlsx"
        df.to_excel(output_filename, index=False)
        print(f"Dataset {dataset_n} saved to {output_filename}")

    print("All IEEE 9-bus OPF datasets generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--datasets", type=int, default=1)
    parser.add_argument("--variation", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 2 or args.datasets < 1 or not 0 <= args.variation < 1:
        parser.error("Require samples >= 2, datasets >= 1 and 0 <= variation < 1")
    generate(args.output, args.datasets, args.samples, args.variation, args.seed)
