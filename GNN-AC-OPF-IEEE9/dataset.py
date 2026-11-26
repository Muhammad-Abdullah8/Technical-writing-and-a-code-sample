# Lead project author and repository maintainer: Muhammad Abdullah Anwar
"""Graph data preparation and training-only normalization."""
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

n_bus = 9
in_channels = 5
out_channels = 4

def slice_df_train_val(df: pd.DataFrame, train_ratio: float = 0.8): #80% for training
    
    N = len(df)
    split_idx = int(N * train_ratio)
    if not 0 < split_idx < N:
        raise ValueError("Dataset must have nonempty training and validation splits.")
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df

def load_9bus_from_df(df: pd.DataFrame, n_bus: int = 9): #Converting Excel data to graph node features and labels
    
    num_samples = len(df)

    F = in_channels   # [P_D, Q_D, is_slack, is_pv, is_pq]
    T = out_channels  # [V_mag, angle_deg, P_G, Q_G]

    x_list = []
    y_list = []

    for i in range(num_samples):
        x_sample = []
        y_sample = []

        for bus in range(1, n_bus + 1):
            # Inputs
            P_D = df[f"P_D_bus{bus}"].iloc[i]
            Q_D = df[f"Q_D_bus{bus}"].iloc[i]
            V_mag = df[f"V_bus{bus}"].iloc[i]
            ang_deg = df[f"d_bus{bus}_deg"].iloc[i]

            # Bus types: 1 = slack, 2,3 = PV, others = PQ
            if bus == 1:
                is_slack, is_pv, is_pq = 1.0, 0.0, 0.0
            elif bus in [2, 3]:
                is_slack, is_pv, is_pq = 0.0, 1.0, 0.0
            else:
                is_slack, is_pv, is_pq = 0.0, 0.0, 1.0

            # >>> Generator outputs as targets <<<
            if bus == 1:
                P_G = df["P_G_ext1"].iloc[i]
                Q_G = df["Q_G_ext1"].iloc[i]
            elif bus == 2:
                P_G = df["P_G_gen1"].iloc[i]
                Q_G = df["Q_G_gen1"].iloc[i]
            elif bus == 3:
                P_G = df["P_G_gen2"].iloc[i]
                Q_G = df["Q_G_gen2"].iloc[i]
            else:
                P_G = 0.0
                Q_G = 0.0

            # Node features
            x_sample.append([P_D, Q_D, is_slack, is_pv, is_pq])

            # Node targets: [V, angle, P_G, Q_G]
            y_sample.append([V_mag, ang_deg, P_G, Q_G])

        x_list.append(x_sample)
        y_list.append(y_sample)

    x_raw = torch.tensor(x_list, dtype=torch.float32)  # [N, 9, 5]
    y_raw = torch.tensor(y_list, dtype=torch.float32)  # [N, 9, 4]
    return x_raw, y_raw

def normalize_9bus(x_train, y_train, x_val, y_val, type_feature_indices=(2, 3, 4)):
   
    # train stats over all samples + buses
    x_mean = x_train.mean(dim=(0, 1), keepdim=True)  # [1, 1, F]
    x_std  = x_train.std(dim=(0, 1), keepdim=True)   # [1, 1, F]
    y_mean = y_train.mean(dim=(0, 1), keepdim=True)  # [1, 1, T]
    y_std  = y_train.std(dim=(0, 1), keepdim=True)   # [1, 1, T]

    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    def _norm_x(x):
        xn = (x - x_mean) / x_std
        # keep one-hot bus-type features unnormalized
        for idx in type_feature_indices:
            xn[:, :, idx] = x[:, :, idx]
        return xn

    def _norm_y(y):
        return (y - y_mean) / y_std

    x_train_norm = _norm_x(x_train)
    y_train_norm = _norm_y(y_train)

    x_val_norm = _norm_x(x_val)
    y_val_norm = _norm_y(y_val)

    return x_train_norm, y_train_norm, x_val_norm, y_val_norm, x_mean, y_mean, x_std, y_std

def build_ieee9_edge_index():
    
    edges = [
        (1, 4), (4, 5), (5, 6), (3, 6),  
        (6, 7), (7, 8), (8, 2), (8, 9), (9, 4),  
    ]

    # Convert to 0-based indices
    edges_0 = [(i - 1, j - 1) for (i, j) in edges]

    src = []
    dst = []
    for i, j in edges_0:
        src.append(i)
        dst.append(j)
        src.append(j)
        dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def get_ieee9_loaders(
    excel_path="data/IEEE9_OPF_Dataset_1.xlsx",

    train_ratio=0.8,
    batch_size=16,
):
    # 1) Load dataset from Excel
    df = pd.read_excel(excel_path)

    # 2) Train/val split
    train_df, val_df = slice_df_train_val(df, train_ratio=train_ratio)
    print(f"Total samples: {len(df)}, Train: {len(train_df)}, Val: {len(val_df)}")

    # 3) Prepare tensors
    x_raw_train, y_raw_train = load_9bus_from_df(train_df, n_bus)
    x_raw_val,   y_raw_val   = load_9bus_from_df(val_df, n_bus)

    # 4) Normalize using train stats
    (
        x_norm_train,
        y_norm_train,
        x_norm_val,
        y_norm_val,
        x_train_mean,
        y_train_mean,
        x_train_std,
        y_train_std,
    ) = normalize_9bus(x_raw_train, y_raw_train, x_raw_val, y_raw_val)

    # 5) Build edge_index
    edge_index = build_ieee9_edge_index()
    print("Edge Index for IEEE 9-bus System:")
    print(edge_index)

    # 6) Create Data objects
    train_data_list = [
        Data(x=x, y=y, edge_index=edge_index)
        for x, y in zip(x_norm_train, y_norm_train)
    ]
    val_data_list = [
        Data(x=x, y=y, edge_index=edge_index)
        for x, y in zip(x_norm_val, y_norm_val)
    ]

    # 7) DataLoaders
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data_list,   batch_size=batch_size, shuffle=False)

    print("IEEE 9-bus data preparation completed.")
    print(f"Train samples: {len(train_data_list)}, Val samples: {len(val_data_list)}")

    # return mean/std too if you want to denormalize later
    stats = {
        "x_mean": x_train_mean,
        "x_std": x_train_std,
        "y_mean": y_train_mean,
        "y_std": y_train_std,
    }

    return train_loader, val_loader, stats

