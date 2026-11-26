# Graph Neural Networks for AC Optimal Power Flow | IEEE 9-Bus

**Lead project author and repository maintainer:** Muhammad Abdullah Anwar  
**Original project supervisor:** Oleg Khamisov, Skoltech

## Overview

This research project explores a supervised graph neural network surrogate for AC optimal power flow (AC-OPF) on the IEEE 9-bus benchmark. Conventional OPF solutions generated with pandapower provide labels for a PyTorch Geometric model. The model predicts bus voltage magnitude, voltage angle and generator active/reactive power from load demands and bus types.

## Methodology

1. Randomly scale each load's active and reactive demand together by up to ±20% and solve AC-OPF with quadratic generation costs.
2. Encode the grid as nine nodes and nine bidirectional physical connections (18 directed edges).
3. Split the dataset 80/20; normalize using training statistics only, keeping bus-type indicators unchanged.
4. Train a three-layer GraphConv model (5 → 100 → 100 → 4) with ReLU, MSE and Adam.
5. Evaluate RMSE in physical units over the full validation set; generator power errors are restricted to generator buses.

## Repository structure

```text
gnn-ac-opf-ieee9/
├── generate_dataset.py   # AC-OPF scenario generation
├── dataset.py            # Graph construction and normalization
├── model.py              # GNN architecture
├── train.py              # Training and evaluation
├── requirements.txt
├── data/                 # Generated Excel datasets
├── outputs/              # Saved model and metrics
└── docs/                 # Implementation notes
```

## Getting started

Use Python 3.11 in a virtual environment.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_dataset.py --samples 500 --variation 0.2
python train.py --epochs 70 --batch-size 16
```
