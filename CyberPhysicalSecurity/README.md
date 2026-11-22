# Cyber-Physical Security Analysis of IEEE 14-Bus and Kundur Two-Area Power Systems

Research scripts for **steady-state AC optimal power flow (AC-OPF), generator-contingency analysis, load-attack scenarios, false-data-injection (FDI)-inspired experiments, bus/generator criticality, and defender–attacker protection decisions** using `pandapower`.

> **Research status:** These are organized versions of supplied research scripts, not independently validated or peer-reviewed simulation benchmarks. Keep the models' simplifying assumptions in mind before reporting quantitative conclusions.

## Repository layout

```text
kundur-ieee14-cyber-physical-security/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── ieee14_generator_tripping_physical_attack.py
│   ├── kundur_integrated_trip_fdi_stackelberg.py
│   ├── kundur_fdi_bus_criticality_plots.py
│   └── kundur_generator_trip_criticality.py
└── originals/
    ├── Pasted text(20261122-202029).txt
    ├── Pasted markdown (2)(1).md
    ├── Pasted text (3).txt
    └── Pasted text (4).txt
```

## Experiments

| Script | System | Main experiment |
|---|---|---|
| `ieee14_generator_tripping_physical_attack.py` | IEEE 14-bus | AC-OPF, generator tripping, physical load attack and protective investment scan |
| `kundur_integrated_trip_fdi_stackelberg.py` | Kundur two-area | Generator trip / FDI-inspired case studies, criticality analysis and discrete defender–attacker games |
| `kundur_fdi_bus_criticality_plots.py` | Kundur two-area | Comparison of false-measurement load injections at Bus 7 and Bus 9, with OPF/PF metrics and figures |
| `kundur_generator_trip_criticality.py` | Kundur two-area | Generator trip criticality ranking and malicious trip case study with generator protection |

## Installation

Python 3.10+ is recommended. Dependencies are listed without version pins so that you can first reproduce the environment that produced the original simulations; pin exact versions after numerical verification.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run experiments

From the repository root:

```bash
python scripts/ieee14_generator_tripping_physical_attack.py
python scripts/kundur_integrated_trip_fdi_stackelberg.py
python scripts/kundur_fdi_bus_criticality_plots.py
python scripts/kundur_generator_trip_criticality.py
```

Run scripts **individually**: they use different network assumptions, OPF setups and experiment settings. Results should not be combined as though they came from a single consistent model. Figures are displayed using Matplotlib; a graphical backend may be needed.

## Research workflow

1. Construct a benchmark network and select reference generator/slack configurations.
2. Configure generator costs, load shedding where applicable, and AC-OPF constraints.
3. Run a nominal reference and evaluate generator trip or manipulated-load scenarios.
4. Calculate scenario-dependent cost, voltage, loading, tie-flow or dispatch-deviation metrics.
5. Compare criticality and defender protection choices or generate sensitivity plots.
