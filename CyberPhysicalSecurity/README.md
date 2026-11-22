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

## Important model notes

- **Physical vs measurement attacks:** In the IEEE-14 script, attack actions directly change network loads; this is a physical-load perturbation model, **not** a state-estimator measurement-only FDI attack. The end-of-file DC SE section is commented out. The Kundur FDI scripts model falsified EMS load beliefs followed by a truth-model check; the interpretation differs from changing actual physical demand.
- **Optimization:** The protection games enumerate particular attack levels, buses and/or corner points; they do **not** establish globally optimal continuous solutions unless separately verified.
- **Network assumptions:** The Kundur representations are manually constructed, steady-state `pandapower` models with user-specified impedances, voltage levels, dispatch and generation limits. These are not full electromagnetic transient/RTDS dynamic models.
- **Collapse and feasibility:** Minimum-voltage thresholds are proxies, not proof of voltage collapse. OPF/PF non-convergence can depend on numerical settings.
- **Reliability:** The scripts' input assumptions, attack units, load-shedding cost conventions and truth-network dispatch treatment should be reviewed and the results cross-checked before publication.
- **Formatting correction:** The supplied integrated Kundur file was exported as markdown with escaped characters and non-breaking spaces; its executable copy has these formatting artifacts removed. Its original file remains unchanged in `originals/`.

