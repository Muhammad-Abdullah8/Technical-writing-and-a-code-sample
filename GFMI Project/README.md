# Joint Optimization of Synchronous Generator Parameters — IEEE 9-Bus System

Research scripts for **joint tuning of inertia constants (H), damping coefficients (D), and governor gains (Kg)** in the IEEE 9-bus test system. The study compares SciPy's **Dual Annealing** and **Differential Evolution** optimization algorithms, with power-flow evaluations performed using pandapower and time-domain dynamics simulated with SciPy.

> **Research status:** These are organized versions of exploratory research scripts, **not verified production-grade power system models**. Numerical output has not been independently validated. The original scripts are preserved under `originals/`.

## Repository layout

```text
ieee9-joint-parameter-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── dual_annealing_two_stage_governor.py
│   ├── differential_evolution_single_stage_governor.py
│   └── dual_annealing_single_stage_governor.py
└── originals/
    ├── dual_annealing_cmplx_H_D_Kg.py
    ├── differntial_evolution_Kg_H_D_02.py
    └── dual_annealing_simpleG_Kg_H_D.py
```

## Experiments

| Script | Optimization method | Model per generator | Load change | Solver and time interval |
|---|---|---|---|---|
| `dual_annealing_two_stage_governor.py` | Dual Annealing | 4 states: rotor angle, speed, mechanical power, valve position | +100 MW at 8 s in ODE | RK23, 0–25 s |
| `differential_evolution_single_stage_governor.py` | Differential Evolution | 3 states: rotor angle, speed, mechanical power | +72 MW at 10 s | Radau, 0–40 s |
| `dual_annealing_single_stage_governor.py` | Dual Annealing | 3 states: rotor angle, speed, mechanical power | +72 MW at 10 s | RK23, 0–25 s |

**Do not treat these three experiments as identical benchmark runs.** Besides the optimizer, their state-space formulations, governor time constants, integration methods, time horizons and some penalties differ.

## Mathematical model

The general swing equation used in the scripts is

```text
d(delta_i)/dt = omega_i - omega_s
d(omega_i)/dt = [P_m,i - P_e,i - D_i (omega_i - omega_s)] / (2 H_i)
```

The **single-stage** governor uses

```text
d(P_m,i)/dt = [-P_m,i + P_m,ref,i - Kg_i (omega_i - omega_s)] / Tg_i
```

The **two-stage** turbine/governor uses

```text
d(P_m,i)/dt = (-P_m,i + v_i) / Tt_i
d(v_i)/dt = [-v_i - Kg_i (omega_i - omega_s) + P_m,ref,i] / Tv_i
```

For each of three generators, the decision-vector layout is

```text
x = [H1, H2, H3, D1, D2, D3, Kg1, Kg2, Kg3]
```

All three scripts use these bounds:

| Parameter | Bound |
|---|---|
| `H` | 1–20 |
| `D` | 0.2–20 |
| `Kg` | 1–20 |

The objective combines rotor-speed deviation (labelled *nadir* in some scripts), a discrete rate-of-change proxy and penalties for generator electrical power/current limits. Values and penalty scalings are script-specific; inspect each script before comparing objective scores. Note that the numerical derivative is in **rad/s²**, rather than conventional RoCoF in **Hz/s**, unless converted appropriately.

## Installation

Use Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

From the project root, run **one experiment at a time**:

```bash
python scripts/dual_annealing_two_stage_governor.py
python scripts/differential_evolution_single_stage_governor.py
python scripts/dual_annealing_single_stage_governor.py
```

Each experiment prints progress/parameter estimates and, when an eligible candidate is stored, displays plots of rotor-speed deviation and electrical generator power. Computation can be expensive because every objective call entails repeated power-flow solutions.

## Code sections

Each script is divided into labeled sections for network construction, base-case measurements, machine parameters, disturbance/initial states, dynamic simulation, objective/constraints, optimization and plotting. The run phase is guarded with `if __name__ == "__main__":` to avoid starting optimization when imported. Network initialization remains at module scope and still runs on import.

## Important limitations and review items

1. **Disturbance timing inconsistency in the two-stage script:** its ODE increases the load at **8 s**, while the power/current constraint scan and plotting use **10 s**. This remains unchanged in the organized version.
2. **Discontinuity and adaptive integration:** the two dual-annealing models update network load when `t >= threshold` without explicitly resetting it for earlier time calls. Adaptive ODE solvers may evaluate non-monotonic times, so reproducibility can be affected. Consider using a time-conditional assignment on every evaluation and splitting integration at the disturbance time.
3. **Early stopping:** original scripts raise a general `Exception` when thresholds are met. The optimizers catch this to stop, but their status may be misleading and other exceptions could be swallowed. A dedicated stop exception is preferable.
4. **Best candidate selection:** the saved `best` candidate is chosen using a speed metric, not the full penalized objective. It may therefore differ from the optimizer's minimum-cost result.
5. **Reference network modeling:** the scripts replace generators with three `ext_grid` elements, impose voltage angles, and overwrite all bus nominal voltages with 345 kV. This is a study-specific approximation; it is not a validated synchronous-machine/network interface.
6. **Constraint and measurement units:** verify whether currents calculated from nominal voltages, the MW reference powers, the RoCoF proxy and relative versus absolute penalties match the intended physical constraints. In particular, limit thresholds and penalty weights differ across files.
7. **Numerical performance:** no benchmark outcomes are asserted; optimization and convergence have not been executed or validated in this packaging step. Dependency versions are intentionally unpinned. Pin versions when preparing a reproducible publication artifact.

## Reproducibility and citation

To compare optimization strategies meaningfully, first harmonize the model, disturbance, solver, objective, constraints, random seeds and termination criteria. Then record Python/library versions, machine specification, successful convergence flags, parameter estimates and the time-series data used to create figures.

If this code supports a thesis or publication, add authorship, affiliation, related-paper citation, dataset provenance, and a suitable `LICENSE` file before public release.
