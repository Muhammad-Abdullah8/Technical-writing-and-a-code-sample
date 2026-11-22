# IEEE 9-Bus Synchronous-Generator Dynamics and Governor-Gain Optimization

Research scripts for studying electromechanical dynamics and governor-gain tuning in an **IEEE 9-bus, three-generator** power system using Python, pandapower and SciPy.

The scripts cover (1) local eigenvalue-based tuning of the governor gain $K_g$, (2) dynamic-simulation-based optimization with a generator-trip contingency, and (3) time-domain simulation and plots of generator behavior under a load change.

> **Research status:** Organized from the original experimental scripts. The physical modeling assumptions and numerical results have **not** been independently validated. These scripts are supplied as research prototypes rather than a verified dynamic-security assessment tool.

## Repository layout

```text
ieee9-synchronous-generator-dynamics/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── optimize_governor_gains_eigenvalues.py
│   ├── optimize_governor_gains_contingency.py
│   └── simulate_and_plot_dynamics.py
└── originals/
    ├── Only_optimize_Kg_4thOrd.py
    ├── OPT_final_4th _order_with_gen_disconnection.py
    └── Plot_4th_order_SynGe_Dynamics_For_OPT_values.py
```

The files in `originals/` are unchanged copies of the uploaded research scripts. The corresponding files in `scripts/` have descriptive names, module docstrings, and numbered section headings. The numerical statements are otherwise retained to preserve the original experimental settings.

## Installation

Use Python 3.10+ in a virtual environment (a recommendation, not a validated version matrix):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies: NumPy, pandapower, SciPy, and Matplotlib. They are unpinned; record exact dependency versions for reproducible research.

## Running the experiments

Run from the repository root:

### 1. Local eigenvalue-based governor tuning

```bash
python scripts/optimize_governor_gains_eigenvalues.py
```

- Builds the IEEE 9-bus network and defines a **six-state local generator model** for an eigenvalue analysis.
- Computes eigenvalues of each generator's local state-space matrix, neglecting network and electrical-current coupling in that matrix.
- Minimizes a penalty based on unstable eigenvalues and damping ratios using `scipy.optimize.minimize` with Nelder–Mead.
- Initial governor-gain vector: **[90, 100, 110]**; nominal bounds: **1–250** for each gain.
- Displays the gain trajectories throughout objective evaluations.

### 2. Dynamic optimization with generator disconnection

```bash
python scripts/optimize_governor_gains_contingency.py
```

- Uses a **seven-state vector per generator**: rotor angle, rotor speed, transient $q$- and $d$-axis voltages, mechanical power, governor-valve state, and integral voltage-control error.
- Couples `solve_ivp` differential equations to pandapower power-flow calculations.
- Trips the generator represented at **bus index 2** at **8 s**.
- Minimizes maximum absolute rotor-speed deviation plus a local-eigenvalue stability penalty.
- Uses Nelder–Mead with gain bounds of **1–250** and an early-stop threshold of **1.75 rad/s** for the deviation metric.
- Simulates **0–22 s**, with 200 requested output samples.

### 3. Dynamic simulation and visualization

```bash
python scripts/simulate_and_plot_dynamics.py
```

- Models the three generators with governor/turbine and simplified excitation dynamics.
- At **10 s**, sets the active load on **bus index 6** to **200 MW**.
- Uses governor gains **[5.3, 5.4, 5.3]**, with commented-out alternative sets in the original source.
- Simulates **0–120 s** and plots mechanical power, transient internal voltages, rotor-speed deviation, stator currents, voltage angles, internal electrical-power quantity, and terminal voltage.

**Note on indexing:** The source code uses pandapower's zero-based bus indices; bus index 2 is the third bus in human-facing one-based numbering.

## Model outline

The time-domain programs use the state vector, for each generator,

$$
y_i = [\delta_i,\,\omega_i,\,E'_{qi},\,E'_{di},\,P_{mi},\,v_i,\,z_i]^T,
$$

where $z_i$ represents the integral of voltage error. Each derivative evaluation uses bus voltages from the generator state, solves a pandapower load flow, and estimates generator currents with `fsolve`. The three 7-state generator vectors form a 21-state integration problem.

The eigenvalue-only optimizer instead uses a *six-state local linearized surrogate* per generator and is **not** a full eigenanalysis of the 21-state network-coupled differential-algebraic system.

## Important limitations and checks before publication

1. **Model order terminology.** Filenames refer to a "4th-order" model, but the dynamic scripts explicitly integrate **seven states per generator** (including governor and integral controller states). Describe these separately in publications.
2. **Optimization side effects.** The generator-trip script modifies the network, excitation array, governor-gain vector and power references inside an ODE right-hand-side callback. Adaptive integrators can evaluate the RHS out of chronological order, and repeated optimization trials may be affected by mutated shared data. This deserves correction before numerical reproducibility claims.
3. **Contingency outside the horizon.** The optimization script also includes an active-load change at **80 s**, but its integration ends at **22 s**; that change is not exercised in that experiment.
4. **Stator current equations.** In the two dynamic scripts, certain reactance/resistance contributions in `current_constr()` are multiplied by zero. Confirm whether this is an intended simplifying assumption.
5. **Transient voltage dynamics.** The $d$-axis transient-voltage derivative is implemented without division by `Tqo_prime` in the dynamic scripts. Verify this against the intended mathematical model.
6. **Reference buses.** The network is modeled with multiple `ext_grid` elements; verify this equivalent-generator representation and slack treatment for the intended dynamic simulation.
7. **Local stability penalties.** The eigenvalue-only script penalizes zero-magnitude eigenvalues, whereas the dynamic optimizer ignores near-zero modes as reference-angle modes. Interpret the resulting objectives separately.
8. **Solver and status checks.** Verify `solve_ivp.success`, `fsolve` convergence, power-flow convergence, and optimizer termination, and establish convergence tests before treating reported optima as validated.
9. **Trace timestamps.** The plotting script records currents inside the ODE callback (adaptive internal evaluations), not exclusively at `t_eval`; those trace timestamps need not match the saved state-solution timestamps.
10. **Plot labeling.** A plot titled `Pe Internal V` is generated from electrical power multiplied by `-100` but has a voltage-unit y-axis label. Check its units before reuse.
11. **Methodology claims.** Nelder–Mead and early stopping do not prove global optimality. Report the objective, gain vector, solver status and repeat-run sensitivity when presenting findings.

## Suggested future improvements

- Refactor model and scenario settings into immutable configuration objects.
- Implement a pure, event-consistent dynamic RHS with separate pre- and post-trip integration.
- Add quantitative regression tests against a validated baseline and document the chosen generator model equations.
- Save gain histories, operating conditions, solver diagnostics and figures to a versioned `results/` directory.
- Pin dependency versions and add a citation/license once the research code is ready for public release.

## Provenance

All three Python scripts were supplied by the repository author. The source files under `originals/` preserve those uploads, and the annotated files under `scripts/` retain the research calculations and experiment parameter choices.
