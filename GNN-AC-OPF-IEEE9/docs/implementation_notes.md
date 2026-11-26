# Implementation notes

- Reference report: *Graph Neural Network Approach for Optimal Power Flow*, Phuriphat Wongsuwan.

- Corrected the apparent target mapping error: pandapower case9 convention places the external grid at bus 1 and the two generators at buses 2 and 3. The original training script assigned the external-grid output to bus 3. Verify this mapping against the installed case9 network before relying on results.
- Replaced machine-specific Windows paths with command-line options; moved execution behind main guards.
- Evaluation now covers all validation samples instead of the first batch only.
- Training remains supervised MSE, without explicit AC feasibility constraints.
