# Step 1 toy problem code

- `grpo_exact.py`: exact expected update and threshold p* for standard GRPO, Dr. GRPO, RLOO, fixed scale (Task 1, Parts 1-2)
- `batch_exact.py`: exact threshold for batch-level std (Task 1, Part 3)
- `sim.py`: SGD simulation of all five rules (Task 2)
- `predict.py`: predicted final answer rate from the exact expected update
- `run_grid.py`: runs the 300-cell grid x 20 seeds x 5 rules (about 2.5 minutes on 2 CPUs), writes grid.npz
- `plot_phase.py`: phase diagram from grid.npz

Run order: `python run_grid.py` then `python plot_phase.py`.
