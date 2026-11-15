# GRPO advantage normalization under asymmetric rewards

ECE 59500 (Reinforcement Learning, Prof. Mahsa Ghasemi), Fall 2026.

GRPO rescales each answer's score by the spread of scores in its group. When a wrong answer costs much more than a right one earns (+1 / −5 / 0 for abstaining), that rescaling weakens the penalty, and the model answers where it should abstain. We measure how much, test which rescaling rules fix it while keeping the abstain option, and check whether the toy-problem results hold on a small language model.

Proposal: [`docs/proposal.pdf`](docs/proposal.pdf)

## Team and tasks

| Step | What | Owner | Due |
|---|---|---|---|
| 1 | Toy problem: exact math and simulation | Pallavi Buwa | Nov 13 (done) |
| 2a | LLM task: arithmetic dataset, ABSTAIN option, +1/−5/0 checker | Vidhya Thangavelu | Nov 18 |
| 2b | LLM training runs with TRL `GRPOTrainer` | Ashwin Sreedhar, Siddharth Murali | Nov 25 |
| 3 | Case study on the earlier 4B/8B code-review runs | Ashwin Sreedhar, Siddharth Murali | Nov 25 |
| – | Related work | All | Nov 25 |
| – | Final report | Ashwin Sreedhar | Dec 4 |

## Layout

```
docs/               proposal (final report goes here too)
step1_toy/          answer-or-abstain toy problem: derivation PDF, phase diagram, code
step2_llm/          Qwen3-0.6B + TRL experiments (in progress)
step3_case_study/   analysis of the code-review GRPO logs (in progress)
```

## Running Step 1

```
pip install -r requirements.txt
cd step1_toy
python grpo_exact.py      # thresholds for standard GRPO, Dr. GRPO, RLOO, fixed scale
python batch_exact.py     # thresholds for batch-level std
python run_grid.py        # 5 rules x 300 cells x 20 seeds (~2.5 min), writes grid.npz
python plot_phase.py      # phase_diagram.png
```

Work on a branch and open a pull request into `main`.
