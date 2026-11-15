# Step 2: Qwen3-0.6B with TRL GRPOTrainer

Status: not started.

**2a (Vidhya):** arithmetic prompts at several difficulty levels, an `ABSTAIN` option, and a checker that scores +1 correct, −5 wrong, 0 abstain, with tests.

**2b (Ashwin, Siddharth):** train with `scale_rewards` set to `group`, `batch`, and `none`, with G = 8, and log answer rate, wrong rate, abstain rate, and total reward per difficulty level.

What Step 1 predicts, for c = 5 and G = 8 (see `../step1_toy/task1_grpo.pdf`):

- `group`: the correct cutoff is 83% accuracy, but GRPO starts answering at about 53%. Between about 53% and 72% it settles at a partial answer rate; above about 72% it always answers. So prompts the model gets right 72–83% of the time are where it should go to always-answer when it should always abstain.
- `none` (Dr. GRPO): abstains below 83% and answers above it.
- `batch`: in between; closer to correct as the batch grows.
