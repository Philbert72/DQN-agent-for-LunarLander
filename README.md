# CSE3008 Assignment 1 — LunarLander DQN

Complete implementation of a Deep Q-Network (DQN) agent for
`LunarLander-v3`, covering all four assignment parts.

---

## File Overview

| File | Purpose |
|---|---|
| `dqn_agent.py` | Core DQN: `ReplayBuffer`, `QNetwork`, `DQNAgent` |
| `utils.py` | Plotting, evaluation, statistics helpers |
| `train.py` | Parts A, B, C — baseline, training, evaluation |
| `hyperparameter_sweep.py` | Part D — three controlled experiments |
| `report.tex` | Part D written report (LaTeX source) |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv rl_env
source rl_env/bin/activate        # Windows: rl_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# On some systems Box2D needs extra system libraries:
#   macOS: brew install swig
#   Linux: sudo apt-get install swig build-essential
```

---

## Running the Assignment

### Full run (Parts A + B + C)
```bash
python train.py
```

### Run only Part A (random baseline)
```bash
python train.py --part a
```

### Run only DQN training (Part B + C)
```bash
python train.py --part b
```

### Evaluate a saved model
```bash
python train.py --eval
```

### Part D — Hyperparameter sweep (all 3 experiments)
```bash
python hyperparameter_sweep.py
```

### Part D — Run a single experiment
```bash
python hyperparameter_sweep.py --exp epsilon   # epsilon decay
python hyperparameter_sweep.py --exp target    # target update freq
python hyperparameter_sweep.py --exp lr        # learning rate
```

---

## Output Structure

After a full run, the following directories are created:

```
checkpoints/         ← model weights saved every 50 episodes
videos/
  random_baseline/   ← Part A: 5 random-agent recordings
  training/          ← Part B: video every 50 training episodes
  test/              ← Part C: 5 test episode recordings
plots/
  partA_random_baseline.png
  partC_training_curves.png     ← 4-panel: reward, loss, ε, Q-value
  partC_test_rewards.png
  partD_epsilon_decay.png
  partD_target_update.png
  partD_learning_rate.png
lunar_lander_dqn.pth            ← best model checkpoint
```

---

## Architecture Summary

### ReplayBuffer
- Circular deque of capacity 100,000
- Stores `(state, action, reward, next_state, done)`
- Returns uniformly sampled mini-batches as tensors

### QNetwork
```
Linear(8, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 4)
```

### DQNAgent
- **ε-greedy** exploration: ε decays from 1.0 → 0.01 at rate 0.995/episode
- **Double DQN** targets: online net selects action, target net evaluates
- **Hard target update**: every 10 episodes
- **Gradient clipping**: max norm 10
- **Optimiser**: Adam, lr = 5e-4
- **Loss**: Huber (robust to large TD errors)

---

## Hyperparameters (defaults)

| Parameter | Value |
|---|---|
| Learning rate | 5e-4 |
| Discount γ | 0.99 |
| Batch size | 64 |
| Buffer size | 100,000 |
| ε start / end / decay | 1.0 / 0.01 / 0.995 |
| Target update (eps) | 10 |
| Hidden layers | 256, 256 |
| Gradient update every | 4 steps |
| Warmup steps | 1,000 |

---

## Expected Performance

| Phase | Mean reward |
|---|---|
| Random baseline | ≈ −150 to −100 |
| After 100 episodes | ≈ −50 to 50 |
| After 300 episodes | ≈ 100 to 150 |
| Solved (≥200 over 100 eps) | ≈ episode 350–500 |

---

## Submission Checklist

- [ ] `dqn_agent.py`, `utils.py`, `train.py`, `hyperparameter_sweep.py`
- [ ] `lunar_lander_dqn.pth` (trained model weights)
- [ ] Plots: `plots/partA_*.png`, `plots/partC_*.png`, `plots/partD_*.png`
- [ ] Videos: 3–5 GIFs/MP4s from `videos/random_baseline/` and `videos/test/`
- [ ] `report.pdf` (compile `report.tex` with pdflatex)
