"""
hyperparameter_sweep.py
=======================
Part D – Hyperparameter experimentation for CSE3008 Assignment 1.

Runs three controlled experiments, each varying one hyper-parameter
while holding the rest constant at the baseline values.

Experiments
-----------
1. Epsilon decay rate          (exploration vs. exploitation trade-off)
2. Target-network update freq  (training stability)
3. Learning rate               (gradient step size)

Usage
-----
  python hyperparameter_sweep.py                # run all experiments
  python hyperparameter_sweep.py --exp lr       # run only the LR sweep

Each experiment saves a comparison plot to plots/partD_*.png.

Lecture connection
------------------
* Epsilon decay   → directly tied to Lecture 1's Exploration vs Exploitation.
  A fast decay exploits earlier but risks a sub-optimal policy; a slow
  decay explores longer at the cost of slower convergence.
* Target-network  → stabilises training by decoupling the TD target from
  the online network.  Too frequent updates → the target 'chases its tail';
  too infrequent → stale targets slow learning.
* Learning rate   → controls the magnitude of gradient updates to the
  Q-network (the function approximator of q*(s,a) per Lecture 2).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import gymnasium as gym

from dqn_agent import DQNAgent
from utils import plot_hyperparameter_comparison, print_stats


# ============================================================
# Shared baseline config
# ============================================================

BASELINE = dict(
    env_id        = "LunarLander-v3",
    seed          = 42,
    n_episodes    = 500,
    max_steps     = 1_000,
    update_every  = 4,
    warmup_steps  = 1_000,
    lr            = 5e-4,
    gamma         = 0.99,
    epsilon_start = 1.0,
    epsilon_end   = 0.01,
    epsilon_decay = 0.995,
    batch_size    = 64,
    buffer_size   = 100_000,
    target_update = 10,
    hidden1       = 256,
    hidden2       = 256,
)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Core training function (no video recording to save time)
# ============================================================

def train_one_config(cfg: dict, label: str) -> list[float]:
    """Train a DQN agent with the given config and return episode rewards."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(cfg["env_id"])
    env.reset(seed=cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim      = state_dim,
        action_dim     = action_dim,
        lr             = cfg["lr"],
        gamma          = cfg["gamma"],
        epsilon_start  = cfg["epsilon_start"],
        epsilon_end    = cfg["epsilon_end"],
        epsilon_decay  = cfg["epsilon_decay"],
        batch_size     = cfg["batch_size"],
        buffer_size    = cfg["buffer_size"],
        target_update  = cfg["target_update"],
        hidden1        = cfg["hidden1"],
        hidden2        = cfg["hidden2"],
        device         = device,
    )

    episode_rewards: list[float] = []
    step_count = 0
    t0 = time.time()
    print(f"\n  [{label}] Starting {cfg['n_episodes']} episodes …")

    for episode in range(1, cfg["n_episodes"] + 1):
        state, _ = env.reset()
        done      = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)

            if (step_count >= cfg["warmup_steps"] and
                    step_count % cfg["update_every"] == 0):
                agent.update()

            state      = next_state
            ep_reward += reward
            step_count += 1

        agent.end_episode()
        episode_rewards.append(ep_reward)

        if episode % 100 == 0:
            mean100 = np.mean(episode_rewards[-100:])
            elapsed = time.time() - t0
            print(f"    Ep {episode:>4} | mean100: {mean100:7.2f} | "
                  f"ε: {agent.epsilon:.4f} | {elapsed:.0f}s")

    env.close()
    print(f"  [{label}] Done. Final mean-100 = {np.mean(episode_rewards[-100:]):.2f}")
    return episode_rewards


# ============================================================
# Experiment 1 – Epsilon decay rate
# ============================================================

def experiment_epsilon_decay() -> None:
    """Compare fast, baseline, and slow epsilon decay schedules.

    Hypothesis
    ----------
    * Fast decay  (0.980): agent commits to exploitation quickly, converges
      fast initially, but may settle in a local optimum.
    * Baseline    (0.995): balanced exploration/exploitation.
    * Slow decay  (0.999): maintains high exploration longer, potentially
      finding a better policy but at the cost of more episodes.
    """
    print("\n" + "=" * 60)
    print("  Experiment 1 – Epsilon Decay Rate")
    print("=" * 60)

    decay_values = {
        "Fast  (0.980)":     0.980,
        "Baseline (0.995)":  0.995,
        "Slow  (0.999)":     0.999,
    }

    results: dict[str, list[float]] = {}
    for label, decay in decay_values.items():
        cfg = {**BASELINE, "epsilon_decay": decay}
        rewards = train_one_config(cfg, label)
        results[label] = rewards
        print_stats(rewards[-100:], label=f"  {label} (last 100 eps)")

    plot_hyperparameter_comparison(
        results    = results,
        param_name = "Epsilon Decay Rate",
        save_path  = os.path.join(PLOT_DIR, "partD_epsilon_decay.png"),
    )
    print(f"\n[Exp 1] Plot → {PLOT_DIR}/partD_epsilon_decay.png")


# ============================================================
# Experiment 2 – Target-network update frequency
# ============================================================

def experiment_target_update() -> None:
    """Compare target network hard-update frequencies.

    Hypothesis
    ----------
    * Frequent (every 1 ep) : target tracks online net closely → potential
      instability and oscillation.
    * Baseline (every 10 ep): well-established default balance.
    * Infrequent (every 50 ep): very stable targets but slow credit
      propagation; may cause the agent to under-correct.
    """
    print("\n" + "=" * 60)
    print("  Experiment 2 – Target Network Update Frequency")
    print("=" * 60)

    freq_values = {
        "Very frequent  (1 ep)" : 1,
        "Baseline (10 ep)"      : 10,
        "Infrequent  (50 ep)"   : 50,
    }

    results: dict[str, list[float]] = {}
    for label, freq in freq_values.items():
        cfg = {**BASELINE, "target_update": freq}
        rewards = train_one_config(cfg, label)
        results[label] = rewards
        print_stats(rewards[-100:], label=f"  {label} (last 100 eps)")

    plot_hyperparameter_comparison(
        results    = results,
        param_name = "Target Network Update Frequency (episodes)",
        save_path  = os.path.join(PLOT_DIR, "partD_target_update.png"),
    )
    print(f"\n[Exp 2] Plot → {PLOT_DIR}/partD_target_update.png")


# ============================================================
# Experiment 3 – Learning rate
# ============================================================

def experiment_learning_rate() -> None:
    """Compare three learning rates for the Adam optimiser.

    Hypothesis
    ----------
    * High LR (5e-3) : fast early learning but risk of divergence / poor
      asymptotic performance due to overshooting the Q-value landscape.
    * Baseline (5e-4): standard choice; stable convergence.
    * Low LR  (5e-5) : extremely stable but slow; may not converge within
      500 episodes.
    """
    print("\n" + "=" * 60)
    print("  Experiment 3 – Learning Rate")
    print("=" * 60)

    lr_values = {
        "High   (5e-3)"  : 5e-3,
        "Baseline (5e-4)": 5e-4,
        "Low    (5e-5)"  : 5e-5,
    }

    results: dict[str, list[float]] = {}
    for label, lr in lr_values.items():
        cfg = {**BASELINE, "lr": lr}
        rewards = train_one_config(cfg, label)
        results[label] = rewards
        print_stats(rewards[-100:], label=f"  {label} (last 100 eps)")

    plot_hyperparameter_comparison(
        results    = results,
        param_name = "Learning Rate (Adam)",
        save_path  = os.path.join(PLOT_DIR, "partD_learning_rate.png"),
    )
    print(f"\n[Exp 3] Plot → {PLOT_DIR}/partD_learning_rate.png")


# ============================================================
# Entry point
# ============================================================

EXPERIMENTS = {
    "epsilon" : experiment_epsilon_decay,
    "target"  : experiment_target_update,
    "lr"      : experiment_learning_rate,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part D – Hyperparameter sweep for LunarLander DQN"
    )
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run (default: all)"
    )
    args = parser.parse_args()

    if args.exp == "all":
        for fn in EXPERIMENTS.values():
            fn()
    else:
        EXPERIMENTS[args.exp]()


if __name__ == "__main__":
    main()
