"""
utils.py
========
Utility helpers for CSE3008 Assignment 1 – LunarLander DQN.

Contents
--------
* moving_average        – smooth a noisy signal for plots
* plot_training_curves  – rewards, loss, epsilon, avg Q-value
* plot_hyperparameter_comparison – overlay curves for Part D
* print_stats           – console summary of an episode list
* evaluate_agent        – run N greedy episodes and return rewards
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; change to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    """Compute a simple rolling mean with *full* output (same length as input).

    The first `window-1` values are computed with a smaller window so the
    output length always equals the input length.
    """
    values = np.array(values, dtype=np.float64)
    out    = np.empty_like(values)
    for i in range(len(values)):
        start     = max(0, i - window + 1)
        out[i]    = values[start : i + 1].mean()
    return out


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(
    episode_rewards:  list[float],
    losses:           list[float],
    epsilons:         list[float],
    avg_q_values:     list[float],
    save_path:        str  = "training_curves.png",
    window:           int  = 50,
    solved_threshold: float = 200.0,
) -> None:
    """Generate a 2×2 figure with the four key training diagnostics.

    Panels
    ------
    (top-left)  Episode reward + 50-ep moving average + solved line
    (top-right) Training loss (step-level, log scale)
    (bot-left)  Epsilon decay over episodes
    (bot-right) Average Q-value per update step
    """
    episodes = np.arange(1, len(episode_rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("DQN Training Diagnostics – LunarLander-v3", fontsize=14, fontweight="bold")

    # --- (0,0) Episode reward -------------------------------------------- #
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.35, linewidth=0.8,
            color="#4C9BE8", label="Episode reward")
    ma = moving_average(episode_rewards, window)
    ax.plot(episodes, ma, linewidth=2.0, color="#1A5FA8",
            label=f"{window}-ep moving avg")
    ax.axhline(solved_threshold, color="#E84C4C", linestyle="--", linewidth=1.4,
               label=f"Solved ({solved_threshold})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Episode Rewards")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (0,1) Loss -------------------------------------------------------- #
    ax = axes[0, 1]
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.4, linewidth=0.6, color="#F4A460",
            label="Huber loss")
    if len(losses) >= window:
        ax.plot(steps, moving_average(losses, window), linewidth=1.8,
                color="#8B4513", label=f"{window}-step MA")
    ax.set_yscale("log")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # --- (1,0) Epsilon ----------------------------------------------------- #
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, linewidth=2.0, color="#6A9F55")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (exploration probability)")
    ax.set_title("Epsilon Decay")
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)

    # --- (1,1) Average Q-value --------------------------------------------- #
    ax = axes[1, 1]
    q_steps = np.arange(1, len(avg_q_values) + 1)
    ax.plot(q_steps, avg_q_values, alpha=0.4, linewidth=0.6, color="#BA68C8",
            label="Mean Q (batch)")
    if len(avg_q_values) >= window:
        ax.plot(q_steps, moving_average(avg_q_values, window), linewidth=1.8,
                color="#6A0DAD", label=f"{window}-step MA")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Mean Q(s,a)")
    ax.set_title("Average Q-Value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Hyper-parameter comparison plot (Part D)
# ---------------------------------------------------------------------------

def plot_hyperparameter_comparison(
    results:     dict[str, list[float]],
    param_name:  str,
    save_path:   str  = "hyperparam_comparison.png",
    window:      int  = 50,
    solved_threshold: float = 200.0,
) -> None:
    """Overlay smoothed reward curves for multiple hyper-parameter values.

    Parameters
    ----------
    results    : dict mapping label (str) → list of episode rewards.
    param_name : human-readable parameter name shown in the title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for (label, rewards), colour in zip(results.items(), colours):
        eps = np.arange(1, len(rewards) + 1)
        ax.plot(eps, moving_average(rewards, window),
                linewidth=2.2, color=colour, label=label)

    ax.axhline(solved_threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Solved ({solved_threshold})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-ep moving avg)")
    ax.set_title(f"Hyperparameter Comparison: {param_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Comparison plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(rewards: list[float], label: str = "Agent") -> dict:
    """Print and return summary statistics for a list of episode rewards."""
    arr = np.array(rewards)
    stats = {
        "mean":    float(arr.mean()),
        "std":     float(arr.std()),
        "median":  float(np.median(arr)),
        "min":     float(arr.min()),
        "max":     float(arr.max()),
        "success": float((arr >= 200).mean() * 100),   # % episodes ≥ 200
    }
    print(
        f"\n{'='*55}\n"
        f"  {label}\n"
        f"{'='*55}\n"
        f"  Episodes   : {len(rewards)}\n"
        f"  Mean reward: {stats['mean']:8.2f}  ±  {stats['std']:.2f}\n"
        f"  Median     : {stats['median']:8.2f}\n"
        f"  Min / Max  : {stats['min']:.2f} / {stats['max']:.2f}\n"
        f"  Success (≥200): {stats['success']:.1f}%\n"
        f"{'='*55}"
    )
    return stats


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent,                       # DQNAgent instance
    env,                         # Gymnasium environment
    n_episodes:   int   = 100,
    render:       bool  = False,
    epsilon_eval: float = 0.0,   # use 0 for fully greedy evaluation
) -> list[float]:
    """Run `n_episodes` episodes with the agent's greedy policy.

    The agent's epsilon is temporarily overridden to `epsilon_eval`
    (default 0 = fully greedy) and restored afterwards.

    Returns
    -------
    List of total rewards, one per episode.
    """
    saved_eps      = agent.epsilon
    agent.epsilon  = epsilon_eval
    rewards        = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done      = False
        ep_reward = 0.0

        while not done:
            action              = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done               = terminated or truncated
            ep_reward         += reward
            state              = next_state

        rewards.append(ep_reward)
        if (ep + 1) % 20 == 0:
            print(f"  Eval episode {ep+1:>3}/{n_episodes} | reward: {ep_reward:7.2f}")

    agent.epsilon = saved_eps
    return rewards
