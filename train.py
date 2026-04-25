"""
train.py
========
Full training pipeline for CSE3008 Assignment 1 – LunarLander DQN.

Parts covered
-------------
  Part A  – random-policy baseline (100 episodes, stats + GIF)
  Part B  – DQN training loop (≥ 500 episodes)
  Part C  – evaluation, learning-curve plots, test recordings

Usage
-----
  python train.py               # full run (A + B + C)
  python train.py --part a      # baseline only
  python train.py --part b      # DQN training only
  python train.py --eval        # evaluate a saved model

Quick-start example
-------------------
  conda activate rl_env         # or: pip install -r requirements.txt
  python train.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from dqn_agent import DQNAgent
from utils import (
    evaluate_agent,
    moving_average,
    plot_training_curves,
    print_stats,
)


# ============================================================
# Configuration
# ============================================================

CFG = dict(
    # --- environment ---
    env_id        = "LunarLander-v3",
    seed          = 42,

    # --- training ---
    n_episodes    = 700,           # ≥ 500 as required by Part B
    max_steps     = 1_000,         # safety cap per episode
    update_every  = 4,             # gradient step every N environment steps
    warmup_steps  = 1_000,         # fill buffer before first update

    # --- DQN hyper-parameters (Part B defaults) ---
    lr            = 5e-4,
    gamma         = 0.99,
    epsilon_start = 1.0,
    epsilon_end   = 0.01,
    epsilon_decay = 0.995,
    batch_size    = 64,
    buffer_size   = 100_000,
    target_update = 10,            # hard update every N episodes
    hidden1       = 256,
    hidden2       = 256,

    # --- checkpointing / output ---
    checkpoint_every = 50,         # save model every N episodes
    video_every      = 50,         # record a video every N training episodes
    checkpoint_dir   = "checkpoints",
    video_dir        = "videos",
    plot_dir         = "plots",
    model_path       = "lunar_lander_dqn.pth",

    # --- evaluation (Part C) ---
    eval_episodes = 100,
)


# ============================================================
# Helper: make environment
# ============================================================

def make_env(render_mode: str = "rgb_array", seed: int = 42) -> gym.Env:
    env = gym.make(CFG["env_id"], render_mode=render_mode)
    env.reset(seed=seed)
    return env


# ============================================================
# Part A – Random-policy baseline
# ============================================================

def run_part_a(n_episodes: int = 100) -> None:
    """Run a random agent and collect statistics + recordings.

    Lecture connection
    ------------------
    The random policy is the most naive strategy: π(a|s) = uniform.
    It has no notion of value functions or optimal control (Lecture 2).
    Its poor performance motivates the need for DQN.
    """
    print("\n" + "=" * 60)
    print("  PART A – Random Policy Baseline")
    print("=" * 60)

    os.makedirs(CFG["video_dir"], exist_ok=True)

    # wrap with video recorder for first 5 episodes
    env = gym.make(CFG["env_id"], render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=os.path.join(CFG["video_dir"], "random_baseline"),
        episode_trigger=lambda ep: ep < 5,
        name_prefix="random",
    )
    env.reset(seed=CFG["seed"])

    rewards   = []
    lengths   = []
    successes = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        done      = False
        ep_reward = 0.0
        steps     = 0

        while not done:
            action = env.action_space.sample()    # uniform random
            state, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            ep_reward += reward
            steps     += 1

        rewards.append(ep_reward)
        lengths.append(steps)
        if ep_reward >= 200:
            successes += 1

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:>3}/{n_episodes} | "
                  f"reward: {ep_reward:7.2f} | steps: {steps}")

    env.close()

    # --- statistics ---
    print_stats(rewards, label="Random Baseline (100 episodes)")
    print(f"  Average episode length : {np.mean(lengths):.1f} steps")

    # --- simple reward plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CFG["plot_dir"], exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.6, color="#4C9BE8", linewidth=0.9, label="Episode reward")
    ax.axhline(np.mean(rewards), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(rewards):.1f}")
    ax.axhline(200, color="red", linestyle="--", linewidth=1.2, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Part A – Random Policy Baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["plot_dir"], "partA_random_baseline.png"), dpi=150)
    plt.close()
    print(f"\n[Part A] Plot saved → {CFG['plot_dir']}/partA_random_baseline.png")
    print(f"[Part A] Videos saved → {CFG['video_dir']}/random_baseline/")


# ============================================================
# Part B + C – DQN Training
# ============================================================

def run_training() -> DQNAgent:
    """Main DQN training loop (Parts B and C).

    Algorithm per episode
    ---------------------
    1. Reset environment → initial state s_0.
    2. For each step t:
       a. Select action a_t via ε-greedy policy.
       b. Execute a_t → (s_{t+1}, r_t, done).
       c. Store (s_t, a_t, r_t, s_{t+1}, done) in replay buffer.
       d. Every `update_every` steps, sample mini-batch and do gradient step.
    3. End of episode: decay ε, optionally hard-update target network.
    4. Save checkpoint every `checkpoint_every` episodes.

    Bellman update (Lecture 2 – Optimal Bellman Equation)
    -------------------------------------------------------
    q*(s, a) = E[ r + γ · max_{a'} q*(s', a') ]

    We minimise the mean Huber loss between Q_online(s, a) and
    the TD target r + γ · max_{a'} Q_target(s', a').
    """
    print("\n" + "=" * 60)
    print("  PART B + C – DQN Training")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CFG["video_dir"],      exist_ok=True)
    os.makedirs(CFG["plot_dir"],       exist_ok=True)

    # --- environment with occasional video recording ------------------- #
    env = gym.make(CFG["env_id"], render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=os.path.join(CFG["video_dir"], "training"),
        episode_trigger=lambda ep: ep % CFG["video_every"] == 0,
        name_prefix="dqn_train",
    )
    env.reset(seed=CFG["seed"])
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.n               # 4

    # --- build agent ----------------------------------------------------- #
    agent = DQNAgent(
        state_dim      = state_dim,
        action_dim     = action_dim,
        lr             = CFG["lr"],
        gamma          = CFG["gamma"],
        epsilon_start  = CFG["epsilon_start"],
        epsilon_end    = CFG["epsilon_end"],
        epsilon_decay  = CFG["epsilon_decay"],
        batch_size     = CFG["batch_size"],
        buffer_size    = CFG["buffer_size"],
        target_update  = CFG["target_update"],
        hidden1        = CFG["hidden1"],
        hidden2        = CFG["hidden2"],
        device         = device,
    )

    # --- bookkeeping ----------------------------------------------------- #
    episode_rewards: list[float] = []
    epsilons:        list[float] = []
    step_count = 0
    best_mean  = -np.inf
    t0         = time.time()

    # ================================================================= #
    #  Main training loop
    # ================================================================= #
    for episode in range(1, CFG["n_episodes"] + 1):

        state, _ = env.reset()
        done      = False
        ep_reward = 0.0
        ep_steps  = 0

        # ----- step loop -------------------------------------------- #
        while not done and ep_steps < CFG["max_steps"]:

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            # gradient update every `update_every` steps
            if (step_count >= CFG["warmup_steps"] and
                    step_count % CFG["update_every"] == 0):
                agent.update()

            state      = next_state
            ep_reward += reward
            ep_steps  += 1
            step_count += 1

        # ----- end of episode --------------------------------------- #
        agent.end_episode()

        episode_rewards.append(ep_reward)
        epsilons.append(agent.epsilon)

        # moving-average over last 100 episodes
        window     = min(100, len(episode_rewards))
        mean_100   = np.mean(episode_rewards[-window:])

        # ----- logging ----------------------------------------------- #
        if episode % 10 == 0 or episode <= 5:
            elapsed = time.time() - t0
            print(
                f"  Ep {episode:>4}/{CFG['n_episodes']} | "
                f"reward: {ep_reward:7.2f} | "
                f"mean100: {mean_100:7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"buf: {len(agent.buffer):>7} | "
                f"elapsed: {elapsed:.0f}s"
            )

        # ----- checkpoint -------------------------------------------- #
        if episode % CFG["checkpoint_every"] == 0:
            ckpt_path = os.path.join(
                CFG["checkpoint_dir"], f"dqn_ep{episode:04d}.pth"
            )
            agent.save(ckpt_path)

        # ----- save best model --------------------------------------- #
        if mean_100 > best_mean and len(episode_rewards) >= 100:
            best_mean = mean_100
            agent.save(CFG["model_path"])
            print(f"  *** New best model saved (mean100 = {best_mean:.2f}) ***")

        # ----- solved? ----------------------------------------------- #
        if mean_100 >= 200 and len(episode_rewards) >= 100:
            print(f"\n  ✔  Environment SOLVED at episode {episode}! "
                  f"Mean-100 = {mean_100:.2f}")

    env.close()

    # ================================================================= #
    #  Part C – plots
    # ================================================================= #
    print("\n[Part C] Saving training curves …")
    plot_training_curves(
        episode_rewards = episode_rewards,
        losses          = agent.losses,
        epsilons        = epsilons,
        avg_q_values    = agent.avg_q_values,
        save_path       = os.path.join(CFG["plot_dir"], "partC_training_curves.png"),
        window          = 50,
    )
    print_stats(episode_rewards, label="DQN Training (all episodes)")

    return agent


# ============================================================
# Part C – Final evaluation
# ============================================================

def run_evaluation(agent: DQNAgent | None = None) -> None:
    import gymnasium as gym
    """Evaluate the saved/given model for 100 greedy episodes.

    Records 5 videos of the learned behaviour.
    """
    print("\n" + "=" * 60)
    print("  PART C – Final Evaluation (100 greedy episodes)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        # load from file
        from dqn_agent import QNetwork
        import gymnasium as gym

        env_tmp    = gym.make(CFG["env_id"])
        state_dim  = env_tmp.observation_space.shape[0]
        action_dim = env_tmp.action_space.n
        env_tmp.close()

        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        agent.load(CFG["model_path"])

    # video recording for test episodes
    eval_env = gym.make(CFG["env_id"], render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=os.path.join(CFG["video_dir"], "test"),
        episode_trigger=lambda ep: ep < 5,      # first 5 test episodes
        name_prefix="dqn_test",
    )
    eval_env.reset(seed=999)

    test_rewards = evaluate_agent(
        agent,
        eval_env,
        n_episodes=CFG["eval_episodes"],
    )
    eval_env.close()

    print_stats(test_rewards, label="DQN Evaluation (100 greedy episodes)")

    # --- final plot ------------------------------------------------------ #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CFG["plot_dir"], exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(test_rewards, alpha=0.6, color="#4C9BE8", linewidth=0.9,
            label="Test episode reward")
    ax.axhline(np.mean(test_rewards), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(test_rewards):.1f}")
    ax.axhline(200, color="red", linestyle="--", linewidth=1.2, label="Solved (200)")
    ax.set_xlabel("Test episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Part C – DQN Test Performance (100 greedy episodes)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["plot_dir"], "partC_test_rewards.png"), dpi=150)
    plt.close()
    print(f"[Part C] Test plot saved → {CFG['plot_dir']}/partC_test_rewards.png")
    print(f"[Part C] Test videos saved → {CFG['video_dir']}/test/")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LunarLander DQN – CSE3008 Assignment 1"
    )
    parser.add_argument(
        "--part", choices=["a", "b", "c", "all"], default="all",
        help="Which part to run (default: all)"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate a saved model (loads from CFG['model_path'])"
    )
    args = parser.parse_args()

    if args.eval:
        run_evaluation()
        return

    if args.part in ("a", "all"):
        run_part_a()

    if args.part in ("b", "all"):
        agent = run_training()
        run_evaluation(agent)

    if args.part == "c":
        run_evaluation()


if __name__ == "__main__":
    main()
