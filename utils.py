import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# ── Video recording ───────────────────────────────────────────────────────────

def make_env_with_video(video_dir='videos/', record_every=50):
    """Create LunarLander environment with video recording every `record_every` episodes."""
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % record_every == 0)
    return env


def record_episodes(num_episodes: int, out_dir: str, policy_fn) -> None:
    """
    Run `num_episodes` using `policy_fn` and save each episode as a GIF.

    Parameters
    ----------
    num_episodes : int
        Number of episodes to record.
    out_dir : str
        Directory in which to save the GIF files.
    policy_fn : callable
        A function that accepts a state (np.ndarray) and returns an integer action.
        For the random baseline use: ``policy_fn=lambda s: env.action_space.sample()``

    Hints
    -----
    - Use ``gym.make('LunarLander-v3', render_mode='rgb_array')`` so that
      ``env.render()`` returns an RGB frame (numpy array).
    - Collect frames in a list during the episode loop, then write them with
      ``imageio.mimsave(path, frames, fps=30)``.
    - Name each file ``episode_{ep+1}.gif`` inside *out_dir*.
    """
    import imageio
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make('LunarLander-v3', render_mode='rgb_array')

    for ep in range(num_episodes):
        state, _ = env.reset()
        frames = []
        done = False
        total_reward = 0.0

        while not done:
            frames.append(env.render())
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        gif_path = os.path.join(out_dir, f'episode_{ep + 1}.gif')
        imageio.mimsave(gif_path, frames, fps=30)
        print(f'  Episode {ep + 1:2d}  reward = {total_reward:7.1f}  → {gif_path}')

    env.close()


# Statistics helper functions

def print_stats(stats: dict) -> None:
    """
    Print a formatted summary of episode statistics to the console.

    The *stats* dictionary is expected to contain at least:
        mean_reward, std_reward, min_reward, max_reward,
        mean_length, success_rate

    Expected output format (example)::

        ----------------------------------------
        Random Policy Statistics
        ----------------------------------------
          Mean reward  :   -123.45 ± 80.23
          Min / Max    :   -456.78 / 12.34
          Mean length  :    250.0 steps
          Success rate :      0.0%
        ----------------------------------------

    Hints
    -----
    - Use f-strings with format specs such as ``:.2f`` and ``:.1f``.
    - The success rate in *stats* is stored as a fraction (0–1); multiply by 100
      to display as a percentage.
    """
    sep = '-' * 40
    print(sep)
    print('Episode Statistics')
    print(sep)
    print(f"  Mean reward  : {stats['mean_reward']:>8.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min / Max    : {stats['min_reward']:>8.2f} / {stats['max_reward']:.2f}")
    print(f"  Mean length  : {stats['mean_length']:>8.1f} steps")
    print(f"  Success rate : {stats['success_rate'] * 100:>7.1f}%")
    print(sep)


# Plotting helper functions

def moving_average(data, window: int = 20) -> np.ndarray:
    """
    Compute a simple moving average using a uniform kernel.

    Parameters
    ----------
    data : array-like
        1-D sequence of values (e.g. per-episode rewards or losses).
    window : int
        Number of elements to average over.

    Returns
    -------
    np.ndarray
        Array of length ``len(data) - window + 1`` containing the smoothed
        values.  The first element is the mean of ``data[0:window]``.

    Hints
    -----
    - ``np.convolve(data, kernel, mode='valid')`` with a uniform kernel of
      length *window* is a one-liner solution.
    - Remember to normalise the kernel so its values sum to 1.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_baseline(stats: dict, out_path: str = "outputs/part_a/baseline_stats.png") -> None:
    """
    Generate a 2×2 figure summarising the random-policy baseline and save it.

    The four sub-plots should be:
        [0,0] Episode reward over time (line plot + mean horizontal line)
        [0,1] Reward distribution (histogram + mean vertical line)
        [1,0] Episode length over time (line plot + mean horizontal line)
        [1,1] Text summary box (mean, std, min, max, mean length, success rate)

    Parameters
    ----------
    stats : dict
        Dictionary returned by ``run_random_baseline()``; must contain
        ``episode_rewards``, ``episode_lengths``, ``mean_reward``,
        ``std_reward``, ``min_reward``, ``max_reward``, ``mean_length``,
        ``success_rate``.
    out_path : str
        File path (PNG) where the figure is saved.

    Hints
    -----
    - Use ``plt.subplots(2, 2, figsize=(12, 8))``.
    - For the text panel use ``axes[1,1].axis('off')`` then
      ``axes[1,1].text(...)``.  A monospace font and a ``bbox`` with
      ``boxstyle='round'`` look clean.
    - Call ``plt.tight_layout()`` before saving.
    - Use ``plt.close()`` after saving to free memory.
    """
    rewards  = stats['episode_rewards']
    lengths  = stats['episode_lengths']
    episodes = range(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Part A – Random Policy Baseline', fontsize=14)

    # [0,0] Reward over time
    axes[0, 0].plot(episodes, rewards, alpha=0.6, color='steelblue')
    axes[0, 0].axhline(stats['mean_reward'], color='red', linestyle='--',
                       label=f"Mean = {stats['mean_reward']:.1f}")
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # [0,1] Reward distribution
    axes[0, 1].hist(rewards, bins=20, color='steelblue', edgecolor='white')
    axes[0, 1].axvline(stats['mean_reward'], color='red', linestyle='--',
                       label=f"Mean = {stats['mean_reward']:.1f}")
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].set_xlabel('Total Reward')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # [1,0] Episode lengths
    axes[1, 0].plot(episodes, lengths, alpha=0.6, color='darkorange')
    axes[1, 0].axhline(stats['mean_length'], color='red', linestyle='--',
                       label=f"Mean = {stats['mean_length']:.1f}")
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # [1,1] Summary text box
    summary = (
        f"Random Policy Statistics\n"
        f"{'─' * 30}\n"
        f"  Mean reward  : {stats['mean_reward']:>8.2f}\n"
        f"  Std  reward  : {stats['std_reward']:>8.2f}\n"
        f"  Min  reward  : {stats['min_reward']:>8.2f}\n"
        f"  Max  reward  : {stats['max_reward']:>8.2f}\n"
        f"  Mean length  : {stats['mean_length']:>8.1f} steps\n"
        f"  Success rate : {stats['success_rate'] * 100:>7.1f}%"
    )
    axes[1, 1].axis('off')
    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Saved baseline plot → {out_path}')
    plt.close()


def plot_training_curves(metrics: dict, out_dir: str = "outputs/part_b_c") -> None:
    """
    Generate a 2×2 figure of DQN training diagnostics and save it.

    The four sub-plots should be:
        [0,0] Episode reward + moving average + solved threshold line
        [0,1] Training loss (Huber) + moving average  (skip NaN episodes)
        [1,0] Epsilon decay over episodes
        [1,1] Mean max Q-value + moving average       (skip NaN episodes)

    Parameters
    ----------
    metrics : dict
        Dictionary returned by the ``train()`` function; must contain
        ``episode_rewards``, ``avg_losses``, ``epsilons``, ``mean_q_values``,
        and ``solved_at`` (int or None).
    out_dir : str
        Directory in which to save ``training_curves.png``.

    Hints
    -----
    - Use your ``moving_average()`` helper for smoothing.
    - Loss and Q-value lists may contain ``float('nan')`` for early episodes
      before the buffer is warm; filter these out before plotting.
    - If ``solved_at`` is not None, draw a vertical dashed line on the reward
      sub-plot to mark the episode where the environment was solved.
    - Save to ``os.path.join(out_dir, 'training_curves.png')``.
    """
    SOLVED_THRESHOLD = 200.0
    window   = 20
    rewards  = metrics['episode_rewards']
    losses   = metrics['avg_losses']
    epsilons = metrics['epsilons']
    q_values = metrics['mean_q_values']
    solved_at = metrics.get('solved_at')
    N        = len(rewards)
    episodes = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('DQN Training – LunarLander-v3', fontsize=14)

    # [0,0] Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color='steelblue', label='per episode')
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        ax.plot(episodes[window - 1:], ma, color='steelblue', linewidth=2,
                label=f'MA-{window}')
    ax.axhline(SOLVED_THRESHOLD, color='green', linestyle='--', alpha=0.7,
               label=f'Solved ({SOLVED_THRESHOLD})')
    if solved_at:
        ax.axvline(solved_at, color='red', linestyle=':', alpha=0.8,
                   label=f'Solved @ ep {solved_at}')
    ax.set_title('Episode Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] Loss curve (filter NaNs)
    ax = axes[0, 1]
    valid_loss = [(i + 1, l) for i, l in enumerate(losses) if not np.isnan(l)]
    if valid_loss:
        ep_l, l_vals = zip(*valid_loss)
        ax.plot(ep_l, l_vals, alpha=0.3, color='darkorange', label='per episode')
        if len(l_vals) >= window:
            ma_l = moving_average(l_vals, window)
            ax.plot(list(ep_l)[window - 1:], ma_l, color='darkorange',
                    linewidth=2, label=f'MA-{window}')
    ax.set_title('Training Loss (Huber)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] Epsilon decay
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color='purple', linewidth=2)
    ax.set_title('Epsilon Decay')
    ax.set_xlabel('Episode')
    ax.set_ylabel('ε (exploration prob.)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # [1,1] Mean Q-values (filter NaNs)
    ax = axes[1, 1]
    valid_q = [(i + 1, q) for i, q in enumerate(q_values) if not np.isnan(q)]
    if valid_q:
        ep_q, q_vals = zip(*valid_q)
        ax.plot(ep_q, q_vals, alpha=0.3, color='crimson', label='per episode')
        if len(q_vals) >= window:
            ma_q = moving_average(q_vals, window)
            ax.plot(list(ep_q)[window - 1:], ma_q, color='crimson',
                    linewidth=2, label=f'MA-{window}')
    ax.set_title('Mean Max Q-Value')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q̄')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    print(f'Saved training curves → {path}')
    plt.close()


# Checkpoint helper functions

def save_checkpoint(agent, episode: int, rewards: list, filename: str) -> None:
    """Save agent weights, optimiser state, and reward history to a .pt file."""
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'rewards_history': rewards,
    }, filename)


def load_checkpoint(agent, filename: str):
    """
    Load a checkpoint saved by ``save_checkpoint`` into *agent*.

    Returns
    -------
    (episode, rewards_history) : (int, list)
    """
    checkpoint = torch.load(filename)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['rewards_history']