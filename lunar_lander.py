import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

from utils import (
    record_episodes,
    make_env_with_video,
    print_stats,
    plot_baseline,
    plot_training_curves,
    save_checkpoint,
    load_checkpoint,
)

# Hyperparameters
LEARNING_RATE = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 1000
TARGET_UPDATE_FREQ = 10

WARMUP_STEPS = 1_000
UPDATE_EVERY = 4

# Create environment
env = gym.make('LunarLander-v3')
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")



# Part B – Class implementations
# Replay Buffer
class ReplayBuffer:

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# Q-Network
class QNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden1: int = 256, hidden2: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# DQN Agent
class DQNAgent:

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimiser and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.HuberLoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Step counter (for UPDATE_EVERY logic)
        self._steps = 0

    # Action selection
    def select_action(self, state: np.ndarray, epsilon: float) -> int:

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax(dim=1).item())

    # Learning step
    def learn(self) -> float | None:
        
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Q(s, a) for the taken action
        q_current = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # TD target: r + γ · max_{a'} Q_target(s', a')
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            q_next = self.target_network(next_states_t).gather(1, next_actions).squeeze(1)
            td_targets = rewards_t + GAMMA * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_current, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # Target network update
    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Mean max Q (monitoring)
    def mean_max_q(self, states: np.ndarray) -> float:
        # Return the average of max Q-values for a batch of states.
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_vals = self.q_network(states_t).max(dim=1).values
        return float(q_vals.mean().item())


# Part A – Random-policy baseline
def run_random_baseline(num_episodes: int = 100) -> dict:

    print("\n" + "=" * 50)
    print("  Part A – Random Policy Baseline")
    print("=" * 50)

    baseline_env = gym.make('LunarLander-v3')
    rewards_list = []
    lengths_list = []

    for ep in range(num_episodes):
        state, _  = baseline_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = baseline_env.action_space.sample()
            state, reward, terminated, truncated, _  = baseline_env.step(action)
            done  = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

        rewards_list.append(ep_reward)
        lengths_list.append(ep_steps)

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:>3}/{num_episodes} | reward: {ep_reward:7.2f}")

    baseline_env.close()

    stats = {
        'episode_rewards': rewards_list,
        'episode_lengths': lengths_list,
        'mean_reward': float(np.mean(rewards_list)),
        'std_reward': float(np.std(rewards_list)),
        'min_reward': float(np.min(rewards_list)),
        'max_reward': float(np.max(rewards_list)),
        'mean_length': float(np.mean(lengths_list)),
        'success_rate': float(np.mean(np.array(rewards_list) >= 200)),
    }

    print_stats(stats)

    # Save baseline plot
    plot_baseline(stats, out_path="outputs/part_a/baseline_stats.png")

    # Record GIF
    print("\n  Recording 5 random-agent episodes …")
    record_episodes(
        num_episodes = 5,
        out_dir = "outputs/part_a/gifs",
        policy_fn = lambda s: baseline_env.action_space.sample()
        
        if hasattr(baseline_env, 'action_space')
        else random.randint(0, action_dim - 1),
    )

    return stats


# DQN training loop
def train(num_episodes: int = 1000) -> dict:
    
    print("\n" + "=" * 50)
    print(" DQN Training")
    print("=" * 50)

    os.makedirs("outputs/part_b_c", exist_ok=True)
    os.makedirs("checkpoints",      exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # Environment with video recorded every 50 episodes
    train_env = make_env_with_video(
        video_dir    = 'outputs/part_b_c/videos/',
        record_every = 50,
    )
    train_env.reset(seed=42)
    torch.manual_seed(42)
    np.random.seed(42)

    agent = DQNAgent(state_dim, action_dim)

    # Metrics
    episode_rewards: list[float] = []
    avg_losses: list[float] = []
    epsilons: list[float] = []
    mean_q_values: list[float] = []
    solved_at = None

    epsilon = EPSILON_START
    step_count = 0
    best_mean = -np.inf

    # Main training loop
    for episode in range(num_episodes):
        state, _  = train_env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []
        ep_states = []

        # Step loop
        while not done:

            # Select action using epsilon-greedy
            action = agent.select_action(state, epsilon)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            ep_states.append(state)

            # Train agent if buffer has enough samples
            if (step_count >= WARMUP_STEPS and
                    step_count % UPDATE_EVERY == 0):
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)

            # Update target network periodically (done per episode below)
            state = next_state
            ep_reward += reward
            step_count += 1

        # End-of-episode bookkeeping

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update target network every TARGET_UPDATE_FREQ episodes
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Track and log statistics
        episode_rewards.append(ep_reward)
        epsilons.append(epsilon)
        avg_losses.append(float(np.mean(ep_losses)) if ep_losses else float('nan'))

        # Mean max Q-value over states visited this episode
        if ep_states:
            mq = agent.mean_max_q(np.array(ep_states[:64], dtype=np.float32))
        else:
            mq = float('nan')
        mean_q_values.append(mq)

        # Moving average over last 100 episodes
        window   = min(100, len(episode_rewards))
        mean_100 = float(np.mean(episode_rewards[-window:]))

        # Logging
        if (episode + 1) % 50 == 0 or episode < 5:
            print(f"  Episode {episode+1:>4}/{num_episodes} | "
                  f"reward: {ep_reward:7.2f} | mean100: {mean_100:7.2f} | "
                  f"ε: {epsilon:.4f} | buf: {len(agent.replay_buffer):>7}")

        # Checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            ckpt = f"checkpoints/dqn_ep{episode+1:04d}.pt"
            save_checkpoint(agent, episode + 1, episode_rewards, ckpt)

        # Save best model
        if mean_100 > best_mean and len(episode_rewards) >= 100:
            best_mean = mean_100
            save_checkpoint(agent, episode + 1, episode_rewards,
                            "outputs/part_b_c/best_model.pt")
            print(f"  *** Best model saved  (mean100 = {best_mean:.2f}) ***")

        # Check solved criterion
        if mean_100 >= 200 and len(episode_rewards) >= 100 and solved_at is None:
            solved_at = episode + 1
            print(f"\n  ✔  Environment SOLVED at episode {solved_at}! "
                  f"mean100 = {mean_100:.2f}\n")

    train_env.close()

    # Metrics dict for plotting
    metrics = {
        'episode_rewards': episode_rewards,
        'avg_losses': avg_losses,
        'epsilons': epsilons,
        'mean_q_values': mean_q_values,
        'solved_at': solved_at,
    }

    # Save training curves
    plot_training_curves(metrics, out_dir="outputs/part_b_c")
    return metrics, agent


# Part C – Testing the trained agent
def test(agent: DQNAgent, num_episodes: int = 100) -> dict:
    # Evaluate the trained agent with a fully greedy policy (ε = 0).
    print("\n" + "=" * 50)
    print("  Part C – Testing Trained Agent (100 greedy episodes)")
    print("=" * 50)

    test_env = gym.make('LunarLander-v3')
    rewards_list = []
    lengths_list = []

    for ep in range(num_episodes):
        state, _  = test_env.reset()
        done = False
        ep_reward  = 0.0
        ep_steps = 0

        while not done:
            # Greedy action (epsilon = 0)
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _  = test_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

        rewards_list.append(ep_reward)
        lengths_list.append(ep_steps)

        if (ep + 1) % 20 == 0:
            print(f"  Test episode {ep+1:>3}/{num_episodes} | reward: {ep_reward:7.2f}")

    test_env.close()

    stats = {
        'episode_rewards': rewards_list,
        'episode_lengths': lengths_list,
        'mean_reward': float(np.mean(rewards_list)),
        'std_reward': float(np.std(rewards_list)),
        'min_reward': float(np.min(rewards_list)),
        'max_reward': float(np.max(rewards_list)),
        'mean_length': float(np.mean(lengths_list)),
        'success_rate': float(np.mean(np.array(rewards_list) >= 200)),
    }

    print_stats(stats)

    # Record 5 GIF episodes of the trained agent
    print("\n  Recording 5 trained-agent episodes …")
    record_episodes(
        num_episodes = 5,
        out_dir = "outputs/part_c/gifs",
        policy_fn = lambda s: agent.select_action(s, epsilon=0.0),
    )

    # Simple test reward plot
    os.makedirs("outputs/part_c", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards_list, alpha=0.6, color='steelblue', linewidth=0.9,
            label='Test episode reward')
    ax.axhline(stats['mean_reward'], color='orange', linestyle='--', linewidth=1.5,
               label=f"Mean = {stats['mean_reward']:.1f}")
    ax.axhline(200, color='red', linestyle='--', linewidth=1.2, label='Solved (200)')
    ax.set_xlabel('Test episode')
    ax.set_ylabel('Total reward')
    ax.set_title('Part C – DQN Test Performance (100 greedy episodes)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/part_c/test_rewards.png", dpi=150)
    plt.close()
    print("  Saved test reward plot → outputs/part_c/test_rewards.png")

    return stats


# Part D – Hyperparameter experimentation

def run_sweep(param_name: str, param_key: str, param_values: dict, num_episodes: int = 500) -> None:
    
    os.makedirs("outputs/part_d", exist_ok=True)

    all_rewards: dict[str, list[float]] = {}

    for label, value in param_values.items():
        print(f"\n  [{param_name}] Running: {label}")

        # Fresh environment (no video to save time)
        sweep_env = gym.make('LunarLander-v3')
        sweep_env.reset(seed=42)
        torch.manual_seed(42)
        np.random.seed(42)

        # Build agent with overridden hyperparameter
        sweep_agent  = DQNAgent(state_dim, action_dim)

        # Override the relevant hyperparameter inside the agent / loop
        if param_key == 'lr':
            for pg in sweep_agent.optimizer.param_groups:
                pg['lr'] = value
        # epsilon_decay and target_update are handled inside the loop below

        ep_rewards = []
        eps = EPSILON_START
        step_count = 0

        for episode in range(num_episodes):
            state, _ = sweep_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = sweep_agent.select_action(state, eps)
                next_state, reward, terminated, truncated, _ = sweep_env.step(action)
                done = terminated or truncated
                sweep_agent.replay_buffer.push(state, action, reward, next_state, done)

                if step_count >= WARMUP_STEPS and step_count % UPDATE_EVERY == 0:
                    sweep_agent.learn()

                state = next_state
                ep_reward += reward
                step_count += 1

            # Decay epsilon using the sweep value or default
            decay = value if param_key == 'epsilon_decay' else EPSILON_DECAY
            eps   = max(EPSILON_END, eps * decay)

            # Update target network using sweep value or default
            freq = value if param_key == 'target_update' else TARGET_UPDATE_FREQ
            if (episode + 1) % freq == 0:
                sweep_agent.update_target_network()

            ep_rewards.append(ep_reward)

            if (episode + 1) % 100 == 0:
                mean100 = np.mean(ep_rewards[-100:])
                print(f"    Ep {episode+1:>4} | mean100: {mean100:7.2f} | ε: {eps:.4f}")

        sweep_env.close()
        all_rewards[label] = ep_rewards
        print(f"  [{label}] Final mean-100: {np.mean(ep_rewards[-100:]):.2f}")

    # Comparison plot
    window = 20
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(all_rewards)))

    for (label, rewards), colour in zip(all_rewards.items(), colours):
        episodes = np.arange(1, len(rewards) + 1)
        # raw (faint)
        ax.plot(episodes, rewards, alpha=0.15, color=colour, linewidth=0.7)
        # smoothed
        if len(rewards) >= window:
            from utils import moving_average
            ma = moving_average(rewards, window)
            ax.plot(episodes[window - 1:], ma, color=colour,
                    linewidth=2.2, label=label)

    ax.axhline(200, color='red', linestyle='--', linewidth=1.2, label='Solved (200)')
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Reward (MA-{window})')
    ax.set_title(f'Part D – Hyperparameter Comparison: {param_name}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = param_name.lower().replace(' ', '_')
    out_path = f"outputs/part_d/{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Saved comparison plot → {out_path}")


def run_part_d() -> None:

    print("\n" + "=" * 50)
    print("  Part D – Hyperparameter Experimentation")
    print("=" * 50)

    # Experiment 1: Epsilon decay rate
    run_sweep(
        param_name = "Epsilon Decay Rate",
        param_key = "epsilon_decay",
        param_values = {
            "Fast  (0.980)":    0.980,
            "Baseline (0.995)": 0.995,
            "Slow  (0.999)":    0.999,
        },
    )

    # Experiment 2: Target network update frequency
    run_sweep(
        param_name = "Target Network Update Frequency",
        param_key = "target_update",
        param_values = {
            "Very frequent (1 ep)":  1,
            "Baseline (10 ep)": 10,
            "Infrequent (50 ep)": 50,
        },
    )

    # Experiment 3: Learning rate
    run_sweep(
        param_name   = "Learning Rate",
        param_key    = "lr",
        param_values = {
            "High (5e-3)": 5e-3,
            "Baseline (5e-4)": 5e-4,
            "Low (5e-5)": 5e-5,
        },
    )

    print("\n[Part D] All experiments done! Plots saved → outputs/part_d/")


# Run Parts A, B, C, D in sequence
if __name__ == '__main__':

    # Random baseline
    baseline_stats = run_random_baseline(num_episodes=100)

    # Train DQN
    metrics, agent = train(num_episodes=1000)

    # Test the trained agent
    test_stats = test(agent, num_episodes=100)

    # Hyperparameter sweep 
    run_part_d()

    print("\nRunning Finished...")
    env.close()