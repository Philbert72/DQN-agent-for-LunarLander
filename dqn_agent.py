"""
dqn_agent.py
============
Deep Q-Network (DQN) agent for LunarLander-v3
CSE3008 – Introduction to Reinforcement Learning
Assignment 1

Architecture
------------
* ReplayBuffer     – circular experience replay (Mnih et al., 2015)
* QNetwork        – MLP Q-function approximator
* DQNAgent        – ε-greedy policy, target network, soft/hard updates

References
----------
Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. Nature, 518, 529–533.
Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An
  Introduction (2nd ed.). MIT Press.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer that stores (s, a, r, s', done) tuples.

    Lecture connection
    ------------------
    Experience replay breaks the temporal correlation between consecutive
    transitions (discussed in Lecture 1 under i.i.d. assumptions for
    supervised learning) and allows data reuse, improving sample efficiency.
    The buffer stores transitions and returns uniformly sampled mini-batches,
    which stabilises the gradient updates to the Q-network.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------ #
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self._buffer.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------ #
    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        """Return a random mini-batch of transitions as stacked tensors.

        Returns
        -------
        states, actions, rewards, next_states, dones
            Each is a float/long tensor on *device* with shape (B, …).
        """
        transitions = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states      = torch.tensor(np.array(states),      dtype=torch.float32, device=device)
        actions     = torch.tensor(np.array(actions),     dtype=torch.long,    device=device)
        rewards     = torch.tensor(np.array(rewards),     dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones       = torch.tensor(np.array(dones),       dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Q-Network (neural function approximator)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Multi-layer perceptron that maps states to Q-values for all actions.

    Lecture connection
    ------------------
    The Q-network approximates the optimal action-value function q*(s, a)
    (Bellman optimality, Lecture 2).  Because the state space of
    LunarLander-v3 is continuous (8-dim), tabular Q-learning is infeasible;
    we use a neural network as a universal function approximator instead.

    Architecture: Linear → ReLU → Linear → ReLU → Linear
    Default widths (256, 256) work well for LunarLander.

    Parameters
    ----------
    state_dim  : dimensionality of the observation vector (8 for Lunar).
    action_dim : number of discrete actions (4 for Lunar).
    hidden1    : width of first hidden layer.
    hidden2    : width of second hidden layer.
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        hidden1:    int = 256,
        hidden2:    int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q(s, ·) for all actions given a batch of states."""
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent implementing the full Mnih et al. (2015) algorithm.

    Key design choices
    ------------------
    1. **Replay buffer** – stores past transitions; mini-batch sampling
       breaks temporal correlations (see ReplayBuffer).
    2. **Separate target network** – a frozen copy of the Q-network whose
       weights are synced periodically.  This prevents the 'moving target'
       problem where the bootstrapped TD target changes at every step.
    3. **ε-greedy exploration** – with probability ε, pick a random action;
       otherwise exploit argmax Q(s, a).  ε decays from ε_start to ε_end
       exponentially so the agent gradually shifts from exploration to
       exploitation (Lecture 1: Exploration vs Exploitation).
    4. **Huber loss** – less sensitive to outlier TD errors than MSE.

    Parameters
    ----------
    state_dim, action_dim : environment dimensions.
    lr            : Adam learning rate.
    gamma         : discount factor (relates to value-function definition
                    in Lecture 2: vπ(s) = E[∑ γ^t R_{t+1}]).
    epsilon_start : initial exploration probability.
    epsilon_end   : minimum exploration probability.
    epsilon_decay : multiplicative decay applied after each episode.
    batch_size    : number of transitions per gradient update.
    buffer_size   : replay buffer capacity.
    target_update : hard update the target network every N steps.
    hidden1/2     : Q-network hidden layer widths.
    device        : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        state_dim:      int,
        action_dim:     int,
        lr:             float = 5e-4,
        gamma:          float = 0.99,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.01,
        epsilon_decay:  float = 0.995,
        batch_size:     int   = 64,
        buffer_size:    int   = 100_000,
        target_update:  int   = 10,        # episodes between hard updates
        hidden1:        int   = 256,
        hidden2:        int   = 256,
        device:         str   = "cpu",
    ) -> None:

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        # --- networks ---------------------------------------------------- #
        self.q_net      = QNetwork(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self._hard_update_target()          # initialise target = online

        # --- optimiser & loss -------------------------------------------- #
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.HuberLoss()     # robust to large TD errors

        # --- replay buffer ----------------------------------------------- #
        self.buffer = ReplayBuffer(capacity=buffer_size)

        # --- counters ---------------------------------------------------- #
        self._step_count    = 0
        self._episode_count = 0

        # --- metrics (populated during training) ------------------------- #
        self.losses:        list[float] = []
        self.avg_q_values:  list[float] = []

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection.

        With probability ε choose a uniformly random action (exploration);
        otherwise choose argmax_a Q(s, a) (exploitation).
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------ #
    def update(self) -> float | None:
        """Sample a mini-batch and perform one gradient descent step.

        TD target (Lecture 2 – Bellman optimality equation)
        ----------------------------------------------------
        y_i = r_i + γ · max_{a'} Q_target(s'_i, a')    if not done
            = r_i                                         if done

        Loss = HuberLoss( Q_online(s_i, a_i),  y_i )

        Returns
        -------
        loss value (float) or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        # --- current Q-values -------------------------------------------- #
        # Q_online(s, a) for the taken action only
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- TD targets -------------------------------------------------- #
        with torch.no_grad():
            # Double-DQN style: select action with online net, evaluate with target
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            q_next       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            td_targets   = rewards + self.gamma * q_next * (1.0 - dones)

        # --- gradient step ----------------------------------------------- #
        loss = self.loss_fn(q_current, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._step_count += 1

        loss_val = loss.item()
        self.losses.append(loss_val)

        # track mean Q for monitoring
        self.avg_q_values.append(q_current.mean().item())

        return loss_val

    # ------------------------------------------------------------------ #
    def end_episode(self) -> None:
        """Call once per episode: decay ε and optionally sync target net."""
        self._episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # hard target-network update every `target_update` episodes
        if self._episode_count % self.target_update == 0:
            self._hard_update_target()

    # ------------------------------------------------------------------ #
    # Target-network helpers
    # ------------------------------------------------------------------ #

    def _hard_update_target(self) -> None:
        """Copy online weights → target network (full replacement)."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save Q-network weights and agent hyper-parameters."""
        torch.save(
            {
                "q_net_state_dict":      self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict":  self.optimizer.state_dict(),
                "epsilon":               self.epsilon,
                "episode_count":         self._episode_count,
            },
            path,
        )
        print(f"[DQNAgent] Model saved → {path}")

    def load(self, path: str) -> None:
        """Load previously saved weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon        = checkpoint["epsilon"]
        self._episode_count = checkpoint["episode_count"]
        print(f"[DQNAgent] Model loaded ← {path}")
