# Assignment 1: Lunar Lander with Deep Q-Learning

## Overview

In this assignment, you will build and evaluate a **Deep Q-Network (DQN)** agent for the **LunarLander-v3** environment. Your goal is to train an agent that can safely land a spacecraft between the two flags while using thrust efficiently.


## Problem Description

The LunarLander environment simulates a spacecraft descending onto the moon's surface. The lander begins near the top-center of the screen with a small random initial force applied. It must learn how to control its descent, maintain a stable orientation, and land safely on the designated landing pad.

This assignment uses the **Gymnasium** implementation of LunarLander-v3, which is part of the actively maintained Gymnasium project.

## Environment Details

The spacecraft is equipped with:
- one **main engine** for upward thrust,
- two **side engines** for rotation and lateral adjustment.

The lander is affected by lunar gravity and must reach the landing zone centered at coordinate `(0, 0)` without crashing. Landing outside the marked pad is possible, but it is typically less rewarding.

### State Space

The state is an 8-dimensional vector containing:
1. horizontal position `x`
2. vertical position `y`
3. horizontal velocity `vx`
4. vertical velocity `vy`
5. angle
6. angular velocity
7. left leg contact indicator
8. right leg contact indicator

### Action Space

The action space is discrete with 4 possible actions:
- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine
- **3**: Fire right orientation engine

### Reward Structure

The environment provides dense rewards to encourage good landing behavior:
- small positive or negative rewards for moving toward or away from the landing pad,
- **-100** for crashing,
- **+100** for a successful landing / coming to rest,
- **+10** for each leg making contact with the ground,
- **-0.3** per frame for firing the main engine,
- **-0.03** per frame for firing a side engine.

### Solved Criterion

The environment is considered solved when the agent achieves an **average reward of at least 200 over 100 consecutive episodes**.

---

## Starter Code

A starter template is provided in `lunar_lander.py`. In addition, `utils.py` includes helper functions that may be useful for completing the assignment.

## Recording Videos

The following example shows how to create the environment and record videos. The `episode_trigger` setting records only every 50th episode to reduce disk usage during long training runs.

```python
# record videos
env = gym.make('LunarLander-v3', render_mode='rgb_array')
env = RecordVideo(env, 'videos/', episode_trigger=lambda x: x % 50 == 0)
```
