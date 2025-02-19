import numpy as np
import random
import gym
from collections import deque

class SpaceNavigationEnv(gym.Env):
    def __init__(self):
        super(SpaceNavigationEnv, self).__init__()
        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 5  # [thrust left, thrust right, thrust up, thrust down, no action]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, vx, vy]
        self.target = np.array([10.0, 10.0])
        self.fuel = 100  # Fuel constraint
        
        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.fuel = 100
        return self.state

    def step(self, action):
        thrust_force = 0.1 if self.fuel > 0 else 0.0
        if action == 0:  # Left
            self.state[2] -= thrust_force
        elif action == 1:  # Right
            self.state[2] += thrust_force
        elif action == 2:  # Up
            self.state[3] += thrust_force
        elif action == 3:  # Down
            self.state[3] -= thrust_force
        
        self.fuel = max(0, self.fuel - (1 if action != 4 else 0))  # Ensure fuel doesn't go negative
        self.state[0] += self.state[2]  # Update x
        self.state[1] += self.state[3]  # Update y
        
        distance = np.linalg.norm(self.state[:2] - self.target)
        reward = -0.02  # Further reduced step penalty
        done = distance < 1.0 or self.fuel <= 0  # Stop when reaching target or fuel runs out
        
        if distance < 1.0:
            reward += 150  # Reward for reaching the target
        else:
            reward += (10.0 - distance) * 0.7  # Encourage moving closer
            reward -= np.abs(self.state[2]) * (0.02 * distance)  # Reduced velocity penalty
            reward -= np.abs(self.state[3]) * (0.02 * distance)
            if self.fuel <= 0:
                reward -= 50  # Lower penalty for fuel depletion
            elif self.fuel > 50:
                reward += 10  # Reward for conserving fuel
        
        return self.state, reward, done, {}