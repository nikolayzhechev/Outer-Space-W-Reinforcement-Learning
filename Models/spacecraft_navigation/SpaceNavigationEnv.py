import numpy as np
import random
import gym
from collections import deque

class SpaceNavigationEnv(gym.Env):
    def __init__(self):
        super(SpaceNavigationEnv, self).__init__()
        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 5  # [thrust left, thrust right, thrust up, thrust down, no action]
        self.fuel = 100  # Fuel constraint
        
        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.reset()
    
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        # Random target location between 8 and 12 for both x and y
        self.target = np.random.uniform(8, 12, size=2)
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
        
        self.fuel = max(0, self.fuel - (1 if action != 4 else 0))
        self.state[0] += self.state[2]
        self.state[1] += self.state[3]
        
        distance = np.linalg.norm(self.state[:2] - self.target)
        reward = -0.02  # Small step penalty
        done = distance < 1.0 or self.fuel <= 0
        
        if distance < 1.0:
            reward += 150  # Big reward for reaching the target
        else:
            reward += (10.0 - distance) * 0.7  # Reward for approaching the target
            reward -= np.abs(self.state[2]) * (0.02 * distance)  # Penalize excessive horizontal velocity
            reward -= np.abs(self.state[3]) * (0.02 * distance)  # Penalize excessive vertical velocity
            if self.fuel <= 0:
                reward -= 50  # Penalty for running out of fuel
            elif self.fuel > 50:
                reward += 10  # Bonus for conserving fuel
        
        # Clip rewards to avoid extreme values
        reward = np.clip(reward, -150, 150)
        return self.state, reward, done, {}