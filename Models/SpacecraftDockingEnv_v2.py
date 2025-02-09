import gym
from gym import spaces
import numpy as np

class SpacecraftDockingEnv_v2(gym.Env):
    def __init__(self):
        super(SpacecraftDockingEnv_v2, self).__init__()
        
        # State space: [x, y, z, vx, vy, vz, theta_x, theta_y, theta_z]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -10, -5, -5, -5, -np.pi, -np.pi, -np.pi]),
            high=np.array([10, 10, 10, 5, 5, 5, np.pi, np.pi, np.pi]),
            dtype=np.float32
        )
        
        # Action space: [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -0.5, -0.5, -0.5]),
            high=np.array([1, 1, 1, 0.5, 0.5, 0.5]),
            dtype=np.float32
        )
        
        self.target_position = np.array([0, 0, 0], dtype=np.float32)
        self.max_steps = 1000
        self.current_step = 0
        
    def reset(self):
        # Initialize with random position near target
        self.state = np.concatenate([
            np.random.uniform(-5, 5, 3),
            np.random.uniform(-1, 1, 3),
            np.random.uniform(-np.pi/4, np.pi/4, 3)
        ])
        self.current_step = 0
        return self.state
    
    def step(self, action):
        self.current_step += 1
        
        # Update state using simplified dynamics
        position = self.state[:3]
        velocity = self.state[3:6]
        orientation = self.state[6:9]
        
        # Apply forces and torques (simplified physics)
        new_velocity = velocity + action[:3] * 0.1
        new_position = position + new_velocity * 0.1
        new_orientation = orientation + action[3:] * 0.1
        
        # Update state
        self.state = np.concatenate([new_position, new_velocity, new_orientation])
        
        # Calculate reward
        distance = np.linalg.norm(position - self.target_position)
        alignment = np.linalg.norm(orientation)
        reward = (
            -distance * 0.1  # Penalize distance
            - np.linalg.norm(action) * 0.01  # Penalize fuel use
            - alignment * 0.05  # Penalize misalignment
        )
        
        # Termination conditions
        done = distance < 0.1 or self.current_step >= self.max_steps
        if distance < 0.1:
            reward += 100  # Successful docking bonus
            
        return self.state, reward, done, {}