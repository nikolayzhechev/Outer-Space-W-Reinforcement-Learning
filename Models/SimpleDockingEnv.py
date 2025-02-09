import gym
from gym import spaces
import numpy as np

class SimpleDockingEnv(gym.Env):
    def __init__(self):
        super(SimpleDockingEnv, self).__init__()
        
        # State space: [x, y, vx, vy]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -5, -5], dtype=np.float32),
            high=np.array([10, 10, 5, 5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [thrust_x, thrust_y]
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Target position (docking port)
        self.target_position = np.array([0, 0], dtype=np.float32)
        
        # Episode length
        self.max_steps = 500
        self.current_step = 0
        
        # Physical constants
        self.dt = 0.1  # Time step
        self.mass = 1.0  # Spacecraft mass
        
        # Store the last action for fuel penalty
        self.last_action = np.array([0, 0], dtype=np.float32)
        
    def reset(self):
        # Initialize with random position and velocity near target
        self.state = np.concatenate([
            np.random.uniform(-5, 5, 2),  # Position
            np.random.uniform(-1, 1, 2)   # Velocity
        ])
        self.current_step = 0
        self.last_action = np.array([0, 0], dtype=np.float32)
        return self.state
    
    def step(self, action):
        self.current_step += 1
        
        # Store the last action for fuel penalty
        self.last_action = action
        
        # Unpack state
        position = self.state[:2]
        velocity = self.state[2:4]
        
        # Apply forces (actions are thrusts)
        acceleration = action / self.mass
        new_velocity = velocity + acceleration * self.dt
        new_position = position + new_velocity * self.dt
        
        # Update state
        self.state = np.concatenate([new_position, new_velocity])
        
        # Calculate reward
        reward = self.calculate_reward(position, velocity, action)
        
        # Termination conditions
        done = False
        distance = np.linalg.norm(position - self.target_position)
        if distance < 0.1:  # Docking success
            done = True
        elif self.current_step >= self.max_steps:  # Timeout
            done = True
            
        return self.state, reward, done, {}

    def calculate_reward(self, position, velocity, action):
        # Distance to target
        distance = np.linalg.norm(position - self.target_position)
        
        # Velocity penalty
        velocity_penalty = np.linalg.norm(velocity)
        
        # Fuel penalty
        fuel_penalty = np.linalg.norm(action)
        
        # Shaped reward
        reward = (
            -distance * 0.1  # Penalize distance to target
            - velocity_penalty * 0.05  # Penalize high velocity
            - fuel_penalty * 0.01  # Penalize fuel use
        )
        
        # Success bonus
        if distance < 0.1:
            reward += 10  # Large bonus for docking
        
        return reward

    def render(self, mode='human'):
        # Simple visualization (optional)
        print(f"Step: {self.current_step}, State: {self.state}")