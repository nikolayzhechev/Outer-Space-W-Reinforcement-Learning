import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

class SpacecraftDockingEnv(gym.Env):
    """A custom environment for spacecraft docking using Gymnasium."""
    
    def __init__(self, config=None):
        super(SpacecraftDockingEnv, self).__init__()
        
        '''
        # Normalize observation space values
        self.state = np.array([
            np.clip(position, -500, 500), 
            np.clip(velocity, -10, 10), 
            np.clip(orientation, -1, 1),
            np.clip(fuel, 0, 1)
        ], dtype=np.float32)

        
        # Define the state space: [x, y, z, vx, vy, vz, qx, qy, qz, qw, fuel]
        self.observation_space = spaces.Box(
            low=np.array([-1e3, -1e3, -1e3, -10, -10, -10, -1, -1, -1, -1, 0]),
            high=np.array([1e3, 1e3, 1e3, 10, 10, 10, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        '''
        # Udpated observation space
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=-1e3, high=1e3, shape=(3,), dtype=np.float32),
            'velocity': spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'orientation': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            'fuel': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        # Define the action space: Thrust in X, Y, Z (-1 to 1)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        # Docking target (position, velocity, orientation)
        self.target_position = np.array([0, 0, 0])  # Docking port location
        self.target_velocity = np.array([0, 0, 0])  # Station is static
        self.target_orientation = np.array([0, 0, 0, 1])  # Quaternion (no rotation)

        # Environment reset
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0  # Track steps
        self.max_steps = 10000  # End episode
        self.state = np.concatenate([
            np.random.uniform(-500, 500, size=3),  # Random position
            np.random.uniform(-1, 1, size=3),  # Random velocity
            R.random().as_quat(),  # Random orientation (quaternion)
            [np.random.uniform(0.7, 1.0)]  # Random fuel level
        ])
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        """Updates the state based on the action taken by the agent."""
        self.step_count += 1
        
        x, y, z, vx, vy, vz, qx, qy, qz, qw, fuel = self.state
        thrust = np.clip(action, -1, 1)  # Limit thrust values
    
        # Apply thrust (scaled by max thrust)
        thrust = np.clip(action, -1, 1) * max_thrust
        vx += thrust[0] * 0.1
        vy += thrust[1] * 0.1
        vz += thrust[2] * 0.1
    
        # Update position
        x += vx
        y += vy
        z += vz
    
        # Fuel consumption
        # fuel = max(0, fuel - np.linalg.norm(thrust) * 0.01)
        # Update to non linear fuel consumption
        reward_fuel = -np.linalg.norm(thrust) * 0.01 * (fuel + 0.1)
    
        # Compute rewards
        reward = self.compute_reward(np.array([x, y, z]), np.array([vx, vy, vz]), np.array([qx, qy, qz, qw]), thrust, fuel)
    
        if np.linalg.norm([x, y, z] - self.target_position) < 0.1 and np.linalg.norm([vx, vy, vz]) < 0.1:
            done = True
            reward += 100  # Positive reward for reaching target
        if fuel <= 0.01:
            done = True
            reward -= 50  # Penalty for running out of fuel
        if self.step_count >= self.max_steps:
            done = True
            reward -= 10  # Penalty for max steps exceeded
        
        self.state = np.array([x, y, z, vx, vy, vz, qx, qy, qz, qw, fuel], dtype=np.float32)

        print(f"Step Reward: {reward}, Done: {done}")  # Debugging output
        return self.state, reward, done, False, {}
        
    # Updated reward
    def compute_reward(self, position, velocity, orientation, thrust, fuel):
        # Declare weights
        distance_weight = -2.0
        velocity_weight = -2.0
        fuel_weight = -0.01
        orientation_weight = -0.05
        
        # Distance to docking port
        reward_distance = -np.linalg.norm(position - self.target_position) * distance_weight  # Increased weight

        # Velocity matching
        reward_velocity = -np.linalg.norm(velocity - self.target_velocity) * velocity_weight  # Increased weight

        # Fuel efficiency
        reward_fuel = -np.linalg.norm(thrust) * fuel_weight  # Small penalty

        # Orientation alignment
        error_quat = R.from_quat(orientation).inv() * R.from_quat(self.target_orientation)
        reward_orientation = -np.linalg.norm(error_quat.as_rotvec()) * orientation_weight  # Reduced weight

        # Terminal rewards
        reward_goal = 100 if (np.linalg.norm(position - self.target_position) < 0.1 and np.linalg.norm(velocity) < 0.1) else 0
        reward_crash = -100 if np.linalg.norm(position) > 1000 else 0

        total_reward = reward_distance + reward_velocity + reward_fuel + reward_orientation + reward_goal + reward_crash
        return total_reward
    '''   
    def compute_reward(self, position, velocity, orientation, thrust, fuel):
        """Computes the reward based on multiple factors."""
        # Distance to docking port
        reward_distance = -np.linalg.norm(position - self.target_position) * 2.0  # Increased weight

        # Velocity matching
        reward_velocity = -np.linalg.norm(velocity - self.target_velocity) * 1.0  # Increased weight

        # Fuel efficiency
        reward_fuel = -np.linalg.norm(thrust) * 0.01  # Small penalty

        # Orientation alignment
        error_quat = R.from_quat(orientation).inv() * R.from_quat(self.target_orientation)
        reward_orientation = -np.linalg.norm(error_quat.as_rotvec()) * 0.1  # Reduced weight

        # Terminal rewards
        reward_goal = 100 if (np.linalg.norm(position - self.target_position) < 0.1 and np.linalg.norm(velocity) < 0.1) else 0
        reward_crash = -100 if np.linalg.norm(position) > 1000 else 0

        total_reward = reward_distance + reward_velocity + reward_fuel + reward_orientation + reward_goal + reward_crash
        return total_reward


    def compute_reward(self, position, velocity, orientation, thrust, fuel):
        """Computes the reward based on multiple factors."""
        # Distance to docking port
        reward_distance = -np.linalg.norm(position - self.target_position) * 1.0

        # Velocity matching
        reward_velocity = -np.linalg.norm(velocity - self.target_velocity) * 0.5

        # Fuel efficiency
        reward_fuel = -np.linalg.norm(thrust) * 0.01  # Penalize excessive thrust

        # Orientation alignment
        error_quat = R.from_quat(orientation).inv() * R.from_quat(self.target_orientation)
        reward_orientation = -np.linalg.norm(error_quat.as_rotvec()) * 0.3  # Penalize misalignment

        # Terminal rewards
        reward_goal = 100 if (np.linalg.norm(position - self.target_position) < 0.1 and np.linalg.norm(velocity) < 0.1) else 0
        reward_crash = -100 if np.linalg.norm(position) > 1000 else 0  # Penalize drifting too far

        total_reward = reward_distance + reward_velocity + reward_fuel + reward_orientation + reward_goal + reward_crash
        return total_reward
    '''