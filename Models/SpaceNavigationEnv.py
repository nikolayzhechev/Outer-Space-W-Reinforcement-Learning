import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SpaceNavigationEnv(gym.Env):
    def __init__(self):
        super(SpaceNavigationEnv, self).__init__()
        
        # Define action space: 6 possible thrust directions (+x, -x, +y, -y, +z, -z)
        self.action_space = spaces.Discrete(6)
        
        # State space: (x, y, z, vx, vy, vz, fuel)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -1, -1, -1, 0]),
                                            high=np.array([10, 10, 10, 1, 1, 1, 100]),
                                            dtype=np.float32)
        
        self.goal = np.array([8, 8, 8])
        self.obstacles = [np.array([5, 5, 5]), np.array([3, 3, 3])]
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.state = np.array([0, 0, 0, 0, 0, 0, 100], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        x, y, z, vx, vy, vz, fuel = self.state
    
        if fuel <= 0:
            return self.state, -10, True, False, {}
    
        # Apply movement
        movement = {
            0: np.array([1, 0, 0]),  # +x
            1: np.array([-1, 0, 0]), # -x
            2: np.array([0, 1, 0]),  # +y
            3: np.array([0, -1, 0]), # -y
            4: np.array([0, 0, 1]),  # +z
            5: np.array([0, 0, -1]), # -z
            6: np.array([0, 0, 0])   # Stay in place
        }
    
        new_pos = np.array([x, y, z]) + movement[int(action)]
        new_fuel = fuel - 1
    
        reward = self.compute_reward(new_pos, new_fuel)
    
        # Update state
        self.state = np.append(new_pos, [vx, vy, vz, new_fuel])
    
        # Determine if episode is done
        done = (reward == 10 or reward == -10 or reward == -5)  # Goal reached, fuel empty, or collision
    
        return self.state, reward, done, False, {}
    
    def render(self):
        print(f"Agent Position: {self.state[:3]}, Fuel: {self.state[6]}")
    
    def close(self):
        pass

    def compute_reward(self, new_pos, fuel):
        # Penalize running out of fuel
        if fuel <= 0:
            return -10
        
        # Reward reaching the goal
        if np.all(new_pos == self.goal):
            return 10  # Large positive reward for reaching the goal
        
        # Penalize hitting obstacles
        for obs in self.obstacles:
            if np.all(new_pos == obs):
                return -5  # Large penalty for collision
        
        # Reward based on distance to the goal
        old_dist = np.linalg.norm(self.state[:3] - self.goal)  # Distance before moving
        new_dist = np.linalg.norm(new_pos - self.goal)  # Distance after moving
        
        distance_reward = (old_dist - new_dist) * 2  # Positive reward if closer, negative if farther
        
        return distance_reward - 1  # Small step penalty to encourage efficiency
