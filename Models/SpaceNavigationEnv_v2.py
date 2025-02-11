import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class MovingObstacle:
    def __init__(self, start_pos, velocity):
        self.position = np.array(start_pos, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)

    def update(self):
        self.position += self.velocity  # Move each timestep

class SpaceNavigationEnv_v2(gym.Env):
    def __init__(self):
        super(SpaceNavigationEnv_v2, self).__init__()
        
        # Define action space: 6 possible thrust directions (+x, -x, +y, -y, +z, -z)
        self.action_space = spaces.Discrete(6)
        
        # State space: (x, y, z, vx, vy, vz, fuel)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -1, -1, -1, 0], dtype=np.float32),
                                            high=np.array([10, 10, 10, 1, 1, 1, 100], dtype=np.float32),
                                            dtype=np.float32)
        
        self.goal = np.array([8, 8, 8], dtype=np.float32)
        
        # Add moving obstacles in the environment
        self.moving_obstacles = [MovingObstacle([5, 5, 5], [0.1, -0.1, 0])]

        self.reset()
        
    def reset(self, seed=None, options=None):
        self.state = np.array([0, 0, 0, 0, 0, 0, 100], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        x, y, z, vx, vy, vz, fuel = self.state
    
        if fuel <= 0:
            return self.state, -10, True, False, {}

        thrust_magnitude = 0.1  # Small force per action
        # Apply movement
        thrust_directions  = {
            0: np.array([1, 0, 0]),  # +x
            1: np.array([-1, 0, 0]), # -x
            2: np.array([0, 1, 0]),  # +y
            3: np.array([0, -1, 0]), # -y
            4: np.array([0, 0, 1]),  # +z
            5: np.array([0, 0, -1]), # -z
            6: np.array([0, 0, 0])   # Stay in place
        }

        # Compute old distance to goal
        old_dist = np.linalg.norm(np.array([x, y, z]) - self.goal)

        # Update velocity based on thrust
        action = int(action) # UPDATED TO INT =============!
        velocity = np.array([vx, vy, vz], dtype=np.float32) + thrust_directions[action] * thrust_magnitude
    
        # Update position using physics
        new_pos = np.array([x, y, z], dtype=np.float32) + velocity

        # Update moving obstacles
        for obs in self.moving_obstacles:
            obs.update()

        # Collision detection
        for obs in self.moving_obstacles:
            if np.linalg.norm(new_pos - obs.position) < 0.5:  # Collision if too close
                return self.state, -10, True, False, {}
        
        new_fuel = fuel - 1
    
        # Compute new distance to goal
        new_dist = np.linalg.norm(new_pos - self.goal)

        # Compute reward
        reward = self.compute_reward(old_dist, new_dist, new_fuel)
    
        # Update state
        self.state = np.concatenate([new_pos, velocity, [new_fuel]])
    
        # Determine if episode is done
        done = (reward == 10 or new_fuel <= 0)  # Goal reached or fuel empty

        return self.state, reward, done, False, {}
    
    def render(self):
        print(f"Agent Position: {self.state[:3]}, Fuel: {self.state[6]}")
    
    def close(self):
        pass

    def compute_reward(self, old_dist, new_dist, fuel):
        """
        Computes the reward based on the agent's progress.
        """
        # Penalize running out of fuel
        if fuel <= 0:
            return -10
        
        # Reward reaching the goal
        if new_dist < 0.5:  # Close enough to goal
            return 10  # Large positive reward for reaching the goal
        
        # Reward for moving closer to the goal
        distance_reward = (old_dist - new_dist) * 2  # Positive if closer, negative if farther

        # Small step penalty to encourage efficiency
        return distance_reward - 1  
