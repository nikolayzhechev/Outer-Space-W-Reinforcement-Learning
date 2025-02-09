import gymnasium as gym
import numpy as np
from gymnasium import spaces
from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit
from poliastro.maneuver import Maneuver
from astropy import units as u
from scipy.spatial.transform import Rotation as R

class SpacecraftOrbitalEnv(gym.Env):
    """Custom Gymnasium environment for spacecraft navigation in orbit using poliastro."""

    def __init__(self, config=None):
        super(SpacecraftOrbitalEnv, self).__init__()

        # Orbital parameters (initial orbit)
        self.initial_orbit = Orbit.from_classical(
            Earth,
            a=7000 * u.km,  # Semi-major axis (7000 km)
            ecc=0.001 * u.one,  # Almost circular orbit
            inc=0 * u.deg,  # Inclination
            raan=0 * u.deg,  # Right ascension of ascending node
            argp=0 * u.deg,  # Argument of periapsis
            nu=0 * u.deg  # True anomaly
        )

        self.target_orbit = Orbit.from_classical(
            Earth,
            a=7050 * u.km,  # Slightly higher orbit (7050 km)
            ecc=0.001 * u.one,
            inc=0 * u.deg,
            raan=0 * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg
        )

        # Define action space: ΔV in (x, y, z) directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space: [r_x, r_y, r_z, v_x, v_y, v_z, fuel]
        self.observation_space = spaces.Box(
            low=np.array([-1e7, -1e7, -1e7, -10, -10, -10, 0]),
            high=np.array([1e7, 1e7, 1e7, 10, 10, 10, 1]),
            dtype=np.float32
        )

        # Reset environment
        self.reset()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        """Resets the spacecraft's orbit."""
        self.orbit = self.initial_orbit  # Reset to initial orbit
        self.fuel = 1.0  # Full fuel tank

        # Extract position and velocity in Cartesian coordinates
        r, v = self.orbit.rv()
        r, v = r.to(u.km).value, v.to(u.km / u.s).value

        self.state = np.concatenate([r, v, [self.fuel]], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        """Applies thrust and updates the orbit."""
        # Get current position and velocity
        r, v = self.orbit.rv()
        r, v = r.to(u.km).value, v.to(u.km / u.s).value

        # Convert action to ΔV (impulse)
        delta_v = np.clip(action, -1, 1) * 0.1  # Small impulse per step

        # Apply impulse maneuver
        maneuver = Maneuver.impulse(delta_v * u.km / u.s)
        new_orbit = self.orbit.apply_maneuver(maneuver)

        # Update fuel consumption
        self.fuel = max(0, self.fuel - np.linalg.norm(delta_v) * 0.05)

        # Extract new state
        r_new, v_new = new_orbit.rv()
        r_new, v_new = r_new.to(u.km).value, v_new.to(u.km / u.s).value

        # Compute reward
        reward = self.compute_reward(r_new, v_new, delta_v)

        # Check if mission is complete
        done = False
        if np.linalg.norm(r_new - self.target_orbit.r.to(u.km).value) < 5:
            done = True  # Successfully reached target orbit
        if self.fuel <= 0:
            done = True  # Out of fuel

        self.state = np.concatenate([r_new, v_new, [self.fuel]], dtype=np.float32)
        return self.state, reward, done, False, {}
    # Updated reward function
    def compute_reward(self, position, velocity, delta_v):
        """Computes a shaped reward function to guide the agent."""
        target_r, target_v = self.target_orbit.rv()
        target_r, target_v = target_r.to(u.km).value, target_v.to(u.km / u.s).value
    
        # Distance reward (scaled better)
        reward_distance = -0.01 * np.linalg.norm(position - target_r)

        # Velocity alignment
        reward_velocity = -0.2 * np.linalg.norm(velocity - target_v)

        # Fuel penalty (scaled stronger)
        reward_fuel = -0.05 * np.linalg.norm(delta_v)

        # Bonus for getting close to the target orbit
        if np.linalg.norm(position - target_r) < 50:
            reward_goal = 50  # Partial bonus
        elif np.linalg.norm(position - target_r) < 10:
            reward_goal = 100  # Major bonus
        else:
            reward_goal = 0

        # Heavy penalty if out of fuel and far from target
        reward_fuel_penalty = -100 if self.fuel <= 0 and np.linalg.norm(position - target_r) > 50 else 0

        total_reward = reward_distance + reward_velocity + reward_fuel + reward_goal + reward_fuel_penalty
        return total_reward

'''
    def compute_reward(self, position, velocity, delta_v):
        """Computes the reward function."""
        # Distance to target orbit
        reward_distance = -np.linalg.norm(position - self.target_orbit.r.to(u.km).value) * 0.1

        # Velocity matching
        reward_velocity = -np.linalg.norm(velocity - self.target_orbit.v.to(u.km / u.s).value) * 0.5

        # Fuel efficiency
        reward_fuel = -np.linalg.norm(delta_v) * 0.01

        # Bonus for reaching target orbit
        reward_goal = 100 if np.linalg.norm(position - self.target_orbit.r.to(u.km).value) < 5 else 0

        # Penalty for fuel depletion
        reward_fuel_penalty = -100 if self.fuel <= 0 else 0

        total_reward = reward_distance + reward_velocity + reward_fuel + reward_goal + reward_fuel_penalty
        return total_reward
'''