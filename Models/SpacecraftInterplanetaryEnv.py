import gymnasium as gym
import numpy as np
from gymnasium import spaces
from poliastro.bodies import Sun, Earth, Mars
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.ephem import Ephem
from astropy import units as u
from astropy.time import Time
from scipy.spatial.transform import Rotation as R

class SpacecraftInterplanetaryEnv(gym.Env):
    """Gym environment for interplanetary transfers (Earth → Mars)."""

    def __init__(self, config=None):
        super(SpacecraftInterplanetaryEnv, self).__init__()

        # Define a reference epoch
        self.epoch = Time("2025-01-01 12:00:00", format="iso", scale="utc")

        # Define initial and target orbits using ephemerides
        self.initial_orbit = Orbit.from_ephem(Earth, Ephem.from_body(Earth, self.epoch), self.epoch)  # Earth's current heliocentric orbit
        self.target_orbit = Orbit.from_ephem(Mars, Ephem.from_body(Mars, self.epoch), self.epoch)    # Mars' orbit around the Sun

        # Define action space: ΔV in (x, y, z) directions (in km/s)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)

        # Define observation space: [r_x, r_y, r_z, v_x, v_y, v_z, fuel]
        self.observation_space = spaces.Box(
            low=np.array([-3e8, -3e8, -3e8, -50, -50, -50, 0]),
            high=np.array([3e8, 3e8, 3e8, 50, 50, 50, 1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the spacecraft's orbit to Earth's orbit."""
        self.orbit = self.initial_orbit  # Start in Earth's heliocentric orbit
        self.fuel = 1.0  # Full fuel tank

        # Extract position and velocity in Cartesian coordinates
        r, v = self.orbit.rv()
        r, v = r.to(u.km).value, v.to(u.km / u.s).value

        self.state = np.concatenate([r, v, [self.fuel]], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        """Applies thrust and updates the orbit."""
        r, v = self.orbit.rv()
        r, v = r.to(u.km).value, v.to(u.km / u.s).value

        # Convert action to ΔV (impulse)
        delta_v = np.clip(action, -5, 5) * u.km / u.s

        # Apply impulse maneuver
        maneuver = Maneuver.impulse(delta_v)
        self.orbit = self.orbit.apply_maneuver(maneuver)

        # Update fuel consumption
        self.fuel = max(0, self.fuel - np.linalg.norm(delta_v.value) * 0.01)

        # Extract new state
        r_new, v_new = self.orbit.rv()
        r_new, v_new = r_new.to(u.km).value, v_new.to(u.km / u.s).value

        # Compute reward
        reward = self.reward(r_new, v_new, delta_v.value)

        # Check if mission is complete
        done = False
        if np.linalg.norm(r_new - self.target_orbit.r.to(u.km).value) < 5e5:  # Close to Mars
            done = True
        if self.fuel <= 0:
            done = True  # Out of fuel

        self.state = np.concatenate([r_new, v_new, [self.fuel]], dtype=np.float32)
        return self.state, reward, done, False, {}

    def reward(self, position, velocity, delta_v):
        """Computes the reward function."""
        reward_distance = -np.linalg.norm(position - self.target_orbit.r.to(u.km).value) * 1e-6
        reward_velocity = -np.linalg.norm(velocity - self.target_orbit.v.to(u.km / u.s).value) * 0.1
        reward_fuel = -np.linalg.norm(delta_v) * 0.05
        reward_goal = 100 if np.linalg.norm(position - self.target_orbit.r.to(u.km).value) < 5e5 else 0
        reward_fuel_penalty = -100 if self.fuel <= 0 else 0

        total_reward = reward_distance + reward_velocity + reward_fuel + reward_goal + reward_fuel_penalty
        return total_reward