import importlib
from SpacecraftDockingEnv import SpacecraftDockingEnv
from SpacecraftOrbitalEnv import SpacecraftOrbitalEnv
from SpacecraftInterplanetaryEnv import SpacecraftInterplanetaryEnv
from SpacecraftDockingEnv_v2 import SpacecraftDockingEnv_v2
from SimpleDockingEnv import SimpleDockingEnv
from SpaceNavigationEnv import SpaceNavigationEnv
from SpaceNavigationEnv_v2 import SpaceNavigationEnv_v2
from enum import Enum

# Define enmum for environments
Environments_enum = Enum(
    'Environments',
    [
        ('Docking', 'docking'),
        ('Orbital', 'orbital'),
        ('Interplanetary', 'interplanetary'),
        ('Docking_v2', 'docking_v2'),
        ('Docking_simple', 'docking_simple'),
        ('Navigation', 'navigation'),
        ('Navigation_v2', 'navigation_v2')
    ]
)

def env_creator(environment_type, config=None):
    '''Instantiate a predefined environment. Environment type is selected from enum.'''
    match environment_type:
        case "docking":
            return SpacecraftDockingEnv(config)
        case "docking_v2":
            return SpacecraftDockingEnv_v2()
        case "docking_simple":
            return SimpleDockingEnv()
        case "orbital":
            return SpacecraftOrbitalEnv(config)
        case "interplanetary":
            return SpacecraftInterplanetaryEnv(config)
        case "navigation":
            return SpaceNavigationEnv()
        case "navigation_v2":
            return SpaceNavigationEnv_v2()
        case _:
            raise ValueError(f"Unknown environment type: {environment_type}")