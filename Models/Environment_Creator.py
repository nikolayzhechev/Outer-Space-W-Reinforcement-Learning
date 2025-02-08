import importlib
from SpacecraftDockingEnv import SpacecraftDockingEnv
from SpacecraftOrbitalEnv import SpacecraftOrbitalEnv
from SpacecraftInterplanetaryEnv import SpacecraftInterplanetaryEnv
from SpacecraftDockingEnv_v2 import SpacecraftDockingEnv_v2
from SimpleDockingEnv import SimpleDockingEnv
from enum import Enum

# Define enmum for environments
Environments_enum = Enum(
    'Environments',
    [
        ('Docking', 'docking'),
        ('Orbital', 'orbital'),
        ('Interplanetary', 'interplanetary'),
        ('Docking_v2', 'docking_v2'),
        ('Docking_simple', 'docking_simple')
    ]
)

def env_creator(environment_type, config=None):
    '''Instantiate a predefined environment. Environment type is selected from enum.'''
    if environment_type == "docking":
        return SpacecraftDockingEnv(config)

    if environment_type == "orbital":
        return SpacecraftOrbitalEnv(config)

    if environment_type == "interplanetary":
        return SpacecraftInterplanetaryEnv(config)

    if environment_type == "docking_v2":
        return SpacecraftDockingEnv_v2()

    if environment_type == "docking_simple":
        return SimpleDockingEnv()
        
    raise ValueError(f"Unknown environment type: {environment_type}")