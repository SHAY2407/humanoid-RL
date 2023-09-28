from dm_control import suite
from dm_control import viewer
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, TD3, a2c
import argparse
import os

# Load the Humanoid environment
env = suite.load(domain_name="humanoid", task_name="walk")

# Define a random policy
def random_policy(time_step):
    return np.random.uniform(low=-1, high=1, size=env.action_spec().shape)

# Visualize the environment
viewer.launch(env, policy=random_policy)
