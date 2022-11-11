"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/ant/
"""
import itertools as it
import pickle as pk
import time
import numpy as np
import os

env_name = 'AntBulletEnv-v0'
root_path = 'ant_ars_results'
T = 1000 # max timesteps

def train():

    import pybullet_envs
    from ars_multiprocessing import gym_env_maker, augmented_random_search
    augmented_random_search(
        gym_env_maker(env_name),
        N = 60,
        b = 20,
        alpha = .015,
        nu = .025,
        num_steps = T,
        num_updates = 1000,
        # num_workers = 1, # laptop
        num_workers = 10, # lab workstation
        save_root_path = root_path,
        resume_filename = None,
    )

def viz():

    import pybullet_envs
    from ars_multiprocessing import gym_env_maker, visualize
    visualize(gym_env_maker(env_name), T, root_path)

def show():

    from ars_plot import plot_metrics
    plot_metrics(root_path)

if __name__ == "__main__":
    train()
    # viz()
    # show()

