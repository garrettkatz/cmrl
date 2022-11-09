"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/ant/
"""
import itertools as it
import pickle as pk
import time
import numpy as np
import os

env_name = 'HumanoidBulletEnv-v0'
root_path = 'humanoid_ars_results'
T = 1000 # max timesteps

make_env = gym_env_maker(env_name)

def train():

    from ars_multiprocessing import gym_env_maker, augmented_random_search
    augmented_random_search(
        make_env,
        N = 230,
        b = 230,
        alpha = .02,
        nu = .0075,
        num_steps = T,
        num_updates = 3000,
        num_workers = 10,
        save_root_path = root_path,
        resume_filename = None,
    )

def viz():

    from ars_multiprocessing import visualize
    visualize(make_env, T, root_path)

def show():

    from ars_plot import plot_metrics
    plot_metrics(root_path)

if __name__ == "__main__":
    train()
    # viz()
    # show()



