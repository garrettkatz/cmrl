import itertools as it
import pickle as pk
import time
import numpy as np
import os, sys
from gym.spaces import Space

root_path = 'ergo_ars_results'
timestep = 1/240
control_period = 60 # 4 position commands per second
T = 20 # 5 seconds

def train():

    from ergo_wrapper import env_maker, initial_policy
    from ars_multiprocessing import augmented_random_search
    augmented_random_search(
        env_maker(timestep, control_period),
        N = 256,
        b = 256,
        alpha = .01,
        nu = .001,
        num_steps = T,
        num_updates = 1000,
        # num_workers = 2, # laptop
        num_workers = 10, # lab workstation
        save_root_path = root_path,
        resume_filename = None,
        # M = initial_policy(),
    )

def viz():

    from ergo_wrapper import env_maker
    from ars_multiprocessing import visualize
    visualize(env_maker(timestep, control_period), T, root_path, record=False)

def show():

    from ars_plot import plot_metrics
    plot_metrics(root_path)

if __name__ == "__main__":
    train()
    # viz()
    # show()

