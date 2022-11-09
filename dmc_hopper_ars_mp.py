"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
import itertools as it
import pickle as pk
import time
import numpy as np
import os

domain_name = 'hopper'
task_name = 'hop'
root_path = 'hopper_ars_results'
T = 1000 # max timesteps

def train():

    from ars_multiprocessing import dmc_env_maker, augmented_random_search
    augmented_random_search(
        dmc_env_maker(domain_name, task_name),
        N = 60,
        b = 20,
        alpha = .015,
        nu = .025,
        num_steps = T,
        num_updates = 1000,
        num_workers = 1, # laptop
        # num_workers = 10, # lab workstation
        save_root_path = root_path,
        resume_filename = None,
    )

def viz():

    from ars_multiprocessing import dmc_env_maker, visualize
    visualize(dmc_env_maker(domain_name, task_name), T, root_path)

def show():

    from ars_plot import plot_metrics
    plot_metrics(root_path)

if __name__ == "__main__":
    train()
    # viz()
    # show()

