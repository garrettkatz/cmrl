"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/humanoid/
"""
import itertools as it
import pickle as pk
import time
import numpy as np

T = 1000 # max timesteps
fname = "humanoid_ars_results.pkl"

def train():

    import gym
    import pybullet as pb
    import pybullet_envs
    from ars import augmented_random_search

    α = .02
    ν = .0075
    N = 230
    b = 230
    p, n = 17, 44 # action dim, observation dim
    num_updates = 5000
    resume = False

    env = gym.make('HumanoidBulletEnv-v0')
    augmented_random_search(env, T, α, ν, N, b, p, n, num_updates, fname, resume)
    env.close()

def viz():

    import gym
    import pybullet as pb
    import pybullet_envs

    with open(fname, "rb") as f: (metrics, M, μ, Σ, nx) = pk.load(f)

    env = gym.make('HumanoidBulletEnv-v0')
    env.render(mode="human")

    r = np.zeros(T)
    x = env.reset()
    for t in range(T):
        print(f"t={t}/{T}")
        a = M @ ((x - μ) / np.sqrt(np.where(Σ < 1e-8, np.inf, Σ)))
        x, r[t], done, _ = env.step(a)
        if done: break
        time.sleep(1/24)

    r = r.sum()
    env.close()

    print(f"r = {r}")

def show():

    import matplotlib.pyplot as pt

    with open(fname, "rb") as f: (metrics, M, μ, Σ, nx) = pk.load(f)

    updates = np.arange(len(metrics['lifetime']))
    walltime = np.cumsum(metrics['runtime']) / (60*60)

    fig, ax = pt.subplots(2, 2, layout='constrained')
    for c,key in enumerate(['lifetime','reward']):
        for r,xticks in enumerate([updates, walltime]):
            ax[r,c].plot(xticks, metrics[key])
            ax[r,c].set_ylabel(key)
            ax[r,c].set_xlabel(["Update","Wall time (hrs)"][r])
    pt.show()

if __name__ == "__main__":
    # train()
    viz()
    # show()

