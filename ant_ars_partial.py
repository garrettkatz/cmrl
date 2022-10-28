"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/ant/
"""
import itertools as it
import pickle as pk
import time
import numpy as np

T = 1000 # max timesteps
fname = "ant_ars_partial_results.pkl"

class PartialAntEnv:
    def __init__(self, env):
        self.env = env
    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs[5:13], obs[19:]))
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs[5:13], obs[19:]))
        return obs, reward, done, info
    def close(self):
        self.env.close()

def train():

    import gym
    import pybullet as pb
    import pybullet_envs
    from ars import augmented_random_search

    α = .015
    ν = .025
    N = 60
    b = 20
    p, n = 8, 17 # action dim, observation dim
    num_updates = 1000

    env = PartialAntEnv(gym.make('AntBulletEnv-v0'))
    augmented_random_search(env, T, α, ν, N, b, p, n, num_updates, fname)
    env.close()

def viz():

    import gym
    import pybullet as pb
    import pybullet_envs

    with open(fname, "rb") as f: (metrics, M, μ, Σ) = pk.load(f)

    env = gym.make('AntBulletEnv-v0')
    env.render(mode="human")
    env = PartialAntEnv(env)

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

    with open(fname, "rb") as f: (metrics, M, μ, Σ) = pk.load(f)

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
    train()
    # viz()
    # show()

