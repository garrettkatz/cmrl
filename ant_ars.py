"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/ant/
ARS:
https://proceedings.neurips.cc/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf
Online mean/variance:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""
import itertools as it
import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as pt
import gym
import pybullet as pb
import pybullet_envs

def train():

    α = .015
    ν = .025
    N = 60
    b = 20
    p, n = 8, 28 # action dim, observation dim
    T = 500 # max timesteps
    num_updates = 100
    show_period = 10 # updates between visualization

    M = np.zeros((p, n))
    μ = np.zeros(n)
    Σ = np.ones(n)
    nx = 0

    runtimes = []

    env = gym.make('AntBulletEnv-v0')

    for j in range(num_updates):
        update_start = time.perf_counter()

        δ = np.random.randn(N, p, n)
        δM = np.stack((M + ν*δ, M - ν*δ), axis=1)
        r = np.zeros((N, 2, T))

        for (k, s) in it.product(range(N), range(2)):
            # print("",k,s)

            x = env.reset()
            for t in range(T):

                dx = x - μ
                μ += dx / (nx + 1)
                Σ = (Σ * nx + dx * (x - μ)) / (nx + 1)
                nx += 1

                a = δM[k,s] @ ((x - μ) / np.sqrt(np.where(Σ < 1e-8, np.inf, Σ)))
                x, r[k,s,t], done, _ = env.step(a)
                if done: break

        r = r.sum(axis=2)
        κ = np.argsort(r.max(axis=1))[-b:]
        dr = (r[κ,0]-r[κ,1]).reshape(-1, 1, 1)
        σR = r[κ].std()
        M = M + α / σR * np.mean(dr * δ[κ], axis=0)

        runtimes.append(time.perf_counter() - update_start)

        print(f"update {j}: reward ~ {r.mean()}, |μ| ~ {np.fabs(μ).mean()}, " + \
              f"|Σ < ∞|={(Σ < np.inf).sum()}, |Σ| ~ {np.fabs(Σ[Σ < np.inf]).mean()} " + \
              f"[{runtimes[-1]}s]")

    env.close()

if __name__ == "__main__":
    train()
