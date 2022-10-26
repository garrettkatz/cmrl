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
    N = 4 # 60
    b = 4 # 20
    p, n = 8, 28 # action dim, observation dim
    T = 500 # max timesteps
    num_updates = 100
    show_period = 10 # updates between visualization

    M = np.zeros((p, n))
    μ = np.zeros(n)
    σ = np.ones(n)

    env = gym.make('AntBulletEnv-v0')

    for j in range(num_updates):

        δ = np.random.randn(N, p, n)
        δM = np.stack((M + ν*δ, M - ν*δ), axis=1)
        r = np.zeros((N, 2, T))
        x = np.zeros((N, 2, T+1, n)) # what if some episodes done early?

        for (k, s) in it.product(range(N), range(2)):
            print("",k,s)

            x[k,s,0] = env.reset()
            for t in range(T):
                a = δM[k,s] @ ((x[k,s,t] - μ) / σ)
                x[k,s,t+1], r[k,s,t], done, _ = env.step(a)
                # if done: break

        r = r.sum(axis=2)
        dr = (r[k,0]-r[k,1]).reshape(-1, 1, 1)
        k = np.argsort(r.max(axis=1))[-b:]
        M = M + α / r[k].std() * np.mean(dr * δ[k], axis=0)

        x = x.mean(axis=(0,1,2))
        xmμ = x - μ
        μ += xmμ / (j + 1)
        σ = (σ**2 * j + xmμ * (x - μ)) / (j+1)
        σ[σ < 1e-8] = np.inf

        print(f"update {j}: reward ~ {r.mean()}")

    env.close()

if __name__ == "__main__":
    train()
