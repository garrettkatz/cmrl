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

T = 1000 # max timesteps

def train():

    import gym
    import pybullet as pb
    import pybullet_envs

    α = .015
    ν = .025
    N = 60
    b = 20
    p, n = 8, 28 # action dim, observation dim
    num_updates = 1000

    M = np.zeros((p, n))
    μ = np.zeros(n)
    Σ = np.ones(n)
    nx = 0

    metrics = {key: [] for key in ('runtime','lifetime','reward')}

    env = gym.make('AntBulletEnv-v0')

    for j in range(num_updates):
        update_start = time.perf_counter()

        δ = np.random.randn(N, p, n)
        δM = np.stack((M + ν*δ, M - ν*δ), axis=1)
        r = np.zeros((N, 2, T))
        alive = np.zeros((N, 2))

        for (k, s) in it.product(range(N), range(2)):
            # print("",k,s)

            x = env.reset()
            for t in range(T):
                alive[k,s] = t

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

        metrics['runtime'].append(time.perf_counter() - update_start)
        metrics['lifetime'].append(alive.mean())
        metrics['reward'].append(r.mean())

        print(f"update {j}/{num_updates}: reward ~ {metrics['reward'][-1]:.2f}, |μ| ~ {np.fabs(μ).mean():.2f}, " + \
              f"|Σ < ∞|={(Σ < np.inf).sum()}, |Σ| ~ {np.fabs(Σ[Σ < np.inf]).mean():.2f}, " + \
              f"T ~ {metrics['lifetime'][-1]:.2f} " + \
              f"[{metrics['reward'][-1]:.2f}s]")

        with open("results", "wb") as f: pk.dump((metrics, M, μ, Σ), f)

    env.close()

def viz():

    import gym
    import pybullet as pb
    import pybullet_envs

    with open("results", "rb") as f: (metrics, M, μ, Σ) = pk.load(f)

    env = gym.make('AntBulletEnv-v0')
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

    with open("results", "rb") as f: (metrics, M, μ, Σ) = pk.load(f)

    updates = np.arange(len(metrics['lifetime']))
    walltime = np.cumsum(metrics['runtime']) / (60*60)

    fig, ax = pt.subplots(2, 2, layout='constrained')
    for c,key in enumerate(['lifetime','reward']):
        for r,xticks in enumerate([updates, walltime]):
            ax[r,c].plot(xticks, metrics[key])
            ax[r,c].set_ylabel(key)
            ax[r,c].set_xlabel(["Update","Wall time (hrs)"][r])
    pt.tight_layout()
    pt.show()

if __name__ == "__main__":
    # train()
    # viz()
    show()

