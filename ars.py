"""
Augmented Random Search:
    https://proceedings.neurips.cc/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf
Online mean/variance calculation:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""
import itertools as it
import pickle as pk
import time
import numpy as np

"""
Inputs:
    env: gym-conformant environment with reset() and step() API
    num_timesteps: maximum time-steps per episode
    α, ν, N, b: ARS hyper-parameters as described in paper
    p, n: dimensionality of action and observation spaces, respectively
    num_updates: number of policy updates before termination
    result_filename: filename for saving outputs
Outputs:
    metrics: dictionary with following key-value pairs:
        'runtime': list of time in seconds for each update
        'lifetime': list of average episode length for each update
        'reward': list of average episode net reward for each update
    M: (p,n) matrix of final linear policy
    μ: (n,) array of element-wise observation mean
    Σ: (n,) array of element-wise observation variance
"""
def augmented_random_search(env, num_timesteps, α, ν, N, b, p, n, num_updates, result_filename, resume=False):

    if resume:

        # Load progress
        with open(result_filename, "rb") as f: (metrics, M, μ, Σ, nx) = pk.load(f)

    else:

        # Initialize linear policy matrix and observation statistics
        M = np.zeros((p, n))
        μ = np.zeros(n)
        Σ = np.ones(n)
        nx = 0 # number of samples for online mean/variance calculation

        # Initialize learning metrics
        metrics = {key: [] for key in ('runtime','lifetime','reward')}

    # Updates so far
    J = len(metrics['runtime'])

    # Iterate policy updates
    for j in range(num_updates):
        update_start = time.perf_counter() # time the update

        # Sample random perturbations to linear policy
        δ = np.random.randn(N, p, n)
        δM = np.stack((M + ν*δ, M - ν*δ), axis=1)

        # Allocate arrays for reward and episode length
        r = np.zeros((N, 2, num_timesteps))
        alive = np.zeros((N, 2))

        # Rollouts for each signed perturbation
        for (k, s) in it.product(range(N), range(2)):

            # Reset environment for current rollout
            x = env.reset()
            for t in range(num_timesteps):
                alive[k,s] = t # update episode length

                # update observation statistics
                dx = x - μ
                μ += dx / (nx + 1)
                Σ = (Σ * nx + dx * (x - μ)) / (nx + 1)
                nx += 1

                # Apply linear policy to standardized observation and perform action
                a = δM[k,s] @ ((x - μ) / np.sqrt(np.where(Σ < 1e-8, np.inf, Σ)))
                x, r[k,s,t], done, _ = env.step(a)
                if done: break # stop episode if done early

        # Policy update rule
        r = r.sum(axis=2)
        κ = np.argsort(r.max(axis=1))[-b:]
        dr = (r[κ,0]-r[κ,1]).reshape(-1, 1, 1)
        σR = r[κ].std()
        M = M + α / σR * np.mean(dr * δ[κ], axis=0)

        # Update learning metrics
        metrics['runtime'].append(time.perf_counter() - update_start)
        metrics['lifetime'].append(alive.mean())
        metrics['reward'].append(r.mean())

        # Print progress update
        print(f"update {J+j}/{J+num_updates}: reward ~ {metrics['reward'][-1]:.2f}, " + \
              f"|μ| ~ {np.fabs(μ).mean():.2f} (nx={nx}), " + \
              f"|Σ < ∞|={(Σ < np.inf).sum()}, |Σ| ~ {np.fabs(Σ[Σ < np.inf]).mean():.2f}, " + \
              f"T ~ {metrics['lifetime'][-1]:.2f} " + \
              f"[{metrics['reward'][-1]:.2f}s]")

        # Save progress
        with open(result_filename, "wb") as f: pk.dump((metrics, M, μ, Σ, nx), f)

    # Return final metrics and policy
    return (metrics, M, μ, Σ, nx)

