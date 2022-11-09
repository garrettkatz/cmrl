"""
Augmented Random Search:
    https://proceedings.neurips.cc/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf
Online mean/variance calculation:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
"""
import itertools as it
import pickle as pk
import time
import yaml
import os
import numpy as np
import gym
from gym.spaces import Space
from dm_control import suite
import multiprocessing
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tensorboardX import SummaryWriter

def run_rollouts(tuple_args):
    env, T, M, delta, mean, var, nu = tuple_args
    delta_M = np.stack((M + nu*delta, M - nu*delta), axis=1)
    N = delta.shape[0]

    # Allocate arrays for reward and episode length
    r = np.zeros((N, 2, T))
    alive = np.zeros((N, 2))
    all_x = []

    for (k, s) in it.product(range(N), range(2)):
        # Reset environment for current rollout
        x = env.reset()
        
        for t in range(T):
            alive[k,s] = t # update episode length

            # Apply linear policy to standardized observation and perform action
            a = delta_M[k,s] @ ((x - mean) / np.sqrt(var + 1e-8))
            x, r[k,s,t], done, _ = env.step(a)
            all_x.append(x)

            if done:
                break

    x = np.stack(all_x)
    nx = x.shape[0] # number of samples
    mean = x.mean(axis=0)
    sx = x.sum(axis=0) # sum of samples
    ssd = ((x - mean)**2).sum(axis=0) # sum of squared differences from mean
    r = r.sum(axis=2)

    return nx, sx, ssd, r, alive

def augmented_random_search(
    make_env, # make_env() should return a gym-conformant environment instance
    N, # batch size
    b, # top-b of batch used for update
    alpha, # step size
    nu, # noise size
    num_steps, # max steps per rollout
    num_updates, # number of parameter updates
    num_workers, # number of parallel workers for rollouts, should divide N
    save_root_path, # directory to save logging data
    resume_filename = None, # checkpoint filename (None starts from scratch)
):

    # Initialize one copy of environment per worker
    envs = [make_env() for _ in range(num_workers)]
    p = envs[0].action_space.shape[0]
    n = envs[0].observation_space.shape[0]

    if resume_filename is not None:
        # Load progress
        with open(resume_filename, "rb") as f: 
            (metrics, M, mean, var, nx) = pk.load(f)
        print('Load from %s!' % resume_filename)
    else:
        # Initialize linear policy matrix and observation statistics
        M = np.zeros((p, n))
        mean = np.zeros(n)
        var = np.ones(n)
        nx = 0 # number of samples for online mean/variance calculation        
        metrics = {key: [] for key in ('runtime','lifetime','reward')} # Initialize learning metrics
        print('Train a new model!')

    # Updates so far
    E = len(metrics['runtime'])

    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    log = SummaryWriter(os.path.join(save_root_path + 'log/'))

    with multiprocessing.Pool(num_workers) as pool:
        for e in range(1, num_updates + 1):
            update_start = time.perf_counter() # time the update

            # Sample random perturbations to linear policy
            delta = np.random.randn(N, p, n)
            
            # chunk perturbations for parallel workers
            delta = delta.reshape(num_workers, -1, p, n)

            # Collect rollout data for policy and state statistics updates
            worker_args = [(envs[i], num_steps, M, delta[i], mean, var, nu) for i in range(num_workers)]
            results = pool.map(run_rollouts, worker_args)
            nxs, sxs, ssds, rs, alives = zip(*results)

            # Unchunk perturbations for parameter update
            delta = delta.reshape(N, p, n)

            # Update policy parameters
            rs = np.concatenate(rs, 0) # [N, 2]
            alives = np.concatenate(alives, 0) # [N, 2]

            kappa = np.argsort(rs.max(axis=1))[-b:]
            drs = (rs[kappa,0]-rs[kappa,1]).reshape(-1, 1, 1)
            sigma_R = rs[kappa].std()
            M = M + alpha / sigma_R * np.mean(drs * delta[kappa], axis=0)

            # Update state statistics
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            nx_old, mean_old, ssd_old = nx, mean, var * nx
            nx_new = sum(nxs)
            mean_new = sum(sxs) / nx_new
            ssd_new = sum(ssds)

            nx = nx_old + nx_new
            dmean = mean_new - mean_old
            mean = mean_old + dmean * nx_new / nx
            ssd = ssd_old + ssd_new + dmean**2 * nx_old * nx_new / nx
            var = ssd / nx

            # Update and log learning metrics
            metrics['runtime'].append(time.perf_counter() - update_start)
            metrics['lifetime'].append(alives.mean())
            metrics['reward'].append(rs.mean())
            log.add_scalar('Lifetime', alives.mean(), e)
            log.add_scalar('Reward', rs.mean(), e)

            # Print progress update
            print(f"update {E+e}/{E+num_updates}: reward ~ {metrics['reward'][-1]:.2f}, " + \
                f"|μ| ~ {np.fabs(mean).mean():.2f} (nx={nx}), " + \
                f"|Σ < ∞|={(var < np.inf).sum()}, |Σ| ~ {np.fabs(var[var < np.inf]).mean():.2f}, " + \
                f"T ~ {metrics['lifetime'][-1]:.2f}, " + \
                f"Running time ~ {metrics['runtime'][-1]:.2f}s")

            # Save progress
            with open(os.path.join(save_root_path, 'progress.pkl'), "wb") as f: 
                pk.dump((metrics, M, mean, var, nx), f)

    log.close()

    # Return final metrics and policy
    return (metrics, M, mean, var, nx)

def gym_env_maker(env_name):
    def make_env():
        return gym.make(env_name)
    return make_env

# gym conformant wrapper
class DMCEnv:
    def __init__(self, env):
        self.env = env
        self.action_space = Space(shape=env.action_spec().shape)
        self.observation_space = Space(shape=(sum(v.shape[0] for v in env.observation_spec().values()),))
    def obs_array(self, ts):
        return np.concatenate(tuple(ts.observation.values()))
    def reset(self):
        ts = self.env.reset()
        return self.obs_array(ts)
    def step(self, action):
        ts = self.env.step(action)
        return (self.obs_array(ts), ts.reward, ts.last(), None) 

def dmc_env_maker(domain_name, task_name):
    def make_env():
        return DMCEnv(suite.load(domain_name, task_name))
    return make_env

def visualize(make_env, max_steps, root_path, show=True):
    with open(os.path.join(root_path, 'progress.pkl'), 'rb') as f:
        (metrics, M, mean, var, nx) = pk.load(f)

    env = make_env()
    if show: env.render(mode="human")

    x = env.reset()
    tr = 0
    steps = 0
    done = False
    recoder = VideoRecorder(env, os.path.join(root_path, 'viz.mp4'), enabled = True)

    while not done:
        a = M @ ((x - mean) / np.sqrt(var + 1e-8))
        x, r, done, _ = env.step(a)
        recoder.capture_frame()

        steps += 1
        tr += r

        if steps == max_steps: break

    recoder.close()
    recoder.enabled = False
    env.close()
    print('Steps = %d | Reward = %.2f' % (steps, tr))

if __name__ == '__main__':

    params = dict(
        # env_name = 'Humanoid-v4',
        # env_name = 'HumanoidBulletEnv-v0',
        # env_name = 'AntBulletEnv-v0',
        env_name = 'HopperEnv-v0',
        N = 64,
        b = 20,
        alpha = 0.015,
        nu = 0.025,
        num_updates = 10000,
        # resume_filename = './pre.pkl',
        save_root_path = './ars_results/hopper/',
        num_workers = 2
    )

    print(params)

    with open('config.yaml', 'w') as f:
        yaml.dump(params, f)

    params["make_env"] = gym_env_maker(params.pop("env_name"))
    augmented_random_search(**params)

    viz(env_name = params['env_name'], fname = os.path.join(params['save_root_path'], 'ars.pkl'))
