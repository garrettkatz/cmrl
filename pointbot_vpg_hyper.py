import pickle as pk
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from policies import NormalPolicy
from vpg import vanilla_policy_gradient
from pointbotenv import PointBotEnv

def main():

    do_train = True
    do_show = False

    # fixed hyperparams
    report_period = 10
    num_episodes = 2**18
    num_steps = 150
    num_domains = 1
    num_hyperparam_samples = 8

    # nonfixed hyperparam ranges
    log10_stdev_range = (-2, 3)
    log10_learning_rate_range = (-4, 0)
    batch_size_range = (2**6, 2**10)

    # random sampler
    rng = np.random.default_rng()

    # init environment
    env = PointBotEnv.sample_domains(num_domains)

    if do_train:

        curves = {} # learning curves
        trajs = {} # trajectories of untrained policies
        for sample in range(num_hyperparam_samples):
    
            # sample the hyperparameters
            log10_stdev = rng.uniform(*log10_stdev_range)
            log10_learning_rate = rng.uniform(*log10_learning_rate_range)
            batch_size = rng.integers(*batch_size_range, endpoint=True)
    
            hparms = (log10_stdev, log10_learning_rate, batch_size)
            stdev = 10 ** log10_stdev
            learning_rate = 10 ** log10_learning_rate
    
            print(f"{sample}/{num_hyperparam_samples} samples: (log sd, log lr, bs) = {hparms}")
    
            # calculate number of gradient updates, keeping total episodes roughly constant
            num_updates = (num_episodes // batch_size) + 1
    
            # set up policy network and optimizer
            linnet = tr.nn.Sequential(
                tr.nn.Linear(env.obs_size, env.act_size),
                tr.nn.Sigmoid(),
            )
            policy = NormalPolicy(linnet, stdev)
            optimizer = tr.optim.SGD(linnet.parameters(), lr=learning_rate)
    
            # save trajectories of untrained policy (fixed batch size for consistent visualization)
            with tr.no_grad():
                trajs[hparms], _, _ = env.run_episode(policy, num_steps, reset_batch_size=64)
        
            # run the training
            policy, reward_curve = vanilla_policy_gradient(env, policy, optimizer, num_updates, num_steps, batch_size, report_period)
    
            # save reward curve
            curves[hparms] = reward_curve

        # save results to disk
        with open("results_pointbot_vpg_hyper.pkl", "wb") as f: pk.dump((trajs, curves), f)

    if do_show:

        # load results from disk
        with open("results_pointbot_vpg_hyper.pkl", "rb") as f: (trajs, curves) = pk.load(f)

        # render background gravity field once
        pt.subplot(1,2,1)
        xpt, ypt, g = env.gravity_mesh()
        pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))
    
        for hparms, reward_curve in curves.items():
    
            _, _, batch_size = hparms
            states = trajs[hparms]
    
            # reward curves
            pt.subplot(1,2,2)
            xpts = np.arange(len(reward_curve)) * batch_size # x-axis units by total number of episodes
            mean = reward_curve.mean(axis=(1,2))
            stdev = reward_curve.std(axis=(1,2))
            line, = pt.plot(xpts, mean, '-', label="sd=$10^{%.2f}$, lr=$10^{%.2f}$, bs=%d" % hparms, zorder = 1)
            pt.fill_between(xpts, mean-stdev, mean+stdev, color=line.get_color(), alpha=0.5, zorder = 0)
    
            # trajectories with same color code
            pt.subplot(1,2,1)
            pt.plot(states[:,:,:,0].flatten(), states[:,:,:,1].flatten(), '.', color=line.get_color(), markersize=1)
    
        pt.subplot(1,2,2)
        pt.ylabel('reward')
        pt.xlabel('episodes')
        pt.legend()
        pt.tight_layout()
        pt.show()

if __name__ == "__main__": main()
