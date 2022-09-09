import pickle as pk
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from policies import NormalPolicy, FixedPolicy
from vpg import vanilla_policy_gradient
from pointbotenv import PointBotEnv

def main():

    # hyperparams
    report_period = 10
    num_updates = 500
    num_steps = 150
    num_domains = 1
    learning_rate = 10
    batch_size = 512

    net_stdev = 0.1
    rw_stdev = 0.1

    # random sampler
    rng = np.random.default_rng()

    # init environment
    env = PointBotEnv.sample_domains(num_domains)

    # set up policy network and optimizer
    net = tr.nn.Sequential(
        # tr.nn.Linear(env.obs_size, env.act_size),
        tr.nn.Linear(env.obs_size, 3),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(3, env.act_size),
        tr.nn.Sigmoid(),
    )
    net_policy = NormalPolicy(net, net_stdev)
    optimizer = tr.optim.SGD(net.parameters(), lr=learning_rate)

    # set up switching probabilities
    # switch_probs = np.ones(num_steps)
    # switch_probs = np.arange(num_steps, 0, -1, dtype=float)
    switch_probs = 1. / np.arange(1, num_steps+1)
    
    switch_probs /= switch_probs.sum()

    # pt.figure()
    # pt.ion()
    # xpt, ypt, g = env.gravity_mesh()

    # run the training
    reward_curve = np.empty((num_updates, num_domains, batch_size))
    weight_stdev = np.empty(num_updates)
    for update in range(num_updates):

        # track performance using rewards from current policy with pure exploitation
        net_policy.reset(explore = False)
        with tr.no_grad():
            _, _, rewards = env.run_episode(net_policy, num_steps, batch_size)
        reward_curve[update] = rewards.sum(axis=0)

        # exploratory rollouts from policy net
        net_policy.reset(explore = True)
        with tr.no_grad():
            states, actions, rewards = env.run_episode(net_policy, num_steps, batch_size)

        # pt.cla()
        # pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))
        # pt.plot(states[:,:,:,0].flatten(), states[:,:,:,1].flatten(), 'b.', markersize=1)
        # pt.title("policy net")
        # pt.pause(.01)
        # input('.')

        # get action deltas for net
        net_deltas = np.empty(actions.shape)
        net_deltas[0] = actions[0]
        net_deltas[1:] = actions[1:] - actions[:-1]

        # get action deltas for random walk along with log probs
        dist0 = tr.distributions.Uniform(tr.zeros(actions[:1].shape), tr.ones(actions[:1].shape))
        rw_deltas0 = dist0.sample()
        rw_log_probs0 = dist0.log_prob(rw_deltas0)

        dist1 = tr.distributions.Normal(tr.zeros(actions[1:].shape), rw_stdev)
        rw_deltas1 = dist1.sample()
        rw_log_probs1 = dist1.log_prob(rw_deltas1)

        rw_deltas = tr.cat((rw_deltas0, rw_deltas1), dim=0).numpy()
        rw_log_probs = tr.cat((rw_log_probs0, rw_log_probs1), dim=0).sum(dim=-1).numpy()

        # set up switching times and mask
        t_switch = rng.choice(num_steps, size=(env.num_domains, batch_size), p=switch_probs)
        timesteps = np.arange(num_steps).reshape(num_steps, 1, 1)
        before_switch = (timesteps < t_switch)

        # replace subsequent actions with random walk
        deltas = np.where(before_switch[:,:,:,np.newaxis], net_deltas, rw_deltas)
        actions = deltas.cumsum(axis=0)

        # rerun episodes with switched actions
        states, _, rewards = env.run_episode(FixedPolicy(actions), num_steps, batch_size)
        returns = rewards.sum(axis=0)

        # pt.cla()
        # pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))
        # pt.plot(states[:,:,:,0].flatten(), states[:,:,:,1].flatten(), 'b.', markersize=1)
        # pt.title("switching")
        # pt.pause(.01)
        # input('.')

        # forward pass of policy net with gradient
        _, log_probs = net_policy(states[:num_steps], actions)

        # importance sampling log-likelihood ratio without gradient
        net_log_probs = log_probs.detach().numpy()
        after_switch = 1. - before_switch # mask ratios before the switch
        # ratios = np.exp((after_switch * (net_log_probs - rw_log_probs)).sum(axis=0)) / switch_probs[t_switch]
        ratios = np.exp((after_switch * net_log_probs).sum(axis=0) - (after_switch * rw_log_probs).sum(axis=0)) / switch_probs[t_switch] # more numerically stable?

        # print('net lp')
        # print(net_log_probs)
        # print('rw lp')
        # print(rw_log_probs)
        # print('net lp sum t')
        # print((after_switch * net_log_probs).sum(axis=0))
        # print('rw lp sum t')
        # print((after_switch * rw_log_probs).sum(axis=0))
        # print('exp')
        # print(np.exp((after_switch * net_log_probs).sum(axis=0) - (after_switch * rw_log_probs).sum(axis=0)))
        # print('switch prob')
        # print(switch_probs[t_switch])

        # importance weights
        weights = tr.tensor(returns * ratios)
        weight_stdev[update] = weights.std().item()
        # print('weights')
        # print(weights)
        weights /= max(weight_stdev[update], 1)

        # importance-weighted policy gradient
        optimizer.zero_grad()
        (-(weights * log_probs).mean()).backward()
        optimizer.step()

        if update % report_period == 0:
            grad_norm = max([p.grad.abs().max() for p in net.parameters()])
            weight_norm = weights.abs().max()
            print(f"{update}/{num_updates} rewards: " + \
                f"exploit={reward_curve[update].mean()}, explore~{returns.mean()} (+/- {returns.std()})  < {returns.max()}, " + \
                f"|grad|={grad_norm}, |weights|={weight_norm} (* {weight_stdev[update]}), " + \
            "")


    # render background gravity field once
    pt.subplot(1,3,1)
    xpt, ypt, g = env.gravity_mesh()
    pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))

    # reward curves
    pt.subplot(1,3,2)
    xpts = np.arange(len(reward_curve)) * batch_size # x-axis units by total number of episodes
    mean = reward_curve.mean(axis=(1,2))
    stdev = reward_curve.std(axis=(1,2))
    line, = pt.plot(xpts, mean, '-', zorder = 1)
    pt.fill_between(xpts, mean-stdev, mean+stdev, color=line.get_color(), alpha=0.5, zorder = 0)

    # trajectories with same color code
    net_policy.reset(explore = True)
    with tr.no_grad():
        states, _, _ = env.run_episode(net_policy, num_steps, batch_size)
    pt.subplot(1,3,1)
    pt.plot(states[:,:,:,0].flatten(), states[:,:,:,1].flatten(), '.', color=line.get_color(), markersize=1)

    pt.subplot(1,3,3)
    xpts = np.arange(len(weight_stdev)) * batch_size # x-axis units by total number of episodes
    pt.plot(xpts, weight_stdev, '-', color=line.get_color())
    pt.xlabel('episodes')
    pt.ylabel('weight stdev')

    pt.subplot(1,3,2)
    pt.ylabel('reward')
    pt.xlabel('episodes')
    # pt.legend()
    pt.tight_layout()
    pt.show()

if __name__ == "__main__": main()


