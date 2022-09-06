"""
So-called "basic random search" from https://arxiv.org/pdf/1803.07055.pdf
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
import itertools as it

def basic_random_search(env, net, alpha, num_dirs, stdev, num_updates, num_steps, batch_size, report_period, viz_rollouts = False):

    with tr.no_grad():

        pvec = tr.nn.utils.parameters_to_vector(net.parameters()).detach().clone()
        pvec.data[:] = 0.
    
        reward_curve = np.empty((num_updates, env.num_domains, batch_size))
        policy = lambda s: (net(tr.tensor(s, dtype=tr.float)).detach().numpy(), None)

        if viz_rollouts:
            pt.ion()
            xpt, ypt, g = env.gravity_mesh()
            pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))
            pt.pause(0.01)
    
        for update in range(num_updates):

            delta = tr.randn(num_dirs, len(pvec))
            net_reward = {sign: tr.empty(num_dirs) for sign in (-1, +1)}
    
            for k, sign in it.product(range(num_dirs), (-1, +1)):

                dvec = pvec + sign * delta[k] * stdev
                tr.nn.utils.vector_to_parameters(dvec, net.parameters())
                states, _, rewards = env.run_episode(policy, num_steps, reset_batch_size=batch_size)
                net_reward[sign][k] = rewards.sum(axis=0).mean()

                # visualize rollouts
                if viz_rollouts:
                    for d in range(env.num_domains):
                        for b in range(batch_size):
                            pt.plot(states[:,d,b,0], states[:,d,b,1], 'b.')
                    pt.pause(0.01)

            # update policy
            pvec = pvec + alpha * ((net_reward[+1] - net_reward[-1]).reshape(-1, 1) * delta).mean(dim=0)
            tr.nn.utils.vector_to_parameters(pvec, net.parameters())

            # measure new reward
            _, _, rewards = env.run_episode(policy, num_steps, reset_batch_size=batch_size)
            reward_curve[update] = rewards.sum(axis=0)

            if update % report_period == 0:
                print(f"{update}/{num_updates}: " + \
                    f"reward={reward_curve[update].mean()} (+/- {reward_curve[update].std()}), " + \
                "")

    return policy, reward_curve


def main():

    from pointbotenv import PointBotEnv

    num_updates = 50
    report_period = 1
    num_steps = 150
    stdev = 0.5
    alpha = 1
    num_dirs = 8

    num_domains = 1
    batch_size = 8

    env = PointBotEnv.sample_domains(num_domains)

    linnet = tr.nn.Sequential(
        tr.nn.Linear(env.obs_size, env.act_size),
        tr.nn.Sigmoid(),
    )

    policy, reward_curve = basic_random_search(env, linnet, alpha, num_dirs, stdev, num_updates, num_steps, batch_size, report_period, viz_rollouts = False)

    mean = reward_curve.mean(axis=(1,2))
    stdev = reward_curve.std(axis=(1,2))
    pt.fill_between(np.arange(len(reward_curve)), mean-stdev, mean+stdev, color='b', alpha=0.5)
    pt.plot(mean)
    pt.show()
    pt.close()

    with tr.no_grad():
        episodes = env.run_episode(policy, num_steps, reset_batch_size=1)
    reward = env.animate(episodes)
    print(reward)
    pt.show()

if __name__ == "__main__": main()

