import numpy as np
import torch as tr
import matplotlib.pyplot as pt

def vanilla_policy_gradient(env, policy, optimizer, num_updates, num_steps, batch_size, report_period):

    num_domains, obs_size, act_size = env.num_domains, env.obs_size, env.act_size
    T = num_steps

    reward_curve = np.empty((num_updates, num_domains, batch_size))
    for update in range(num_updates):

        # rollouts
        with tr.no_grad():
            states, actions, rewards = env.run_episode(policy, num_steps, batch_size)

        # rewards-to-go
        returns = rewards.sum(axis=0) - rewards.cumsum(axis=0) + rewards

        # forward pass
        _, log_probs = policy(states[:T], actions)

        # policy gradient
        optimizer.zero_grad()
        (-(tr.tensor(returns) * log_probs).mean()).backward()
        optimizer.step()

        # track performance
        reward_curve[update] = returns[0]

        if update % report_period == 0:
            print(f"{update}/{num_updates}: " + \
                f"reward={returns[0].mean()} (+/- {returns[0].std()}), " + \
            "")

    return policy, reward_curve


def main():

    from pointbotenv import PointBotEnv

    # hyperparams
    num_updates = 1000
    report_period = 10
    num_steps = 150
    stdev = 0.5
    learning_rate = 0.01
    num_domains = 1
    batch_size = 64

    env = PointBotEnv.sample_domains(num_domains)

    # wrapper class around policy network
    class Policy:
        def __init__(self, net, stdev):
            self.net = net
            self.stdev = stdev
            self.explore = True

        def reset(self, explore):
            """
            set exploration flag to true or false
            use true during training, false during testing
            when false, network output mu is the returned action
            when true, actions are sampled from normal centered at mu
            """
            self.explore = explore

        def __call__(self, observation, action=None):
            """
            calls the policy network on the provided observation
            when collecting rollouts, use action == None to randomly sample actions from the output distribution
            when gradient-updating from previously collected rollouts, provide the actions from the rollouts
            
            returns action and its log probability
            """

            # normal random exploration centered around predicted action mu
            mu = self.net(tr.tensor(observation, dtype=tr.float))
            dist = tr.distributions.Normal(mu, self.stdev)

            # sample random action if not already provided from previous rollout
            if action is None:
                action = dist.sample() if self.explore else mu
            else:
                action = tr.tensor(action, dtype=tr.float)

            # calculate log_prob, summed across action dim
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.numpy(), log_prob

    # set up policy network and optimizer
    linnet = tr.nn.Sequential(
        tr.nn.Linear(env.obs_size, env.act_size),
        tr.nn.Sigmoid(),
    )
    policy = Policy(linnet, stdev)
    optimizer = tr.optim.SGD(linnet.parameters(), lr=learning_rate)

    # visualize untrained rollouts
    with tr.no_grad():
        states, actions, rewards = env.run_episode(policy, num_steps, batch_size)
    xpt, ypt, g = env.gravity_mesh()
    pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))
    for d in range(env.num_domains):
        for b in range(batch_size):
            pt.plot(states[:,d,b,0], states[:,d,b,1], 'b.')
    pt.show()

    # run the training
    policy, reward_curve = vanilla_policy_gradient(env, policy, optimizer, num_updates, num_steps, batch_size, report_period)

    # plot the learning curve
    mean = reward_curve.mean(axis=(1,2))
    stdev = reward_curve.std(axis=(1,2))
    pt.fill_between(np.arange(len(reward_curve)), mean-stdev, mean+stdev, color='b', alpha=0.5)
    pt.plot(mean)
    pt.show()
    pt.close()

    # visualize the trained policy
    policy.reset(explore=False)
    with tr.no_grad():
        episodes = env.run_episode(policy, num_steps, reset_batch_size=1)
    reward = env.animate(episodes)
    print(reward)
    pt.show()

if __name__ == "__main__": main()

