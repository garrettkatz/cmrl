import numpy as np
import torch as tr
import matplotlib.pyplot as pt

# customized to pass in previous actions
def run_episode(env, policy, T, batch_size):
    num_domains, obs_size, act_size = env.num_domains, env.obs_size, env.act_size

    states = np.empty((T+1, num_domains, batch_size, obs_size))
    actions = np.empty((T, num_domains, batch_size, act_size))
    previous = np.zeros((T+1, num_domains, batch_size, act_size))
    rewards = np.empty((T, num_domains, batch_size))

    states[0] = env.reset(batch_size)
    for t in range(T):
        with tr.no_grad(): actions[t], _ = policy(states[t], previous[t])
        states[t+1], rewards[t], _, _ = env.step(actions[t])
        previous[t+1] = actions[t]

    return states, actions, previous, rewards

def delta_vanilla_policy_gradient(env, policy, optimizer, num_updates, num_steps, batch_size, report_period):

    num_domains, obs_size, act_size = env.num_domains, env.obs_size, env.act_size
    T = num_steps

    reward_curve = np.empty((num_updates, num_domains, batch_size))
    for update in range(num_updates):

        # rollouts
        states, actions, previous, rewards = run_episode(env, policy, T, batch_size)

        # rewards-to-go
        returns = rewards.sum(axis=0) - rewards.cumsum(axis=0) + rewards

        # forward pass
        _, log_probs = policy(states[:T], previous[:T], actions)

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

    num_updates = 100
    report_period = 10
    num_steps = 150
    stdev = 0.1
    learning_rate = 0.1

    num_domains = 1
    batch_size = 64

    env = PointBotEnv.sample_domains(num_domains)

    class Policy:
        def __init__(self, net, stdev):
            self.net = net
            self.stdev = stdev
            self.explore = True

        def reset(self, explore):
            self.explore = explore

        def __call__(self, observation, previous_action, action=None):

            # include previous action in input
            observation = tr.tensor(observation, dtype=tr.float)
            previous_action = tr.tensor(previous_action, dtype=tr.float)
            inp = tr.cat((observation, previous_action), dim=-1)

            # normal random exploration centered around delta on previous action
            delta = self.net(inp)
            mu = previous_action + delta
            dist = tr.distributions.Normal(mu, self.stdev)

            # sample random action if not already provided from previous rollout
            if action is None:
                action = dist.sample() if self.explore else mu
            else:
                action = tr.tensor(action, dtype=tr.float)

            # calculate log_prob, summed across action dim
            log_prob = dist.log_prob(action).sum(dim=-1)

            return action.numpy(), log_prob

    class LinNet(tr.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = tr.nn.Linear(env.obs_size + env.act_size, env.act_size)
        def forward(self, inp):
            return stdev * tr.tanh(self.lin(inp))

    linnet = LinNet()
    policy = Policy(linnet, stdev)
    optimizer = tr.optim.SGD(linnet.parameters(), lr=learning_rate)

    policy, reward_curve = delta_vanilla_policy_gradient(env, policy, optimizer, num_updates, num_steps, batch_size, report_period)

    mean = reward_curve.mean(axis=(1,2))
    std = reward_curve.std(axis=(1,2))
    pt.fill_between(np.arange(len(reward_curve)), mean-std, mean+std, color='b', alpha=0.5)
    pt.plot(mean)
    pt.show()
    pt.close()

    # get episode with trained policy
    policy.reset(explore=False)
    states, actions, previous, rewards = run_episode(env, policy, num_steps, batch_size=1)
    episodes = (states, actions, rewards)
    print(actions)
    env.animate(episodes)
    pt.show()

if __name__ == "__main__": main()



