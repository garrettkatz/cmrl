import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from policies import FixedPolicy

def blind_rollouts(epsilon, stdev, num_updates, num_steps, env, batch_size, report_period):

    num_domains, obs_size, act_size = env.num_domains, env.obs_size, env.act_size
    T = num_steps

    best_value = -np.inf
    best_values = -np.inf * np.ones(num_domains)
    best_deltas = np.zeros((T, 1, act_size))
    best_deltas[0] = np.random.rand(1, act_size)

    reward_curve = np.empty((num_updates, num_domains))
    for episode in range(num_updates):

        states = np.empty((T+1, num_domains, batch_size, obs_size))
        rewards = np.empty((T, num_domains, batch_size))

        # explore random trajectory deltas epsilon of the time
        deltas = np.tile(best_deltas, (1, batch_size, 1)) # (T, B, A)
        for t in range(T):
            explore = np.flatnonzero(np.random.rand(batch_size) < epsilon)
            if t == 0:
                deltas[t][explore, :] = np.random.rand(len(explore), act_size)
            else:
                deltas[t][explore, :] = np.random.randn(len(explore), act_size) * stdev

        # rollout
        actions = env.bound(deltas.cumsum(axis=0)).reshape(T, 1, batch_size, act_size) # broadcast over domains
        states[0] = env.reset(batch_size)
        for t in range(T):
            states[t+1], rewards[t], _, _ = env.step(actions[t])

        # update from best value in batch averaged across domains
        values = rewards.sum(axis=0) # (D,B)
        mean_values = values.mean(axis=0) # (B,)
        best_idx = np.argmax(mean_values)
        if mean_values[best_idx] > best_value:
            best_deltas[:, 0, :] = deltas[:, best_idx, :]
            best_values = values[:, best_idx]
            best_value = mean_values[best_idx]

        reward_curve[episode] = best_values

        if episode % report_period == 0:
            print(f"{episode}/{num_updates}: " + \
                f"reward={best_values.mean()} (+/- {best_values.std()}), " + \
            "")

    best_actions = env.bound(best_deltas.cumsum(axis=0)).reshape(T, 1, 1, act_size) # broadcast over domains and batch
    policy = FixedPolicy(best_actions)
    return policy, reward_curve


def main():

    from pointbotenv import PointBotEnv, FixedPolicy

    num_updates = 100
    report_period = 10
    num_steps = 150
    epsilon = 1.
    stdev = 0.1

    num_domains = 2
    batch_size = 32

    env = PointBotEnv.sample_domains(num_domains)
    policy, reward_curve = blind_rollouts(epsilon, stdev, num_updates, num_steps, env, batch_size, report_period)

    pt.ioff()

    pt.plot(reward_curve)
    pt.ylabel("reward")

    pt.show()
    pt.close()

    episodes = env.run_episode(policy, num_steps, reset_batch_size=1)
    reward = env.animate(episodes)
    print(reward)
    print(reward.sum(axis=0))
    pt.show()

if __name__ == "__main__": main()



