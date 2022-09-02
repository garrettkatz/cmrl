import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from pointbotenv import PointBotEnv, FixedPolicy

def epsilon_greedy(epsilon, num_episodes, num_steps, env, batch_size, report_period):

    num_domains = env.num_domains
    T, obs_size, act_size = num_steps, 4, 2

    best_states = np.empty((T+1, num_domains, 1, obs_size))
    best_rewards = np.empty((T, num_domains, 1))
    best_values = -np.inf * np.ones((num_domains, 1))
    # best_actions = np.random.rand() * np.ones((T, num_domains, 1, act_size))
    best_deltas = np.zeros((T, num_domains, 1, act_size))
    best_deltas[0] = np.random.rand() * np.ones((num_domains, 1, act_size))

    reward_curve = np.empty((num_episodes, num_domains))
    for episode in range(num_episodes):

        # actions = np.tile(best_actions, (1, 1, batch_size, 1))
        deltas = np.tile(best_deltas, (1, 1, batch_size, 1))
        states = np.empty((T+1, num_domains, batch_size, obs_size))
        rewards = np.empty((T, num_domains, batch_size))

        # explore random deltas epsilon of the time
        for t in range(T):
            explore = np.flatnonzero(np.random.rand(batch_size) < epsilon)
            if t == 0:
                deltas[t][:, explore, :] = np.random.rand(num_domains, len(explore), act_size)
            else:
                deltas[t][:, explore, :] = np.random.randn(num_domains, len(explore), act_size) * 0.1

        # rollout
        actions = env.bound(deltas.cumsum(axis=0))
        states[0] = env.reset(batch_size)
        for t in range(T):
            states[t+1], rewards[t], _, _ = env.step(actions[t])

        # update best so far
        values = rewards.sum(axis=0) # (D, B)
        best_idx = np.argmax(values, axis=1) # (D,)
        is_better = np.squeeze(np.take_along_axis(values, np.expand_dims(best_idx, axis=1), axis=1) > best_values) # (D,)

        best_states[:, is_better, 0, :] = np.take_along_axis(states, np.expand_dims(best_idx, axis=(0,2,3)), axis=2)[:, is_better, 0, :] # (T, D, 1, S)
        # best_actions[:, is_better, 0, :] = np.take_along_axis(actions, np.expand_dims(best_idx, axis=(0,2,3)), axis=2)[:, is_better, 0, :] # (T, D, 1, A)
        best_deltas[:, is_better, 0, :] = np.take_along_axis(deltas, np.expand_dims(best_idx, axis=(0,2,3)), axis=2)[:, is_better, 0, :] # (T, D, 1, A)
        best_rewards[:, is_better, 0] = np.take_along_axis(rewards, np.expand_dims(best_idx, axis=(0,2)), axis=2)[:, is_better, 0] # (T, D, 1)
        best_values[is_better, 0] = np.take_along_axis(values, np.expand_dims(best_idx, axis=1), axis=1)[is_better, 0] # (D, 1)

        reward_curve[episode] = np.squeeze(best_values, axis=1)

        if episode % report_period == 0:
            print(f"{episode}/{num_episodes}: " + \
                f"reward={reward_curve[episode].mean()}, " + \
            "")

    best_actions = env.bound(best_deltas.cumsum(axis=0))
    policy = FixedPolicy(best_actions)
    return policy, reward_curve


def main():

    num_episodes = 20
    report_period = 10
    num_steps = 150
    epsilon = 1.

    num_domains = 1
    batch_size = 64

    env = PointBotEnv.sample_domains(num_domains)
    policy, reward_curve = epsilon_greedy(epsilon, num_episodes, num_steps, env, batch_size, report_period)

    pt.ioff()

    pt.plot(reward_curve)
    pt.ylabel("reward")

    pt.show()
    pt.close()

    pt.figure()
    pt.ion()

    env.animate(policy, num_steps, ax=pt.gca(), reset_batch_size=1)

if __name__ == "__main__": main()



