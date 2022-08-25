import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from pointbotenv import PointBotEnv

class Scaffold:
    def __init__(self, capacity, num_steps, act_size, s0):

        num_domains, batch_size, obs_size = s0.shape
        num_s0 = num_domains * batch_size

        self.states = np.empty((capacity, obs_size))
        self.actions = np.empty((capacity, act_size))
        self.values = np.ones(capacity) * -np.inf
        self.rewards = np.empty(capacity)
        self.parents = np.empty(capacity, dtype=int)
        self.timesteps = np.empty(capacity, dtype=int)

        self.states[:num_s0] = s0.reshape(-1, obs_size)
        self.rewards[:num_s0] = 0
        self.parents[0] = -1
        self.timesteps[:num_s0] = 0
        self.children = {n: [] for n in range(num_s0)}
        self.num_nodes = num_s0

        self.num_steps = num_steps
        self.act_size = act_size

    def nearest(self, states): # (D, B, S)
        nonterms = np.flatnonzero(self.timesteps[:self.num_nodes] < self.num_steps-1) # exclude last time-step when no actions are taken
        diffs = np.expand_dims(self.states[nonterms], axis=(1,2)) - states # (N, D, B, S)
        dists = (diffs**2).sum(axis=-1) # (N, D, B)
        amins = dists.argmin(axis=0) # (D, B)
        return nonterms[amins.flat].reshape(amins.shape)

    def states_of(self, nodes):
        return self.states[nodes.flatten()].reshape(nodes.shape + (-1,))

    def timesteps_of(self, nodes):
        return self.timesteps[nodes.flatten()].reshape(nodes.shape)

    def policy_at(self, nodes):
        """
        nodes: (D, B) indices of states input to policy
        return actions: (D, B, A) array of actions
        """
        D, B, A = nodes.shape + (self.act_size,)
        actions = np.empty((D, B, A)).reshape(D*B, A)
        for n, node in enumerate(nodes.flat):
            child_idx = self.children[node]
            best = self.values[child_idx].argmax()
            actions[n] = self.actions[child_idx[best]]
        actions = actions.reshape(D, B, A)
        return actions

    def incorporate(self, init_nodes, timesteps, states, actions, rewards):
        """
        init_nodes: (D, B) indices of rollout inits
        timesteps: (D, B) timesteps of rollout inits
        states (T+1, D, B, S): states of rollouts
        actions, rewards (T, D, B, S): actions, rewards of rollouts
        T, D, B, S: timesteps, num_domains, batch_size, obs_size
        """
        T, D, B, S = states[1:].shape

        timesteps = timesteps[np.newaxis] + np.arange(T)[:,np.newaxis,np.newaxis]
        mask = np.flatnonzero(timesteps < self.num_steps)
        N, M = self.num_nodes, len(mask)

        new_nodes = np.empty((T, D, B), dtype=int)
        new_nodes.flat[mask] = N + np.arange(M)

        parents = np.empty((T, D, B), dtype=int)
        parents[0] = init_nodes
        parents[1:T] = new_nodes[:T-1]

        values = rewards.sum(axis=0, keepdims=True) - rewards.cumsum(axis=0) + rewards

        self.states[N:N+M] = states[1:].reshape(T*D*B, -1)[mask]
        self.actions[N:N+M] = actions.reshape(T*D*B, -1)[mask]
        self.values[N:N+M] = values.reshape(T*D*B)[mask]
        self.rewards[N:N+M] = rewards.reshape(T*D*B)[mask]
        self.timesteps[N:N+M] = timesteps.reshape(T*D*B)[mask]
        self.parents[N:N+M] = parents.reshape(T*D*B)[mask]

        for n in range(N, N+M): self.children[n] = []
        for n,p in enumerate(self.parents[N:N+M]):
            self.children[p].append(N+n)

        self.num_nodes = N + M

        # backpropagate values up the ancestors
        nodes = init_nodes
        while len(nodes) > 0:
            parents = self.parents[nodes]
            nonroots = (parents > -1)
            parents = parents[nonroots]
            self.values[parents] = np.maximum(self.values[parents], self.rewards[parents] + self.values[nodes[nonroots]])
            nodes = parents

if __name__ == "__main__":

    num_updates = 500
    report_period = 10
    num_steps = 150

    # Set up spring parameters for bot motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = np.random.uniform(.25, .9)*critical # random underdamping

    num_domains = 1
    batch_size = 8

    mass = m + np.random.randn(num_domains) * 0.1
    gravity = 10 + np.random.randn(num_domains)
    restore = k + np.random.randn(num_domains) * 0.1
    damping = b + np.random.randn(num_domains) * 0.1

    control_rate = 10
    dt = 1/240 * np.ones(num_domains)

    env = PointBotEnv(mass, gravity, restore, damping, control_rate, dt)

    obs_size, act_size = 4, 2

    # Initialize policy net
    net = tr.nn.Linear(obs_size, act_size)
    # hid_size = 16
    # net = tr.nn.Sequential(
    #     tr.nn.Linear(obs_size, hid_size),
    #     tr.nn.LeakyReLU(),
    #     tr.nn.Linear(hid_size, hid_size),
    #     tr.nn.LeakyReLU(),
    #     # tr.nn.Linear(hid_size, hid_size),
    #     # tr.nn.LeakyReLU(),
    #     # tr.nn.Linear(hid_size, hid_size),
    #     # tr.nn.LeakyReLU(),
    #     tr.nn.Linear(hid_size, act_size),
    # )

    # optimizer = tr.optim.SGD(net.parameters(), lr=0.01)
    optimizer = tr.optim.Adam(net.parameters(), lr=0.01)

    # Initialize scaffold
    s0 = env.reset(batch_size)
    scaffold = Scaffold(num_updates*num_domains*batch_size*num_steps, num_steps, act_size, s0)

    reward_curve = np.empty((num_updates,))
    error_curve = np.empty((num_updates,))

    for update in range(num_updates):

        # sample states in scaffold
        states = np.random.rand(num_domains, batch_size, obs_size)
        init_nodes = scaffold.nearest(states)
        timesteps = scaffold.timesteps_of(init_nodes)
        T = num_steps - timesteps.min()

        states = np.empty((T+1, num_domains, batch_size, obs_size))
        states[0] = scaffold.states_of(init_nodes)

        actions = np.empty((T, num_domains, batch_size, act_size))
        rewards = np.empty((T, num_domains, batch_size))

        # rollout from random action
        env.reset(state = states[0])
        for t in range(T):
            if t == 0:
                actions[t] = np.random.rand(num_domains, batch_size, act_size)
            else:
                with tr.no_grad(): actions[t] = env.bound(net(tr.tensor(states[t], dtype=tr.float)).numpy())
            states[t+1], rewards[t], _, _ = env.step(actions[t])

        # incorporate rollouts into scaffold
        scaffold.incorporate(init_nodes, timesteps, states, actions, rewards)

        # train net on scaffold
        states = np.random.rand(num_domains, batch_size, obs_size)
        train_nodes = scaffold.nearest(states)
        states = scaffold.states_of(train_nodes)
        actions = scaffold.policy_at(train_nodes)

        predictions = net(tr.tensor(states, dtype=tr.float))
        errors = tr.tensor(actions, dtype=tr.float) - predictions
        loss = tr.mean(errors**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error_curve[update] = errors.abs().max().item()

        reward_curve[update] = scaffold.values[(scaffold.timesteps == 0) & (scaffold.values > -np.inf)].mean() # initial states

        if update % report_period == 0:
            print(f"{update}/{num_updates}: " + \
                f"error={error_curve[update]}, " + \
                f"reward={reward_curve[update]}, " + \
            "")

    pt.ioff()

    pt.subplot(2,1,1)
    pt.plot(reward_curve)
    pt.ylabel("reward")

    pt.subplot(2,1,2)
    pt.plot(error_curve)
    pt.ylabel("error")

    pt.show()
    pt.close()

    pt.figure()
    pt.ion()
    with tr.no_grad():
        policy = lambda obs: (env.bound(net(tr.tensor(obs, dtype=tr.float)).numpy()), None)
        env.animate(policy, num_steps, ax=pt.gca(), reset_batch_size=batch_size)



