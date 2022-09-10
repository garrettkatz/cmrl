import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from pointbotenv import PointBotEnv

class StateActionTree:
    def __init__(self, env, capacity):

        self.env = env
        self.capacity = capacity

        self.states = np.empty((env.num_domains, capacity, env.obs_size))
        self.actions = np.empty((env.num_domains, capacity, env.act_size))
        self.rewards = np.empty((env.num_domains, capacity))
        self.parents = np.empty((env.num_domains, capacity), dtype=int)

        self.states[:, :1] = env.reset(batch_size=1)
        self.rewards[:, 0] = 0
        self.parents[:, 0] = -1
        self.used = np.zeros((env.num_domains, 1)) == np.arange(capacity)

    def get_nodes(self):
        return np.stack([np.flatnonzero(self.used[d]) for d in range(self.env.num_domains)])

    def get_states(self, nodes):
        # nodes: (D, B)
        return np.stack([self.states[d, nodes[d]] for d in range(self.env.num_domains)])
        # return self.states[self.used].reshape(self.env.num_domains, -1, self.env.obs_size)

    def add_children(self, nodes, states, actions):
        # nodes: (D,)
        # states: (D, B, S)
        # actions: (D, B, A)
        # children: (D, B)
        children = np.empty(states.shape[:2], dtype=int)
        for d in range(len(nodes)):
            children[d] = np.flatnonzero(~self.used[d])[:len(children[d])]
            self.parents[d, children[d]] = nodes[d]
            self.states[d, children[d], :] = states[d]
            self.actions[d, children[d], :] = actions[d]
            self.used[d, children[d]] = True
        return children

    def children_of(self, nodes):
        # nodes: (D,)
        return {d: np.flatnonzero(self.parents[d, self.used] == node[d])}

    def parents_of(self, nodes):
        # nodes: (D,)
        return np.take_along_axis(self.parents, nodes[:,np.newaxis], axis=1)

if __name__ == "__main__":

    num_steps = 150
    num_domains = 1
    capacity = 1024

    env = PointBotEnv.sample_domains(num_domains)
    tree = StateActionTree(env, capacity)

    nodes = tree.get_nodes()
    for n in range(100):
        states = tree.get_states(nodes[:,:1])
        actions = np.random.randn(num_domains, 2, env.act_size)

        env.reset(state = states)
        new_states, _, _, _ = env.step(actions)
        nodes = tree.add_children(nodes[:,:1], new_states, actions)

    nodes = tree.get_nodes()
    states = tree.get_states(nodes)
    
    xpt, ypt, g = env.gravity_mesh()
    pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))
    pt.plot(states[:,:,0].flatten(), states[:,:,1].flatten(), 'bo', markersize=1)
    pt.show()

