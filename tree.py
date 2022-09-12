import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from matplotlib.collections import PatchCollection
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
        # assumes same number of nodes are added/deleted for each domain
        return np.stack([np.flatnonzero(self.used[d]) for d in range(self.env.num_domains)])

    def get_states(self, nodes):
        # nodes: (D, B)
        return np.stack([self.states[d, nodes[d]] for d in range(self.env.num_domains)])

    def get_actions(self, nodes):
        # nodes: (D, B)
        return np.stack([self.actions[d, nodes[d]] for d in range(self.env.num_domains)])

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
        used_idx = self.get_nodes()
        child_nodes = {d: used_idx[d, self.parents[d, used_idx[d]] == nodes[d]] for d in range(self.env.num_domains)}
        child_states = {d: self.states[d, child_nodes[d]] for d in range(self.env.num_domains)}
        child_actions = {d: self.actions[d, child_nodes[d]] for d in range(self.env.num_domains)}
        return child_nodes, child_states, child_actions

    def parents_of(self, nodes):
        # nodes: (D,)
        parent_nodes = np.take_along_axis(self.parents, nodes[:,np.newaxis], axis=1).squeeze(axis=1) # (D,)
        parent_states = np.take_along_axis(self.states, parent_nodes[:,np.newaxis,np.newaxis], axis=1).squeeze(axis=1) # (D, S)
        parent_states[parent_nodes == -1] = np.nan
        return parent_nodes, parent_states

if __name__ == "__main__":

    num_steps = 150
    num_domains = 1
    capacity = 256
    branching = 1
    walk_stdev = 0.1

    env = PointBotEnv.sample_domains(num_domains)
    tree = StateActionTree(env, capacity)

    for itr in range(capacity-1):
        print(itr)
        
        nodes = tree.get_nodes()
        states = tree.get_states(nodes)
        actions = tree.get_actions(nodes)

        # get radii of each ball
        ball_radii = np.zeros(nodes.shape)
        for n in range(nodes.shape[1]):
            child_nodes, child_states, _ = tree.children_of(nodes[:,n])
            parent_nodes, parent_states = tree.parents_of(nodes[:,n])
            for d in range(env.num_domains):
                if len(child_nodes[d]) > 0:
                    child_dists = np.linalg.norm(child_states[d] - states[d,n], axis=-1) # (B,)
                    ball_radii[d,n] = child_dists.max()
                if parent_nodes[d] > -1:
                    parent_dist = np.linalg.norm(parent_states[d] - states[d,n])
                    ball_radii[d,n] = max(parent_dist, ball_radii[d,n])

        # count nodes in each ball and balls covering each node
        all_dists = np.linalg.norm(states[:,:,np.newaxis,:] - states[:,np.newaxis,:,:], axis=-1) # (D, B, B)
        ball_counts = (all_dists <= ball_radii[:,:,np.newaxis]).sum(axis=-1) # (D, B)
        cover_counts = (all_dists <= ball_radii[:,np.newaxis,:]).sum(axis=-1) # (D, B)

        # count children sampled at each state
        child_counts = np.empty(nodes.shape)
        for n in range(nodes.shape[1]):
            child_nodes, _, _ = tree.children_of(nodes[:,n])
            for d in range(env.num_domains):
                child_counts[d,n] = len(child_nodes[d])

        # count actions sampled in each ball
        action_counts = ((all_dists <= ball_radii[:,:,np.newaxis]) * child_counts[:,np.newaxis,:]).sum(axis=-1) # (D, B)

        # select nodes
        pop = cover_counts.argmin(axis=1) # covered by fewest balls
        # pop = action_counts.argmin(axis=1) # covering the fewest actions
        pop_nodes = np.take_along_axis(nodes, pop[:,np.newaxis], axis=1)
        pop_states = tree.get_states(pop_nodes)

        # pop_actions = np.random.rand(num_domains, branching, env.act_size)
        pop_actions = tree.get_actions(pop_nodes)
        pop_actions[pop_nodes == 0] = np.random.rand((pop_nodes==0).sum(), env.act_size)
        pop_actions.reshape(num_domains, 1, env.act_size)
        pop_actions += np.random.randn(num_domains, branching, env.act_size) * walk_stdev

        # add children
        env.reset(state = pop_states)
        child_states, _, _, _ = env.step(pop_actions)
        child_nodes = tree.add_children(pop_nodes, child_states, pop_actions)

    nodes = tree.get_nodes()
    states = tree.get_states(nodes)

    # get radii of each ball
    ball_radii = np.zeros(nodes.shape)
    viz_radii = np.zeros(nodes.shape)
    for n in range(nodes.shape[1]):
        child_nodes, child_states, _ = tree.children_of(nodes[:,n])
        parent_nodes, parent_states = tree.parents_of(nodes[:,n])
        for d in range(env.num_domains):
            if len(child_nodes[d]) > 0:
                child_dists = np.linalg.norm(child_states[d] - states[d,n], axis=-1) # (B,)
                ball_radii[d,n] = child_dists.max()
                viz_radii[d,n] = np.linalg.norm(child_states[d][:,:2] - states[d,n,:2], axis=-1).max()
            if parent_nodes[d] > -1:
                parent_dist = np.linalg.norm(parent_states[d] - states[d,n])
                ball_radii[d,n] = max(parent_dist, ball_radii[d,n])
                viz_radii[d,n] = max(np.linalg.norm(parent_states[d,:2] - states[d,n,:2]), viz_radii[d,n])

    # count nodes in each ball and balls covering each node
    all_dists = np.linalg.norm(states[:,:,np.newaxis,:] - states[:,np.newaxis,:,:], axis=-1) # (D, B, B)
    ball_counts = (all_dists <= ball_radii[:,:,np.newaxis]).sum(axis=-1) # (D, B)
    cover_counts = (all_dists <= ball_radii[:,np.newaxis,:]).sum(axis=-1) # (D, B)

    viz_dists = np.linalg.norm(states[:,:,np.newaxis,:2] - states[:,np.newaxis,:,:2], axis=-1) # (D, B, B)
    viz_counts = (viz_dists <= viz_radii[:,:,np.newaxis]).sum(axis=-1) # (D, B)
    viz_covers = (viz_dists <= viz_radii[:,np.newaxis,:]).sum(axis=-1) # (D, B)

    xpt, ypt, g = env.gravity_mesh()
    # sx, sy, sr, sc = states[:,:,0].flatten(), states[:,:,1].flatten(), viz_radii.flatten(), viz_counts.flatten()
    sx, sy, sr, sc = states[:,:,0].flatten(), states[:,:,1].flatten(), viz_radii.flatten(), viz_covers.flatten()
    pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))

    balls = [pt.Circle((x,y), r) for (x,y,r) in zip(sx, sy, sr)]
    pt.gca().add_collection(PatchCollection(balls, edgecolors='b', facecolors='none'))

    pt.plot(sx, sy, 'go', markersize=2)

    # for (x, y, c) in zip(sx, sy, sc): pt.text(x, y, str(c))


    pt.show()

