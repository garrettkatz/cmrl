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
    capacity = 128
    branching = 1
    walk_stdev = 0.1
    probe_size = 32
    convrate = .9

    env = PointBotEnv.sample_domains(num_domains)
    tree = StateActionTree(env, capacity)

    child_counts = np.zeros((num_domains, 1)) # total actions sampled at state
    ball_weights = np.ones((num_domains, 1)) # relative action sampling rate within ball
    ball_radii = np.inf*np.ones((num_domains, 1)) # radius of ball
    viz_radii = np.inf*np.ones((num_domains, 1)) # radius of ball

    for itr in range(capacity-1):
        print(f"{itr} of {capacity-1}")
        
        nodes = tree.get_nodes()
        states = tree.get_states(nodes)

        # pop least tried nodes
        pop = ball_weights.argmin(axis=1)
        pop_nodes = np.take_along_axis(nodes, pop[:,np.newaxis], axis=1)
        pop_states = tree.get_states(pop_nodes)

        # sample new action
        pop_actions = np.random.rand(num_domains, 1, env.act_size)
        # pop_actions = tree.get_actions(pop_nodes)
        # pop_actions[pop_nodes == 0] = np.random.rand((pop_nodes==0).sum(), env.act_size)
        # pop_actions.reshape(num_domains, 1, env.act_size)
        # pop_actions += np.random.randn(num_domains, branching, env.act_size) * walk_stdev

        # apply action to popped
        env.reset(state = pop_states)
        pop_child_states, _, _, _ = env.step(pop_actions)
        tree.add_children(pop_nodes, pop_child_states, pop_actions)

        # increment pop child count
        for d in range(num_domains): child_counts[d, pop[d]] += 1

        # update subsample of existing balls for speed
        probe = np.arange(states.shape[1])
        # probe = np.random.choice(states.shape[1], size=min(states.shape[1], probe_size), replace=False)

        # apply same action to probe
        env.reset(state = states[:,probe])
        probe_child_states, _, _, _ = env.step(pop_actions)

        # pair-wise distances
        viz_dists = np.linalg.norm(pop_states[:,:,np.newaxis,:2] - states[:,np.newaxis,probe,:2], axis=-1) # (D, 1, P)
        dists = np.linalg.norm(pop_states[:,:,np.newaxis,:] - states[:,np.newaxis,probe,:], axis=-1) # (D, 1, P)
        child_dists = np.linalg.norm(pop_child_states[:,:,np.newaxis,:] - probe_child_states[:,np.newaxis,:,:], axis=-1) # (D, 1, P)
        contracted = child_dists <= convrate * dists # (D, 1, P)

        # update radii
        viz_radii[:,probe] = np.minimum(viz_radii[:,probe], np.where(contracted, np.inf, viz_dists).min(axis=1)) # smallest radius that did not contract
        ball_radii[:,probe] = np.minimum(ball_radii[:,probe], np.where(contracted, np.inf, dists).min(axis=1)) # smallest radius that did not contract

        # update weights within probe
        dists = np.linalg.norm(states[:,probe,np.newaxis,:] - states[:,np.newaxis,probe,:], axis=-1) # (D, P, P)
        ball_weights[:,probe] = np.where(dists <= ball_radii[:,probe,np.newaxis], child_counts[:,np.newaxis,probe], 0).sum(axis=2) # (D, P)
        ball_weights[:,probe] /= child_counts[:,probe].sum(axis=1)

        # initial estimate of child ball weights
        dists = np.linalg.norm(pop_child_states[:,:,np.newaxis,:] - states[:,np.newaxis,probe,:], axis=-1) # (D, 1, P)
        child_weights = np.where(dists <= ball_radii[:,np.newaxis,probe], child_counts[:,np.newaxis,probe], 0).sum(axis=2) # (D, 1)
        child_weights /= child_counts[:,probe].sum(axis=1)

        # append balls for new children (assumes no nodes deleted)
        viz_radii = np.concatenate((viz_radii, np.inf*np.ones((num_domains, 1))), axis=1)
        ball_radii = np.concatenate((ball_radii, np.inf*np.ones((num_domains, 1))), axis=1)
        child_counts = np.concatenate((child_counts, np.zeros((num_domains, 1))), axis=1)
        ball_weights = np.concatenate((ball_weights, child_weights), axis=1)

    nodes = tree.get_nodes()
    states = tree.get_states(nodes)

    xpt, ypt, g = env.gravity_mesh()
    pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))

    # sx, sy, sr, sc = states[:,:,0].flatten(), states[:,:,1].flatten(), ball_radii.flatten(), child_counts.flatten()
    # balls = [pt.Circle((x,y), r**.5) for (x,y,r) in zip(sx, sy, sr) if r < np.inf] # sqrt of r since projecting 4d to 2d
    sx, sy, sr, sc = states[:,:,0].flatten(), states[:,:,1].flatten(), viz_radii.flatten(), child_counts.flatten()
    balls = [pt.Circle((x,y), r) for (x,y,r) in zip(sx, sy, sr) if r < np.inf]
    # pt.gca().add_collection(PatchCollection(balls, edgecolors='b', facecolors='none'))
    pt.gca().add_collection(PatchCollection(balls, edgecolors=[(0,0,1,1/(1+c)) for c in sc], facecolors='none'))

    pt.plot(sx, sy, 'go', markersize=2)

    # for (x, y, c) in zip(sx, sy, sc): pt.text(x, y, str(c))

    pt.show()



    # nodes = tree.get_nodes()
    # states = tree.get_states(nodes)

    # # get radii of each ball
    # ball_radii = np.zeros(nodes.shape)
    # viz_radii = np.zeros(nodes.shape)
    # for n in range(nodes.shape[1]):
    #     child_nodes, child_states, _ = tree.children_of(nodes[:,n])
    #     parent_nodes, parent_states = tree.parents_of(nodes[:,n])
    #     for d in range(env.num_domains):
    #         if len(child_nodes[d]) > 0:
    #             child_dists = np.linalg.norm(child_states[d] - states[d,n], axis=-1) # (B,)
    #             ball_radii[d,n] = child_dists.max()
    #             viz_radii[d,n] = np.linalg.norm(child_states[d][:,:2] - states[d,n,:2], axis=-1).max()
    #         if parent_nodes[d] > -1:
    #             parent_dist = np.linalg.norm(parent_states[d] - states[d,n])
    #             ball_radii[d,n] = max(parent_dist, ball_radii[d,n])
    #             viz_radii[d,n] = max(np.linalg.norm(parent_states[d,:2] - states[d,n,:2]), viz_radii[d,n])

    # # count nodes in each ball and balls covering each node
    # all_dists = np.linalg.norm(states[:,:,np.newaxis,:] - states[:,np.newaxis,:,:], axis=-1) # (D, B, B)
    # ball_counts = (all_dists <= ball_radii[:,:,np.newaxis]).sum(axis=-1) # (D, B)
    # cover_counts = (all_dists <= ball_radii[:,np.newaxis,:]).sum(axis=-1) # (D, B)

    # viz_dists = np.linalg.norm(states[:,:,np.newaxis,:2] - states[:,np.newaxis,:,:2], axis=-1) # (D, B, B)
    # viz_counts = (viz_dists <= viz_radii[:,:,np.newaxis]).sum(axis=-1) # (D, B)
    # viz_covers = (viz_dists <= viz_radii[:,np.newaxis,:]).sum(axis=-1) # (D, B)

    # xpt, ypt, g = env.gravity_mesh()
    # # sx, sy, sr, sc = states[:,:,0].flatten(), states[:,:,1].flatten(), viz_radii.flatten(), viz_counts.flatten()
    # sx, sy, sr, sc = states[:,:,0].flatten(), states[:,:,1].flatten(), viz_radii.flatten(), viz_covers.flatten()
    # pt.contourf(xpt, ypt, g, levels = 100, colors = np.array([1,1,1]) - np.linspace(0, 1, 100)[:,np.newaxis] * np.array([0,1,1]))

    # balls = [pt.Circle((x,y), r) for (x,y,r) in zip(sx, sy, sr)]
    # pt.gca().add_collection(PatchCollection(balls, edgecolors='b', facecolors='none'))

    # pt.plot(sx, sy, 'go', markersize=2)

    # # for (x, y, c) in zip(sx, sy, sc): pt.text(x, y, str(c))

    # pt.show()

