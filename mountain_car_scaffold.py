import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection, PatchCollection
import gym

def draw(states, parents, radii):
    num_steps = len(parents)
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    # parent-child edges
    for t in range(num_steps):
        # segments[i][j,:2] = (x,y) for jth point in ith line
        segments = np.stack((states[t-1][parents[t]], states[t]), axis=1)
        # pt.gca().add_collection(LineCollection(segments, colors=colors[t], linestyle='-', alpha=0.2))
        pt.gca().add_collection(LineCollection(segments, colors=colors[t], linestyle='-'))
        pt.plot(states[t][:,0], states[t][:,1], '.', color=colors[t])

    # # child circles
    # for t in range(num_steps):
    #     balls = [pt.Circle(xy, r) for (xy,r) in zip(states[t], radii[t])]
    #     pt.gca().add_collection(PatchCollection(balls, edgecolors=colors[t], facecolors='none'))

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")

    pt.pause(0.1)
    pt.show()

def main():

    num_steps = 200
    branching = 4
    sampling = 32

    rng = np.random.default_rng()

    env = gym.make("MountainCarContinuous-v0").unwrapped
    init_state, _ = env.reset()

    states = {-1: init_state[np.newaxis,:]}
    rewards = {}
    actions = {}
    parents = {}
    radii = {-1: np.zeros(1)}

    # pt.ion()

    for t in range(num_steps):

        P = len(states[t-1])
        child_actions = rng.uniform(-1, 1, (P, branching, 1)).astype(np.float32)
        child_states = np.empty((P, branching, 2), dtype=np.float32)
        child_rewards = np.empty((P, branching), dtype=np.float32)

        for p in range(P):
            for b in range(branching):
                env.state = states[t-1][p].copy()
                child_states[p,b], child_rewards[p,b], _, _, _ = env.step(child_actions[p,b])

        dists = np.linalg.norm(child_states[:,:,np.newaxis,:] - child_states[:,np.newaxis,:,:], axis=3)
        dists += np.diag(np.inf*np.ones(branching))[np.newaxis,:,:] # omit distance to self
        min_dists = dists.min(axis=(1,2))
        child_radii = np.broadcast_to(min_dists[:,np.newaxis], (P, branching))

        child_actions = child_actions.reshape(-1, 1)
        child_states = child_states.reshape(-1, 2)
        child_rewards = child_rewards.reshape(-1)
        child_radii = child_radii.reshape(-1)

        # # random subsample
        # subsample = rng.choice(len(child_states), size=min(len(child_states), sampling), replace=False)

        # farthest point algorithm
        # as shown in https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        # originally from https://doi.org/10.1016/0304-3975(85)90224-5
        if sampling >= len(child_states):
            subsample = np.arange(len(child_states))
        else:
            dists = np.linalg.norm(child_states[:,np.newaxis,:] - child_states[np.newaxis,:,:], axis=2)
            subsample = list(np.unravel_index(dists.argmax(), dists.shape))
            for p in range(2, sampling):
                s = dists[subsample].min(axis=0).argmax()
                subsample.append(s)
            subsample = np.array(subsample)

        states[t] = child_states[subsample]
        actions[t] = child_actions[subsample]
        rewards[t] = child_rewards[subsample]
        radii[t] = child_radii[subsample]
        parents[t] = subsample // branching

        # draw(states, parents, radii)

    env.close()

    pt.ioff()
    draw(states, parents, radii)

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()



