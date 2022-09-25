import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection, PatchCollection
import gym

def draw(states, paths):
    for path in paths:
        path_states = np.stack(path)
        pt.plot(path_states[:,0], path_states[:,1], 'b-')

    # num_steps = max(map(len, paths))
    # spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    # colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    # # parent-child edges
    # for t in range(num_steps):
    #     # segments[i][j,:2] = (x,y) for jth point in ith line
    #     segments = np.stack((states[t-1][parents[t]], states[t]), axis=1)
    #     # pt.gca().add_collection(LineCollection(segments, colors=colors[t], linestyle='-', alpha=0.2))
    #     pt.gca().add_collection(LineCollection(segments, colors=colors[t], linestyle='-'))
    #     pt.plot(states[t][:,0], states[t][:,1], '.', color=colors[t])

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

    num_steps = 1000
    branching = 5
    sampling = 64

    rng = np.random.default_rng()

    env = gym.make("MountainCarContinuous-v0").unwrapped
    init_state, _ = env.reset()

    states = init_state[np.newaxis,:]
    paths = [[init_state]]

    # pt.ion()

    for t in range(num_steps):
        print(f"{t} of {num_steps}")

        P = len(states)
        child_actions = rng.uniform(-1, 1, (P, branching, 1)).astype(np.float32)
        child_states = np.empty((P, branching, 2), dtype=np.float32)
        child_paths = []

        for p in range(P):
            for b in range(branching):
                env.state = states[p].copy()
                child_states[p,b], _, _, _, _ = env.step(child_actions[p,b])
                child_paths.append(paths[p] + [child_states[p,b]])
        child_states = child_states.reshape(-1, 2)

        all_states = np.concatenate((states, child_states), axis=0)
        all_paths = paths + child_paths

        # farthest point algorithm
        # as shown in https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        # originally from https://doi.org/10.1016/0304-3975(85)90224-5
        if sampling >= len(all_states):
            subsample = np.arange(len(all_states))
        else:
            dists = np.linalg.norm(all_states[:,np.newaxis,:] - all_states[np.newaxis,:,:], axis=2)
            subsample = list(np.unravel_index(dists.argmax(), dists.shape))
            for p in range(2, sampling):
                s = dists[subsample].min(axis=0).argmax()
                subsample.append(s)
            subsample = np.array(subsample)

        states = all_states[subsample]
        paths = [all_paths[s] for s in subsample]

        # draw(states, paths)

    env.close()

    pt.ioff()
    draw(states, paths)

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()





