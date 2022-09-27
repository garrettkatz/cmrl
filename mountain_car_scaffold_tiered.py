import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection, PatchCollection
import gym

def draw(states):

    num_steps = len(states)
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for t in range(len(states)):
        pt.scatter(states[t][:,0], states[t][:,1], color=colors[t], marker='.')

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()

def main():

    num_steps = 200
    branching = 4
    beam = 16
    sampling = 16

    rng = np.random.default_rng()

    env = gym.make("MountainCarContinuous-v0").unwrapped
    init_state, _ = env.reset()

    states = [init_state[np.newaxis,:]]

    pt.ion()

    for t in range(num_steps):
        print(f"step {t} of {num_steps}...")

        P = len(states[t])
        child_actions = rng.uniform(-1, 1, (P, branching, 1)).astype(np.float32)
        child_states = np.empty((P, branching, 2), dtype=np.float32)

        for p in range(P):
            for b in range(branching):
                env.state = states[t][p].copy()
                child_states[p,b], _, _, _, _ = env.step(child_actions[p,b])
        child_states = child_states.reshape(-1, 2)
        C = len(child_states)

        if C < beam:
            states.append(child_states)
            continue

        # set-difference farthest point algorithm
        # modified from
        #    https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        #    https://doi.org/10.1016/0304-3975(85)90224-5

        all_states = np.concatenate([child_states] + states, axis=0)
        dists = np.linalg.norm(child_states[:,np.newaxis,:] - all_states[np.newaxis,:,:], axis=2)
        included = np.ones(len(all_states), dtype=bool)
        included[:C] = False
        A = len(all_states) - C
        if A > sampling:
            idx = np.random.choice(A, size=sampling, replace=False)
            included[C:] = False
            included[C + idx] = True
        excluded = list(range(C))

        a = dists[:,included].min(axis=1).argmax()
        included[excluded.pop(a)] = True
        
        for p in range(1, beam):
            a = dists[excluded][:,included].min(axis=1).argmax()
            included[excluded.pop(a)] = True

        subsample = np.flatnonzero(included[:C])
        states.append(child_states[subsample])

        # pt.cla()
        # draw(states)
        # # input('.')

    env.close()

    pt.ioff()
    pt.cla()
    draw(states)

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()

