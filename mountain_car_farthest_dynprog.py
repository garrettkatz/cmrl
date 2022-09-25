import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection, PatchCollection
import gym

def draw(states, caps):
    # states[t].shape == (s,t+1,2)

    num_steps = len(states)+1
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    # # for t in range(len(states)):
    # for t in [len(states)-1]:
    #     for p, path in enumerate(states[t][:caps[t]]):
    #         # segments[i][j,:2] = (x,y) for jth point in ith line
    #         segments = np.stack((path[:-1], path[1:]), axis=1)
    #         pt.gca().add_collection(LineCollection(segments, colors=colors[:len(segments)], linestyle='-'))
    #         pt.scatter(path[:,0], path[:,1], c=colors[:len(path)], marker='.')
    #         pt.text(path[-1,0], path[-1,1], str(p))
    #     pt.scatter(states[t][:,-1,0], states[t][:,-1,1], color=colors[t+1])

    for t in range(len(states)):
        pt.scatter(states[t][:,-1,0], states[t][:,-1,1], color=colors[t+1], marker='.')

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()

def main():

    num_steps = 5
    branching = 8
    per_step_cap = 64
    capacity = 512

    assert capacity >= num_steps # keep at least one point for each timestep

    rng = np.random.default_rng()

    env = gym.make("MountainCarContinuous-v0").unwrapped
    init_state, _ = env.reset()

    states = [init_state[np.newaxis,np.newaxis,:]] # [t].shape == s,t+1,2
    max_gaps = [np.zeros(1)]
    min_gaps = [np.ones(1)]
    gap_ratios = [np.zeros(1)]

    # dyn prog [T, C]: minmax ratio after T timesteps with C points total
    minmax_ratio = {(1, 1): gap_ratios[0][0]}
    minmax_caps = {(1, 1): [1]}

    pt.ion()

    for T in range(1,num_steps):
        print(f"step {T} of {num_steps}...")

        counts = list(map(len, states))
        full_usage = min(capacity, sum(counts))
        P = minmax_caps[T, full_usage][T-1]
        # P = len(states[T-1])

        child_actions = rng.uniform(-1, 1, (P, branching, 1))
        child_states = np.empty((P, branching, T+1, 2))

        for p in range(P):
            for b in range(branching):
                env.state = states[T-1][p, T-1].copy()
                child_states[p,b,:T] = states[T-1][p]
                child_states[p,b, T], _, _, _, _ = env.step(child_actions[p,b])
        states.append(child_states.reshape(-1, T+1, 2))

        # farthest point algorithm on new states
        # as shown in https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        # originally from https://doi.org/10.1016/0304-3975(85)90224-5
        sample_cap = min(per_step_cap, len(states[T]))
        max_gaps.append(np.empty(sample_cap))
        min_gaps.append(np.ones(sample_cap))
        dists = np.linalg.norm(states[T][:,np.newaxis,T,:] - states[T][np.newaxis,:,T,:], axis=2)
        sort_idx = list(np.unravel_index(dists.argmax(), dists.shape))
        max_gaps[T][0] = dists[tuple(sort_idx)]
        min_gaps[T][1] = max_gaps[T][0] / 2
        for p in range(2, sample_cap+1):
            nearest = dists[sort_idx].argmin(axis=0)
            s = dists[sort_idx].min(axis=0).argmax()
            max_gaps[T][p-1] = dists[sort_idx[nearest[s]], s]
            if p < sample_cap:
                min_gaps[T][p] = max_gaps[T][p-1] / 2
                sort_idx.append(s)
        gap_ratios.append(max_gaps[T] / min_gaps[T])
        states[T] = states[T][sort_idx]

        # dynamic program to decide which states to keep
        counts = list(map(len, states))
        full_usage = min(capacity, sum(counts))
        max_cap = 0
        for C in range(T+1, full_usage+1):
            sub_ratios = []
            sub_caps = []
            for k in range(max(1, C - sum(counts[:-1])), min(len(states[T]), C - T) + 1):
                sub_ratios.append(max(gap_ratios[T][k - 1], minmax_ratio[T, C - k]))
                sub_caps.append(minmax_caps[T, C - k] + [k])
            best = np.argmin(sub_ratios)
            minmax_ratio[T+1, C] = sub_ratios[best]
            minmax_caps[T+1, C] = sub_caps[best]
            max_cap = max(max_cap, sub_caps[best][T])

        # use caps to avoid beam explosion
        states[T] = states[T][:max_cap]
        mnx_caps = minmax_caps[T+1, full_usage]
        print('counts:')
        print(counts)
        print('minmax caps:')
        print(mnx_caps, f" ~> {sum(mnx_caps)}")
        print("max, min gaps, ratios:")
        print(max_gaps[T])
        print(min_gaps[T])
        print(gap_ratios[T])
        print('gap ratios:')
        # print(["%.2f" % gap_ratios[t][mnx_caps[t]-1] for t in range(T+1)], f" ~> {sum(mnx_caps)}")
        print(["%e" % gap_ratios[t][mnx_caps[t]-1] for t in range(T+1)], f" ~> {sum(mnx_caps)}")

        if np.isnan(np.array([gap_ratios[t][mnx_caps[t]-1] for t in range(T+1)])).any():
            pass

        # if T % 1 == 0:
        #     pt.cla()
        #     draw(states, mnx_caps)
        #     # input('.')

    env.close()

    pt.ioff()
    pt.cla()
    draw(states, minmax_caps[num_steps, capacity])

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()


