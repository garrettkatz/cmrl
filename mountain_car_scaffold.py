import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection
import gym

def main():

    num_steps = 200
    branching = 5
    sampling = 30

    rng = np.random.default_rng()

    env = gym.make("MountainCarContinuous-v0").unwrapped
    init_state, _ = env.reset()

    states = {-1: init_state[np.newaxis,:]}
    rewards = {}
    actions = {}
    parents = {}

    for t in range(num_steps):

        P = len(states[t-1])
        child_actions = rng.uniform(-1, 1, (P, branching, 1)).astype(np.float32)
        child_states = np.empty((P, branching, 2), dtype=np.float32)
        child_rewards = np.empty((P, branching), dtype=np.float32)

        for p in range(P):
            for b in range(branching):
                env.state = states[t-1][p].copy()
                child_states[p,b], child_rewards[p,b], _, _, _ = env.step(child_actions[p,b])

        child_actions = child_actions.reshape(-1, 1)
        child_states = child_states.reshape(-1, 2)
        child_rewards = child_rewards.reshape(-1)

        subsample = rng.choice(len(child_states), size=min(len(child_states), sampling), replace=False)
        states[t] = child_states[subsample]
        actions[t] = child_actions[subsample]
        rewards[t] = child_rewards[subsample]
        parents[t] = subsample // branching

    env.close()

    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for t in range(num_steps):
        # segments[i][j,:2] = (x,y) for jth point in ith line
        segments = np.stack((states[t-1][parents[t]], states[t]), axis=1)
        pt.gca().add_collection(LineCollection(segments, colors=colors[t], linestyle='-'))
        # pt.plot(states[t][:,0], states[t][:,1], 'o', color=colors[t])


    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")
    pt.show()

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()



