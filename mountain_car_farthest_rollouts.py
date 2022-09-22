import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection, PatchCollection
import gym

def draw(states, tails):

    num_steps, num_rollouts = states.shape[:2]

    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for r in range(num_rollouts):
        segments = np.stack((states[:tails[r], r], states[1:tails[r]+1, r]), axis=1)
        pt.gca().add_collection(LineCollection(segments, colors=colors[:len(segments)], linestyle='-'))
        pt.scatter(states[:tails[r]+1,r,0], states[:tails[r]+1,r,1], c=colors[:tails[r]+1], marker='.')

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()

def main():

    num_steps = 200
    num_rollouts = 8
    sampling = 64
    num_reps = 100

    rng = np.random.default_rng()

    env = gym.make("MountainCarContinuous-v0").unwrapped
    init_state, _ = env.reset()

    states = np.empty((num_steps+1, num_rollouts, 2))
    states[0,:,:] = np.broadcast_to(init_state, (num_rollouts, 2))
    tails = np.zeros(num_rollouts, dtype=int)

    pt.ion()

    for rep in range(num_reps):
        print(f"rep {rep} of {num_reps}")

        # rerun rollouts to end
        for r in range(num_rollouts):
            env.state = states[tails[r], r].copy()
            actions = rng.uniform(-1, 1, (num_steps - tails[r], 1)).astype(np.float32)
            for t in range(tails[r], num_steps):
                states[t+1, r], _, _, _, _ = env.step(actions[t - tails[r]])

        pt.cla()
        draw(states, tails)
        pt.pause(0.1)

        # farthest points
        # as shown in https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        # originally from https://doi.org/10.1016/0304-3975(85)90224-5
        states = states.reshape(-1,2)
        dists = np.linalg.norm(states[:,np.newaxis,:] - states[np.newaxis,:,:], axis=2)
        subsample = list(np.unravel_index(dists.argmax(), dists.shape))
        for p in range(2, sampling):
            s = dists[subsample].min(axis=0).argmax()
            subsample.append(s)
        subsample = np.array(subsample)
        states = states.reshape(num_steps+1, num_rollouts, 2)

        # recompute tails
        steps, rollouts = np.unravel_index(subsample, (num_steps+1, num_rollouts))
        tails = np.zeros(num_rollouts, dtype=int)
        for t,r in zip(steps, rollouts):
            tails[r] = max(tails[r], t)

        print(tails)
        if (tails >= num_steps).all():
            print("All tails sampled, stopping")

        pt.cla()
        draw(states, tails)

        # input('.')

    env.close()

    pt.ioff()
    pt.cla()
    pt.title("Done...")
    draw(states, tails)

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()







