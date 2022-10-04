import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection, PatchCollection
import gym
import heapq

def draw(states):

    states = np.stack(states)

    # goal rectangle
    pt.gca().add_patch(mp.Rectangle((.45, 0), .15, .07, linewidth=0, facecolor='gray'))

    pt.scatter(states[:,0], states[:,1], color='r', marker='.')

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()

def transit(env, state, action):
    env.state = state.copy()
    new_state, _, _, _ = env.step(np.clip(action, -1, 1))
    return new_state

def run():

    branching = 4
    probe_size = 3
    capacity = 10000

    env = gym.make("MountainCarContinuous-v0").unwrapped
    rng = np.random.default_rng()

    init = env.reset()
    frontier = [(0, init)]
    explored = []
    total_steps = 0

    # while condition:
    while len(frontier) < capacity:

        # pop lowest cost pt from frontier
        (cost, state) = heapq.heappop(frontier)

        # put in explored
        explored.append(state)

        # generate children
        for b in range(branching):

            child = transit(env, state, rng.uniform(-1, 1, (1,)))
            total_steps += 1

            # calculate priority for each child: distance to closest explored pt
            probe = rng.choice(explored, min(len(explored), probe_size))
            distance = np.linalg.norm(child - probe, axis=1).min()
            cost = -distance # larger distance is better (lower cost)

            # put child in frontier
            heapq.heappush(frontier, (cost, child))

        print(f"|frontier|={len(frontier)} < {capacity}, total steps = {total_steps}")

    env.close()

    print(f"total steps = {total_steps}")

    with open("mcqf.pkl","wb") as f: pk.dump(explored, f)

def show():

    with open("mcqf.pkl","rb") as f: states = pk.load(f)

    pt.ioff()
    pt.cla()
    draw(states)


if __name__ == "__main__":
    run()
    show()



