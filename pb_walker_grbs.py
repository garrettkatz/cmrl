"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/walker2d/
"""
import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection, PatchCollection
import gym
import pybullet as pb
import pybullet_envs

def draw(states):

    num_steps = len(states)
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for t in range(len(states)):
        # TODO: use first two principle components instead of coordinates
        pt.scatter(states[t][:,0], states[t][:,1], color=colors[t], marker='.')

    # pt.xlim([-1.2, 0.6])
    # pt.ylim([-0.07, 0.07])
    # pt.xlabel("Position")
    # pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()

def run():

    num_steps = 999
    branching = 4
    beam = 8
    sampling = 8
    obs_size = 22 # apparently pb obs space larger than mujoco 17
    act_size = 6

    env = gym.make('Walker2DBulletEnv-v0')
    # env.render(mode="human")

    init_state = env.reset()
    init_sid = pb.saveState()

    rng = np.random.default_rng()

    states = [init_state[np.newaxis,:]]
    sids = [np.array([init_sid])]
    actions = [None]

    pt.ion()
    total_steps = 0
    for t in range(num_steps):
        print(f"step {t} of {num_steps}...")

        P = len(states[t])
        child_states = np.empty((P, branching, obs_size), dtype=np.float32)
        child_sids = np.empty((P, branching), dtype=int)
        child_actions = rng.uniform(-1, 1, (P, branching, act_size)).astype(np.float32)

        for p in range(P):
            for b in range(branching):
                # reset state
                env.state_id = sids[t][p]
                env.reset()
                # take step
                child_states[p,b], _, _, _ = env.step(np.clip(child_actions[p,b], -1, 1))
                child_sids[p,b] = pb.saveState()
                total_steps += 1
                # input(f"t={t}, p={p}, b={b}")

        child_states = child_states.reshape(P*branching, obs_size)
        child_sids = child_sids.reshape(P*branching)
        child_actions = child_actions.reshape(P*branching, act_size)

        # clean up no-longer-needed parent pb states
        for p in range(P):
            pb.removeState(sids[t][p])

        if len(child_states) < beam:
            states.append(child_states)
            sids.append(child_sids)
            actions.append(child_actions)
            continue

        # set-difference farthest point algorithm
        # modified from
        #    https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        #    https://doi.org/10.1016/0304-3975(85)90224-5

        previous_states = np.concatenate(states, axis=0)
        if len(previous_states) > sampling:
            subsample = np.random.choice(len(previous_states), size=sampling, replace=False)
            previous_states = previous_states[subsample]
        bkgd_states = np.concatenate((child_states, previous_states), axis=0)

        dists = np.linalg.norm(child_states[:,np.newaxis,:] - bkgd_states[np.newaxis,:,:], axis=2)
        included = np.ones(len(bkgd_states), dtype=bool)
        included[:len(child_states)] = False
        excluded = list(range(len(child_states)))

        a = dists[:,included].min(axis=1).argmax()
        included[excluded.pop(a)] = True
        
        for p in range(1, beam):
            a = dists[excluded][:,included].min(axis=1).argmax()
            included[excluded.pop(a)] = True

        new_actions = child_actions[included[:len(child_states)]]
        new_sids = child_sids[included[:len(child_states)]]
        new_states = child_states[included[:len(child_states)]]
        states.append(new_states)
        sids.append(new_sids)
        actions.append(new_actions)

        # clean up no-longer-needed child pb states
        for x in excluded:
            pb.removeState(child_sids[x])

        # pt.cla()
        # draw(states)
        # # input('.')

    env.close()

    # print(f"total steps = {num_steps}*{beam}*{branching} = {num_steps*beam*branching}")
    print(f"total steps = {num_steps}*{beam}*{branching} = {num_steps*beam*branching} =? {total_steps}")

    with open("pw_grbs.pkl","wb") as f: pk.dump(states, f)

def show():

    with open("pw_grbs.pkl","rb") as f: states = pk.load(f)

    pt.ioff()
    pt.cla()
    draw(states)


if __name__ == "__main__":
    run()
    show()

