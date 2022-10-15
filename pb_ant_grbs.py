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

    # pca the states
    X = np.concatenate(states, axis=0)
    Xm = X.mean(axis=0)
    Xc = X - Xm
    w, v = np.linalg.eigh(Xc.T @ Xc)
    PC = v[:,-2:] # top-2 PCs
    

    num_steps = len(states)
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for t in range(len(states)):
        # project onto top-2 PCs
        proj = (states[t] - Xm) @ PC
        # use first two principle components instead of coordinates
        pt.scatter(proj[:10,0], proj[:10,1], color=colors[t], marker='.')

    # pt.xlim([-1.2, 0.6])
    # pt.ylim([-0.07, 0.07])
    # pt.xlabel("Position")
    # pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()

def run():

    num_steps = 999
    branching = 16
    beam = 16
    sampling = 16
    act_size = 8

    env = gym.make('AntBulletEnv-v0').unwrapped
    # env.render(mode="human")

    init_state = env.reset()
    init_sid = pb.saveState()
    pb.saveBullet("pa_grbs.blt")

    obs_size = len(init_state)

    rng = np.random.default_rng()

    states = [init_state[np.newaxis,:]]
    potentials = [np.array([env.potential])] # used for progress reward]
    sids = [np.array([init_sid])]
    actions = [None]
    utility = [np.array([0])]
    reward = [np.array([0])]
    parents = [None]
    
    first_rewards = np.empty((1, branching))

    total_steps = 0
    for t in range(num_steps):
        print(f"step {t} of {num_steps}, {len(states[t])} states, {utility[t].mean():.2f}+/-{utility[t].std():.2f}<{utility[t].max():.2f} reward")

        P = len(states[t])
        child_states = np.empty((P, branching, obs_size), dtype=np.float32)
        child_sids = np.empty((P, branching), dtype=int)
        child_potentials = np.empty((P, branching))
        child_actions = rng.uniform(-1, 1, (P, branching, act_size)).astype(np.float32)
        # child_actions = .5*np.ones((P, branching, act_size)).astype(np.float32)
        child_utility = np.empty((P, branching))
        child_reward = np.empty((P, branching))
        child_done = np.empty((P, branching), dtype=bool)
        child_parents = np.empty((P, branching), dtype=int)

        for p in range(P):
            for b in range(branching):
                # reset state
                pb.restoreState(sids[t][p])
                env.potential = potentials[t][p]
                # env.state_id = sids[t][p]
                # env.reset()

                # take step
                child_states[p,b], child_reward[p,b], child_done[p,b], _ = env.step(np.clip(child_actions[p,b], -1, 1))
                total_steps += 1

                if t==0:
                    first_rewards[p,b] = child_reward[p,b]
                    # print(p,b,env.rewards)

                child_potentials[p,b] = env.potential
                child_utility[p,b] = utility[t][p] + child_reward[p,b]
                child_parents[p,b] = p

                # if child_done[p,b]:
                #     print(p, b, 'done')
                #     print(states[t][p][0], env.robot.initial_z, env.robot.body_rpy)

                if not child_done[p,b]: child_sids[p,b] = pb.saveState()
                # input(f"t={t}, p={p}, b={b}")

        child_states = child_states.reshape(P*branching, obs_size)
        child_sids = child_sids.reshape(P*branching)
        child_potentials = child_potentials.reshape(P*branching)
        child_done = child_done.reshape(P*branching)
        child_reward = child_reward.reshape(P*branching)
        child_utility = child_utility.reshape(P*branching)
        child_parents = child_parents.reshape(P*branching)
        child_actions = child_actions.reshape(P*branching, act_size)

        # # clean up no-longer-needed parent pb states (except for root)
        # if t > 0:
        #     for p in range(P):
        #         pb.removeState(sids[t][p])

        # stop early if all children done
        if child_done.all():
            print("all children done")
            break
        else:
            print(f" {child_done.sum()} of {len(child_done)} children done")

        # remove children where rollout is done
        child_states = child_states[~child_done]
        child_sids = child_sids[~child_done]
        child_potentials = child_potentials[~child_done]
        child_reward = child_reward[~child_done]
        child_utility = child_utility[~child_done]
        child_parents = child_parents[~child_done]
        child_actions = child_actions[~child_done]

        if len(child_states) < beam:
            states.append(child_states)
            sids.append(child_sids)
            potentials.append(child_potentials)
            parents.append(child_parents)
            reward.append(child_reward)
            utility.append(child_utility)
            actions.append(child_actions)
            continue

        # set-difference farthest point algorithm
        # modified from
        #    https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        #    https://doi.org/10.1016/0304-3975(85)90224-5

        C = len(child_states)

        previous_states = np.concatenate(states, axis=0)
        previous_utility = np.concatenate(utility)
        if len(previous_states) > sampling:
            subsample = np.random.choice(len(previous_states), size=sampling, replace=False)
            previous_states = previous_states[subsample]
            previous_utility = previous_utility[subsample]

        cvs = child_states
        pvs = previous_states

        # uniform state-value space
        cvs = np.concatenate((child_states, child_utility[:,np.newaxis]), axis=-1)
        pvs = np.concatenate((previous_states, previous_utility[:,np.newaxis]), axis=-1)

        bvs = np.concatenate((cvs, pvs), axis=0)
        dists = np.linalg.norm(cvs[:,np.newaxis,:] - bvs[np.newaxis,:,:], axis=2)

        included = np.ones(len(bvs), dtype=bool)
        included[:C] = False
        excluded = list(range(C))

        # # keep one elite
        # a = child_utility.argmax()
        # included[excluded.pop(a)] = True

        # start with farthest from all prior beams
        a = dists[:,included].min(axis=1).argmax()
        included[excluded.pop(a)] = True
        
        for p in range(1, beam):
            a = dists[excluded][:,included].min(axis=1).argmax()
            included[excluded.pop(a)] = True

        beam_actions = child_actions[included[:C]]
        beam_reward = child_reward[included[:C]]
        beam_utility = child_utility[included[:C]]
        beam_parents = child_parents[included[:C]]
        beam_sids = child_sids[included[:C]]
        beam_potentials = child_potentials[included[:C]]
        beam_states = child_states[included[:C]]

        states.append(beam_states)
        sids.append(beam_sids)
        potentials.append(beam_potentials)
        parents.append(beam_parents)
        reward.append(beam_reward)
        utility.append(beam_utility)
        actions.append(beam_actions)

        # clean up no-longer-needed child pb states
        for x in excluded:
            pb.removeState(child_sids[x])

        # pt.cla()
        # draw(states)
        # # input('.')

    print(f"t={num_steps}: {len(states[-1])} states, {utility[-1].mean():.2f}+/-{utility[-1].std():.2f}<{utility[-1].max():.2f} reward")
    # print(f"total steps = {num_steps}*{beam}*{branching} = {num_steps*beam*branching}")
    print(f"total steps = {num_steps}*{beam}*{branching} = {num_steps*beam*branching} =? {total_steps}")

    # save best final state
    pb.restoreState(sids[-1][utility[-1].argmax()])
    pb.saveBullet("pa_grbs_final.blt")

    # extract best plan
    b = utility[-1].argmax()
    plan = []
    for t in reversed(range(1,len(parents))):
        # print(f"rev {t}: b={b}, u={utility[t][b]}, r={reward[t][b]}, p={parents[t][b]}")
        # print("s",states[t][b,:4])
        # print("a", actions[t][b])
        plan.insert(0, actions[t][b])
        b = parents[t][b]

    # print("first rewards")
    # print(first_rewards)
    # print("first states")
    # print(states[1][:,:4])

    # run forward and check rewards again
    # env.reset()
    pb.restoreState(init_sid)
    env.potential = potentials[0][0]
    utility = 0
    for t,action in enumerate(plan):
        obs, reward, done, _ = env.step(action)
        utility += reward
        # print(f"t={t+1}, u={utility}, r={reward}")
        # print("s", obs[:4])
        # print("a", action)
        # print("r", env.rewards)

    env.close()

    with open("pa_grbs.pkl","wb") as f: pk.dump((states, potentials, plan), f)

def show():

    with open("pa_grbs.pkl","rb") as f: (states, potentials, plan) = pk.load(f)

    input('..')
    env = gym.make('AntBulletEnv-v0').unwrapped
    env.render(mode="human")
    env.reset()
    for comp in range(4):
        pb.restoreState(fileName="pa_grbs_final.blt")
        time.sleep(.5)
        pb.restoreState(fileName="pa_grbs.blt")
        time.sleep(.5)
    env.potential = potentials[0][0]

    utility = 0
    for t,action in enumerate(plan):
        obs, rew, done, _ = env.step(action)
        utility += rew
        print(f"t={t}, r={rew}, u={utility}")
        time.sleep(1/48.)
    env.close()
    print(f"total reward = {utility}")

    pt.ioff()
    pt.cla()
    draw(states)


if __name__ == "__main__":
    run()
    show()

