import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp
import gym

def blind_search(env, num_steps, max_rollouts, walk_stdev = None):

    for r in range(max_rollouts):

        if walk_stdev is None:
            actions = np.random.uniform(-1, 1, (num_steps, 1)).astype(np.float32)
        else:
            actions = np.random.randn(num_steps, 1).astype(np.float32) * walk_stdev
            actions[0,0] = np.random.uniform(-1, 1).astype(np.float32)
            actions = actions.cumsum(axis=1)
            actions = np.clip(actions, -1, 1)

        state, _ = env.reset()
        for t in range(num_steps):
            state, reward, terminated, _, _ = env.step(actions[t])
            if success:
                return actions[:t+1], True

    return None, False

def run(env, actions):

    num_rollouts, num_steps = actions.shape[:2]

    states = np.empty((num_rollouts, num_steps+1, 2), dtype=np.float32)
    rewards = np.empty((num_rollouts, num_steps)).astype(np.float32)

    success = None
    for r in range(num_rollouts):
        states[r, 0] = env.reset()
        for t in range(num_steps):
            states[r, t+1], rewards[r, t], terminated, _ = env.step(actions[r, t])
            if terminated:
                success = (r, t)

    return states, rewards, success

def do():

    num_rollouts = 8 * 4 # comparable computational expense to beam * branching
    num_steps = 999
    walk_stdev = 0.1

    expert_actions = np.ones((1, num_steps, 1), dtype=np.float32)
    expert_actions[:,:20,:] = -1

    uniform_actions = np.random.uniform(-1, 1, (num_rollouts, num_steps, 1)).astype(np.float32)

    walk_actions = np.random.randn(num_rollouts, num_steps, 1).astype(np.float32) * walk_stdev
    walk_actions[:,:1,:] = np.random.uniform(-1, 1, (num_rollouts, 1, 1)).astype(np.float32)
    walk_actions = walk_actions.cumsum(axis=1)
    walk_actions = np.clip(walk_actions, -1, 1)

    env = gym.make("MountainCarContinuous-v0")
    expert_states, expert_rewards, expert_success = run(env, expert_actions)
    uniform_states, uniform_rewards, uniform_success = run(env, uniform_actions)
    walk_states, walk_rewards, walk_success = run(env, walk_actions)
    env.close()

    print(f"expert success at {expert_success}")
    print(f"walk success at {walk_success}")
    print(f"total steps = {num_steps}*{num_rollouts} = {num_steps*num_rollouts}")

    with open("mcr.pkl","wb") as f: pk.dump(uniform_states, f)

def show():

    with open("mcr.pkl","rb") as f: uniform_states = pk.load(f)
    num_rollouts, num_steps = uniform_states.shape[:2]

    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    # goal rectangle
    pt.gca().add_patch(mp.Rectangle((.45, 0), .15, .07, linewidth=0, facecolor='gray'))
    for t in range(num_steps):
        pt.scatter(uniform_states[:,t,0], uniform_states[:,t,1], color=colors[t], marker='.')

    # for r in range(num_rollouts):
    #     pt.plot(uniform_states[r, :, 0].flatten(), uniform_states[r, :, 1].flatten(), 'b.-')
    #     pt.plot(walk_states[r, :, 0].flatten(), walk_states[r, :, 1].flatten(), 'r.-')
    # pt.plot(expert_states[:, :, 0].flatten(), expert_states[:, :, 1].flatten(), 'g.-')

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")
    pt.show()

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__":
    do()
    show()

