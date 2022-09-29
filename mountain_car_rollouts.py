import time
import numpy as np
import matplotlib.pyplot as pt
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
        states[r, 0], _ = env.reset()
        for t in range(num_steps):
            states[r, t+1], rewards[r, t], terminated, _, _ = env.step(actions[r, t])
            if terminated:
                success = (r, t)

    return states, rewards, success


def main():

    num_rollouts = 8 * 4 # comparable computational expense to beam * branching
    num_steps = 200
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

    for r in range(num_rollouts):
        pt.plot(uniform_states[r, :, 0].flatten(), uniform_states[r, :, 1].flatten(), 'b.-')
        pt.plot(walk_states[r, :, 0].flatten(), walk_states[r, :, 1].flatten(), 'r.-')
    pt.plot(expert_states[:, :, 0].flatten(), expert_states[:, :, 1].flatten(), 'g.-')

    pt.xlim([-1.2, 0.6])
    pt.ylim([-0.07, 0.07])
    pt.xlabel("Position")
    pt.ylabel("Velocity")
    pt.show()

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # state, _ = env.reset()

if __name__ == "__main__": main()

