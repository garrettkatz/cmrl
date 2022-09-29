from time import perf_counter
import numpy as np
import matplotlib.pyplot as pt
import gym

def blind_search(env, initial_state, num_steps, max_rollouts, walk_stdev = None):

    for r in range(max_rollouts):

        if walk_stdev is None:
            actions = np.random.uniform(-1, 1, (num_steps, 1)).astype(np.float32)
        else:
            actions = np.random.randn(num_steps, 1).astype(np.float32) * walk_stdev
            actions[0,0] = np.random.uniform(-1, 1)
            actions = actions.cumsum(axis=0)
            actions = np.clip(actions, -1, 1)

        env.state = initial_state
        for t in range(num_steps):
            # state, reward, terminated, _, _ = env.step(actions[t]) # newer gym returns info
            state, reward, terminated, _ = env.step(actions[t])
            if terminated:
                return actions[:t+1], r+1, True

    return None, max_rollouts, False

def main():

    num_reps = 30
    max_rollouts = 32 * 8 # comparable computational expense to beam * branching
    num_steps = 999
    walk_stdev = 0.1

    env = gym.make("MountainCarContinuous-v0").unwrapped

    success = np.empty((num_reps, 2), dtype=bool)
    success_indicator = np.zeros((num_reps, 2, max_rollouts * num_steps))
    run_times = np.empty((num_reps, 2))
    for rep in range(num_reps):
        print(f"rep {rep} of {num_reps}")
        # init_state, _ = env.reset() # newer gym returns info
        init_state = env.reset()
        for dw, do_walk in enumerate((False, True)):
            stdev = walk_stdev if do_walk else None
            start = perf_counter()
            actions, num_rollouts, success[rep, dw] = blind_search(env, init_state, num_steps, max_rollouts, stdev)
            run_times[rep, dw] = perf_counter() - start
            if success[rep, dw]:
                step_counts = num_rollouts * num_steps + len(actions)
                success_indicator[rep, dw, step_counts:] = 1

    env.close()

    uni_avg, rw_avg = success_indicator[:,0,:].mean(axis=0), success_indicator[:,1,:].mean(axis=0)
    uni_std, rw_std = success_indicator[:,0,:].std(axis=0), success_indicator[:,1,:].std(axis=0)
    xpts = np.arange(len(uni_avg))

    pt.subplot(2,1,1)
    ax = pt.gca()

    ax.fill_between(xpts, uni_avg-uni_std, uni_avg+uni_std, color='r', alpha=0.2)
    ax.fill_between(xpts, rw_avg-rw_std, rw_avg+rw_std, color='b', alpha=0.2)

    ax.plot(xpts, uni_avg, color='r', linestyle='-', label="Uniform")
    ax.plot(xpts, rw_avg, color='b', linestyle='-', label="Random walk")
    ax.legend(loc='lower right')
    ax.set_ylabel("Pr(success)")
    ax.set_xlabel("Num env steps")

    pt.subplot(2,1,2)
    uni_xpts = np.concatenate(([0.], np.sort(run_times[success[:,0],0])))
    rw_xpts = np.concatenate(([0.], np.sort(run_times[success[:,1],1])))
    uni = np.triu(np.ones((num_reps, len(uni_xpts))), k=1)
    rw = np.triu(np.ones((num_reps, len(rw_xpts))), k=1)
    if rw_xpts[-1] > uni_xpts[-1]:
        uni_xpts = np.append(uni_xpts, [rw_xpts[-1]])
        uni = np.concatenate((uni, uni[:,-1:]), axis=1)
    elif rw_xpts[-1] < uni_xpts[-1]:
        rw_xpts = np.append(rw_xpts, [uni_xpts[-1]])
        rw = np.concatenate((rw, rw[:,-1:]), axis=1)


    uni_avg, rw_avg = uni.mean(axis=0), rw.mean(axis=0)
    uni_std, rw_std = uni.std(axis=0), rw.std(axis=0)

    ax = pt.gca()

    ax.fill_between(uni_xpts, uni_avg-uni_std, uni_avg+uni_std, color='r', alpha=0.2)
    ax.fill_between(rw_xpts, rw_avg-rw_std, rw_avg+rw_std, color='b', alpha=0.2)

    ax.plot(uni_xpts, uni_avg, color='r', linestyle='-', marker='.', label="Uniform")
    ax.plot(rw_xpts, rw_avg, color='b', linestyle='-', marker='.', label="Random walk")
    ax.legend(loc='lower right')
    ax.set_ylabel("Pr(success)")
    ax.set_xlabel("Wall Time (s)")
    
    pt.tight_layout()
    pt.show()


if __name__ == "__main__": main()

