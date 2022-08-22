import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from pointbotenv import PointBotEnv

if __name__ == "__main__":

    num_domains = 1
    batch_size = 1
    num_updates = 10000
    verbose_period = 10
    show_period = 500
    num_steps = 150
    arena_size = 5
    alpha = 0.5 # mix actions with this much randomness during exploration
    walk_stdev = 0.1 # stdev for random walk

    # Set up spring parameters for bot motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = .9 * critical
    # b = np.random.uniform(.25, .9)*critical # random underdamping

    width = height = arena_size
    gravity = 10
    mass = m
    spring_constant = k
    damping = b
    dt = 1/24

    env = PointBotEnv(width, height, gravity, mass, spring_constant, damping, dt, batch_size)

    obs_size, act_size = 4, 2

    # Initialize policy net
    # net = tr.nn.Linear(obs_size, act_size)
    hid_size = 16
    net = tr.nn.Sequential(
        tr.nn.Linear(obs_size, hid_size),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(hid_size, hid_size),
        tr.nn.LeakyReLU(),
        # tr.nn.Linear(hid_size, hid_size),
        # tr.nn.LeakyReLU(),
        # tr.nn.Linear(hid_size, hid_size),
        # tr.nn.LeakyReLU(),
        tr.nn.Linear(hid_size, act_size),
    )
    # optimizer = tr.optim.SGD(net.parameters(), lr=0.01)
    optimizer = tr.optim.Adam(net.parameters(), lr=0.01)

    # Initialize episodic memory buffer
    memory = {
        "observation": np.empty((num_steps+1, num_domains, obs_size)),
        "action": np.empty((num_steps, num_domains, act_size)),
        "reward": np.empty((num_steps, num_domains)),
    }
    # print(memory["action"][:,0,:])
    # pt.plot(memory["action"][:,0,0], memory["action"][:,0,1], 'k.-')
    # pt.show()

    memory["observation"][0] = env.reset()
    for t in range(num_steps):
        with tr.no_grad(): memory["action"][t] = env.bound(net(tr.tensor(memory["observation"][t], dtype=tr.float)).numpy())
        memory["observation"][t+1], memory["reward"][t], _, _ = env.step(memory["action"][t])

    reward_curve = {
        "memory": np.empty((num_updates,)),
        "student": np.empty((num_updates,))
    }
    better_curve = {
        "branch": np.empty((num_updates,), dtype=int),
        "student": np.empty((num_updates,), dtype=int)
    }
    error_curve = np.empty((num_updates,))

    for update in range(num_updates):

        ## update memory batch with random actions
        batch_idx = np.random.choice(num_domains, size=batch_size, replace=False)
        branch = {key: arr[:,batch_idx].copy() for (key, arr) in memory.items()}
        branch["action"] *= (1-alpha)
        # branch["action"] += alpha * np.random.rand(num_steps, batch_size, act_size) * env.shape
        branch["action"] += alpha * ((np.random.randn(num_steps, batch_size, act_size) * env.shape * walk_stdev).cumsum(axis=0) + env.shape * .5)
        branch["action"] = env.bound(branch["action"])
        branch["observation"][0] = env.reset()
        for t in range(num_steps):
            branch["observation"][t+1], branch["reward"][t], _, _ = env.step(branch["action"][t])
        branch_better = branch["reward"].sum(axis=0) > memory["reward"][:,batch_idx].sum(axis=0)
        for key in memory.keys():
            memory[key][:, batch_idx[branch_better]] = branch[key][:, branch_better]
        better_curve["branch"][update] = branch_better.sum()
        reward_curve["memory"][update] = memory["reward"][:, batch_idx].sum(axis=0).mean()

        # if update % show_period == 0:
        #     pt.ioff()
        #     pt.figure()
        #     env.plot(pt.gca())
        #     pt.plot(branch["action"][:,0,0], branch["action"][:,0,1], 'mo-')
        #     pt.show()
        #     pt.close()

        ## update memory batch with student actions
        batch_idx = np.random.choice(num_domains, size=batch_size, replace=False)
        student = {key: arr[:,batch_idx].copy() for (key, arr) in memory.items()}
        student["observation"][0] = env.reset()
        for t in range(num_steps):
            with tr.no_grad():
                student["action"][t] = env.bound(net(tr.tensor(student["observation"][t], dtype=tr.float)).numpy())
            student["observation"][t+1], student["reward"][t], _, _ = env.step(student["action"][t])
        student_better = student["reward"].sum(axis=0) > memory["reward"][:,batch_idx].sum(axis=0)
        for key in memory.keys():
            memory[key][:, batch_idx[student_better]] = student[key][:, student_better]
        better_curve["student"][update] = student_better.sum()
        reward_curve["student"][update] = student["reward"].sum(axis=0).mean()
        
        ## update policy
        batch_idx = np.random.choice(num_domains, size=batch_size, replace=False)
        observation = memory["observation"][:num_steps, batch_idx, :]
        action = memory["action"][:, batch_idx, :]
        prediction = net(tr.tensor(observation, dtype=tr.float))
        errors = tr.tensor(action, dtype=tr.float) - prediction
        loss = tr.mean(errors**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error_curve[update] = errors.abs().max().item()

        # if update % show_period == 0:
        #     pt.figure()
        #     pt.ion()
        #     with tr.no_grad():
        #         policy = lambda obs: (env.bound(net(tr.tensor(obs, dtype=tr.float)).numpy()), None)
        #         env.animate(policy, num_steps, ax=pt.gca())
        #     pt.close()

        if update % verbose_period == 0:
            print(f"{update}/{num_updates}: " + \
                f"memory reward={reward_curve['memory'][update]}, " + \
                f"student reward={reward_curve['student'][update]}, " + \
                f"error={error_curve[update]}, " + \
                f"branch better={better_curve['branch'][update]}, " + \
                f"student better={better_curve['student'][update]} "
            )

    pt.ioff()

    pt.subplot(3,1,1)
    for key, arr in reward_curve.items(): pt.plot(arr, label=key)
    pt.ylabel("reward")
    pt.legend()

    pt.subplot(3,1,2)
    for key, arr in better_curve.items(): pt.plot(arr, label=key)
    pt.ylabel("better")
    pt.legend()

    pt.subplot(3,1,3)
    pt.plot(error_curve, label=key)
    pt.ylabel("error")
    pt.legend()

    pt.show()
    pt.close()

    pt.figure()
    pt.ion()
    with tr.no_grad():
        policy = lambda obs: (env.bound(net(tr.tensor(obs, dtype=tr.float)).numpy()), None)
        env.animate(policy, num_steps, ax=pt.gca())

