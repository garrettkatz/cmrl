import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from catmouse import CatMouseEnv

if __name__ == "__main__":

    num_explorations = 5000
    num_updates = 20
    verbose_period = 50
    show_period = 1
    num_steps = 100
    batch_size = 128
    arena_size = 5

    # Set up spring parameters for mouse motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = .75*critical # underdamping

    # Initialize environment
    env = CatMouseEnv(
        width = arena_size,
        height = arena_size,
        cat_speed = 2,
        spring = k/m,
        damping = b/m,
        dt = 1/24,
        random_cat = False,
        batch_size=batch_size)

    # Initial conditions
    cp = 0.25*np.ones((env.batch_size, 2))*env.shape
    cv = np.tile(np.array([1., 0.]), (env.batch_size,1))*env.cat_speed
    mp = 0.5*np.ones((env.batch_size, 2))*env.shape

    alpha = 1.0
    net_rewards = np.empty((num_explorations, batch_size))
    final_errors = np.empty(num_explorations)
    net = tr.nn.Linear(8, 2)

    for exploration in range(num_explorations):

        observation = np.empty((2, num_steps+1, batch_size, 8))
        action = np.empty((2, num_steps, batch_size, 2))
        reward = np.zeros((2, num_steps, batch_size))

        branch_point = np.random.randint(num_steps, size=batch_size)
        branch_action = np.random.rand(batch_size, 2) * env.shape

        for br in (0, 1):
            observation[br, 0] = env.reset(cp = cp, cv = cv, mp = mp)

            for t in range(num_steps):
                with tr.no_grad():
                    action[br, t] = net(tr.tensor(observation[br, t], dtype=tr.float)).numpy()
                if br > 0:
                    action[br, t, t == branch_point] *= (1-alpha)
                    action[br, t, t == branch_point] += alpha * branch_action[t == branch_point]
                observation[br, t+1], reward[br, t], _, _ = env.step(action[br, t])
    
        net_reward = reward.sum(axis=1, keepdims=True)[:,:,:,np.newaxis]
        net_rewards[exploration] = net_reward[0,0,:,0]

        branch_worse = net_reward[0] > net_reward[1]
        if branch_worse.all(): continue

        observation = np.where(branch_worse, observation[0], observation[1])
        action = np.where(branch_worse, action[0], action[1])

        observation = observation[:num_steps].reshape(num_steps*batch_size, 8)
        action = action.reshape(num_steps*batch_size, 2)

        optimizer = tr.optim.SGD(net.parameters(), lr=0.01)
        errors = np.empty(num_updates)
        for update in range(num_updates):
            pred = net(tr.tensor(observation, dtype=tr.float))
            error = tr.mean((pred - tr.tensor(action, dtype=tr.float))**2)
            errors[update] = error.item()
            # print(f"  {update}/{num_updates}: err = {error:f}")
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
        final_errors[exploration] = errors[-1]

        if exploration % verbose_period == 0:
            print(f"{exploration}/{num_explorations}: " + \
                f"R~{net_rewards[exploration].mean()}, " + \
                f"{branch_worse.sum()}/{batch_size} worse branches, " + \
                f"error: {errors[0]} -> {errors[-1]}")

    pt.subplot(2,1,1)
    mean = net_rewards.mean(axis=1)
    if env.batch_size > 3:
        stdev = net_rewards.std(axis=1)
        pt.fill_between(np.arange(num_explorations), mean-stdev, mean+stdev, color='b', alpha=0.5)
    pt.plot(mean)
    pt.subplot(2,1,2)
    pt.plot(final_errors)
    pt.show()

    pt.ion()
    # pt.subplot(1,2,2)
    env.reset(cp = cp, cv = cv, mp = mp)
    policy = lambda obs: (net(tr.tensor(obs, dtype=tr.float)).numpy(), None)
    with tr.no_grad():
        env.animate(policy, num_steps, ax=pt.gca(), reset=False)

