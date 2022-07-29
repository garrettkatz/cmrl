import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from catmouse import CatMouseEnv

if __name__ == "__main__":

    # training hyperparams
    num_updates = 5000
    verbose_period = 1
    show_period = 100
    num_steps = 100
    horizon = 10
    discount = 0.001 ** (1 / horizon) # discount**horizon = 0.001
    
    # Set up spring parameters for mouse motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = .75*critical # underdamping

    # Initialize environment
    env = CatMouseEnv(
        width = 5,
        height = 5,
        cat_speed = 2,
        spring = k/m,
        damping = b/m,
        dt = 1/24,
        # random_cat = False,
        random_cat = True,
        # batch_size=256)
        batch_size=512)

    # Initial conditions
    # cp = 0.25*np.ones((env.batch_size, 2))*env.shape
    # cv = np.tile(np.array([1., 0.]), (env.batch_size,1))*env.cat_speed
    # mp = 0.5*np.ones((env.batch_size, 2))*env.shape
    cp = cv = mp = None

    # Policy
    net = tr.nn.Sequential(
        # tr.nn.Linear(8, 2),

        tr.nn.Linear(8,32),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(32,2),

        tr.nn.Sigmoid(),
    )
    def policy(observation):
        dist_params = net(tr.tensor(observation, dtype=tr.float))
        dist = tr.distributions.Normal(dist_params, .1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action.numpy() * env.shape, log_prob

    optimizer = tr.optim.SGD(net.parameters(), lr=0.00001)

    net_rewards = np.empty((num_updates, env.batch_size))
    grad_norms = np.empty(num_updates)

    pt.figure(figsize=(8,4), constrained_layout=True)

    for update in range(num_updates):

        if update % show_period == 0:

            pt.ioff()
            pt.subplot(1,2,1)
            pt.cla()
            mean = net_rewards[:update].mean(axis=1)
            if env.batch_size > 3:
                stdev = net_rewards[:update].std(axis=1)
                pt.fill_between(np.arange(update), mean-stdev, mean+stdev, color='b', alpha=0.5)
            pt.plot(mean)

            pt.ion()
            pt.subplot(1,2,2)
            pt.cla()
            observation = env.reset(
                cp = cp,
                cv = cv,
                mp = mp,
            )
            with tr.no_grad():
                env.animate(policy, num_steps=20, ax=pt.gca(), reset=False)


        observation = env.reset(
            cp = cp,
            cv = cv,
            mp = mp,
        )

        rewards, log_probs = {}, {}
        for t in range(num_steps):
            action, log_probs[t] = policy(observation)
            observation, rewards[t], _, _ = env.step(action)
        net_rewards[update] = sum(rewards.values())

        optimizer.zero_grad()

        # # full-episode reward
        # (-tr.tensor(net_rewards[update]) * sum(log_probs.values())).mean().backward()

        # reward to go
        for t in reversed(range(num_steps-1)):
            rewards[t] += rewards[t+1]
            (-tr.tensor(rewards[t]) * log_probs[t]).mean().backward()

        # # discounted reward to go, doesn't work well (this implementation is not mathematically well-founded)
        # for t in range(num_steps - horizon):
        #     for th in range(1, horizon):
        #         rewards[t] += rewards[t+th][0] * discount**th
        #     (-tr.tensor(rewards[t]) * log_probs[t]).mean().backward()

        grad_norms[update] = sum((p.grad**2).sum() for p in net.parameters())**0.5

        optimizer.step()

        if update % verbose_period == 0:
            print(f"{update}/{num_updates}: R~{net_rewards[update].mean()}, |g|~{grad_norms[update]}")


    pt.close()
    pt.ioff()
    mean = net_rewards[:update].mean(axis=1)
    if env.batch_size > 3:
        stdev = net_rewards[:update].std(axis=1)
        pt.fill_between(np.arange(update), mean-stdev, mean+stdev, color='b', alpha=0.5)
    pt.plot(mean)
    pt.show()

    pt.ion()
    observation = env.reset(
        cp = cp,
        cv = cv,
        mp = mp,
    )
    with tr.no_grad():
        env.animate(policy, num_steps, ax=pt.gca(), reset=False)
