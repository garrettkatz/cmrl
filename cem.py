import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from catmouse import CatMouseEnv

tr.set_grad_enabled(False)

size = 5

class Net(tr.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = tr.nn.Sequential(
            tr.nn.Linear(8,2),
            tr.nn.Sigmoid(),
        )
    def forward(self, x):
        return self.seq(x)*size

if __name__ == "__main__":

    # training hyperparams
    num_updates = 5000
    verbose_period = 1
    show_period = 1
    num_steps = 100
    population_size = 30
    num_elites = 5
    
    # Set up spring parameters for mouse motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = .75*critical # underdamping

    # Initialize environment
    env = CatMouseEnv(
        width = size,
        height = size,
        cat_speed = 2,
        spring = k/m,
        damping = b/m,
        dt = 1/24,
        random_cat = False,
        batch_size=1)

    # Initial conditions
    cp = 0.25*np.ones((env.batch_size, 2))*env.shape
    cv = np.tile(np.array([1., 0.]), (env.batch_size,1))*env.cat_speed
    mp = 0.5*np.ones((env.batch_size, 2))*env.shape

    # parameter distribution
    mean_net = Net()
    means = mean_net.state_dict()
    for k in means.keys(): means[k].data *= 0
    stdevs = {k: 0.01*tr.ones(means[k].shape) for k in means.keys()}

    net_rewards = np.empty((num_updates, population_size, env.batch_size))
    
    for update in range(num_updates):

        if update % show_period == 0:

            pt.ioff()
            pt.subplot(1,2,1)
            pt.cla()
            mean = net_rewards[:update].mean(axis=(1,2))
            if env.batch_size > 3:
                stdev = net_rewards[:update].std(axis=(1,2))
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
            print(list(mean_net.parameters()))
            print(means)
            policy = lambda obs: (mean_net(tr.tensor(obs, dtype=tr.float)).numpy(), None)
            env.animate(policy, num_steps=20, ax=pt.gca(), reset=False)

        # Sample and evaluate population
        params = {}
        for indiv in range(population_size):

            net = Net()
            params[indiv] = net.state_dict()
            for k in means.keys():
                params[indiv][k].data[:] = means[k] + stdevs[k] * tr.randn(means[k].shape)

            observation = env.reset(
                cp = cp,
                cv = cv,
                mp = mp,
            )

            rewards = {}
            for t in range(num_steps):
                action = net(tr.tensor(observation, dtype=tr.float))
                observation, rewards[t], _, _ = env.step(action.numpy())
            
            net_rewards[update, indiv] = sum(rewards.values())

        # identify elites
        fitness = net_rewards[update].mean(axis=1)
        elites = fitness.argsort()[:-num_elites]

        max_stdev = 0
        for k in means.keys():
            means[k].data[:] = sum(params[i][k] for i in elites) / num_elites
            stdevs[k].data[:] = (sum((params[i][k] - means[k])**2 for i in elites) / num_elites)**0.5
            max_stdev = max(max_stdev, stdevs[k].max())

        print(means)

        if update % verbose_period == 0:
            print(f"{update}/{num_updates}: R~{net_rewards[update].mean()}, s<={max_stdev}")

