import itertools as it
from scipy.spatial.distance import pdist, squareform
from policies import FixedPolicy

def main():

    from pointbotenv import PointBotEnv

    num_updates = 2
    report_period = 1
    num_steps = 150
    stdev = 0.1

    num_domains = 1
    branching = 4
    beam_size = 8

    env = PointBotEnv.sample_domains(num_domains)

    # initialize beam
    beam_states = np.empty((num_steps, env.num_domains, beam_size, env.obs_size))
    beam_actions = np.empty((num_steps, env.num_domains, beam_size, env.act_size))
    beam_rewards = np.empty((num_steps, env.num_domains, beam_size))

    for step, beam in it.product(range(num_steps), range(beam_size)):

        # initialize random walk rollout actions
        deltas = np.random.randn(num_steps - step, env.num_domains, beam_size * branching, env.act_size) * stdev
        deltas[0] = np.random.rand(env.num_domains, beam_size * branching, env.act_size)
        actions = env.bound(deltas.cumsum(axis=0))

        # expand current states for branching
        current = beam_states[step, :, :, np.newaxis, :]
        current = np.broadcast_to(current, (env.num_domains, beam_size, branching, env.obs_size))
        current = current.reshape(env.num_domains, beam_size * branching, env.obs_size)

        # get rollouts
        env.reset(state=current)
        states, actions, rewards = env.run_episode(FixedPolicy(actions), num_steps - step)

        # merge states at current step into beam_size
        for domain in range(num_domains):
            dists = pdist(states[0,domain])




if __name__ == "__main__": main()


