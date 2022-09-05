from fixed_policy import FixedPolicy

def main():

    from pointbotenv import PointBotEnv

    num_updates = 2
    report_period = 1
    num_steps = 150
    learning_rate = 0.01

    num_domains = 1
    batch_size = 64
    beam_size = 3

    env = PointBotEnv.sample_domains(num_domains)

    for update in range(num_updates):

        # initialize beam memory
        beam_actions = np.empty((num_steps, env.num_domains, beam_size, env.act_size))

        for depth in range(num_steps):

            T = num_steps - depth

            for node in range(beam_size):

                # random search actions
                actions = np.random.rand(env.num_domains, batch_size, env.act_size)
                states = np.empty((T+1, env.num_domains, batch_size, env.obs_size))
                states[0] = np.broadcast_to(beam_states[depth, :, node, :], states[0].shape)
                rewards = np.random.rand(T, env.num_domains, batch_size)

                # rollout with net after random actions in first step
                env.reset(states[0])
                states[1] = env.step(actions[0])
                with tr.no_grad():
                    states[1:], actions[1:], rewards[1:] = env.run_episode(net_policy, T-1)

                # merge new states one at a time to maintain beam size
                for rollout in range(batch_size):

                    # find closest state already in beam
                    dists = np.linalg.norm(states[1,:,rollout,:] - beam_states[depth+1], axis=-1)
                    closest = dists.argmin()

                    # recombine rollouts
                
                # train

    net_policy.reset(explore=False)
    with tr.no_grad():
        episodes = env.run_episode(net_policy, num_steps, reset_batch_size=1)
    reward = env.animate(episodes)
    print(reward)
    pt.show()


if __name__ == "__main__": main()


