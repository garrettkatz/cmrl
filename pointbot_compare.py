import matplotlib.pyplot as pt
from pointbotenv import PointBotEnv
from blind_rollouts import blind_rollouts

if __name__ == "__main__":

    do_train = True
    do_test = True
    do_show = True

    ntrain = 3
    ntest = 1

    epsilon = 1.
    num_episodes = 500
    num_steps = 150
    batch_size = 32
    report_period = 10

    train_env = PointBotEnv.sample_domains(ntrain)
    test_env = PointBotEnv.sample_domains(ntest)

    policy, reward_curve = blind_rollouts(epsilon, num_episodes, num_steps, train_env, batch_size, report_period)

    policy.reset()
    _, _, test_reward = test_env.run_episode(policy, num_steps, reset_batch_size=1)

    pt.ioff()

    pt.plot(reward_curve)
    for d in range(ntest):
        pt.plot([0, len(reward_curve)], [test_reward[:,d,:].sum()]*2, '--')
    pt.ylabel("reward")

    pt.show()
    pt.close()

    pt.figure()
    pt.ion()

    ax = pt.subplot(1,2,1)
    policy.reset()
    train_env.animate(policy, num_steps, ax, reset_batch_size=1)

    ax = pt.subplot(1,2,2)
    policy.reset()
    test_env.animate(policy, num_steps, ax, reset_batch_size=1)

