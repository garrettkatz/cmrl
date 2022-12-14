import pybullet as pb
import numpy as np
import pybullet_envs
import gym

# env = gym.make('Walker2DBulletEnv-v0')
# env = gym.make('HumanoidBulletEnv-v0')
env = gym.make('AntBulletEnv-v0')

# env.render(mode="human")

T = 10

# A = np.random.uniform(-1, 1, size=(2*T, 6,)) # walker2d
# A = np.random.uniform(-1, 1, size=(2*T, 17,)) # humanoid
A = np.random.uniform(-1, 1, size=(2*T, 8,)) # ant

state = env.reset()

print('start')
for t in range(T):
    obsT, rewT, _, _ = env.step(A[t])
rewardsT = list(env.rewards)

sid = pb.saveState()

for r in range(100):

    pb.restoreState(sid)

    print("T", obsT, rewT, rewardsT)
    for t in range(T, 2*T):
        obs2T, rew2T, _, _ = env.step(A[t])
    print("2T", obs2T, rew2T, env.rewards)

    print(r)
    input('.')

env.close()

