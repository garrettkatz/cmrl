import pybullet as pb
import numpy as np
import pybullet_envs
import gym

env = gym.make('Walker2DBulletEnv-v0') 
env.render(mode="human")

state = env.reset()
print(env.stateId)

# print(env.action_space)
# print(dir(env))

for t in range(1):
    env.step(np.random.uniform(-1, 1, size=(6,)))

sid = pb.saveState()
print(sid)

input('.')

for t in range(500):
    env.step(np.random.uniform(-1, 1, size=(6,)))

input('.')

env.state_id = sid
env.reset()

for t in range(1):
    env.step(np.random.uniform(-1, 1, size=(6,)))

input('.')

for t in range(500):
    env.step(np.random.uniform(-1, 1, size=(6,)))

env.close()

