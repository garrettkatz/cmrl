import numpy as np
import os, sys
import pybullet as pb
from gym.spaces import Space

sys.path.append(os.path.join(os.environ["HOME"], "poppy-simulations", "ergo", "pybullet", "envs"))
from ergo import PoppyErgoEnv

ERGO_JOINTS = 36

class Wrapper:
    def __init__(self, timestep, control_period):
        self.timestep = timestep
        self.control_period = control_period
        self.env = None
        self.show = False
        self.action_space = Space(shape=(ERGO_JOINTS,))
        self.observation_space = Space(shape=(ERGO_JOINTS,))

    def render(self, mode):
        self.show = True if mode == "human" else False

    def close(self):
        if self.env is not None: self.env.close()

    def obs(self):
        return self.env.get_position()

    def reset(self):

        self.close()
        self.env = PoppyErgoEnv(
            control_mode=pb.POSITION_CONTROL,
            timestep=self.timestep,
            control_period=self.control_period,
            show=self.show,
            step_hook=None,
            use_fixed_base=False,
            use_self_collision=False,
        )
        return self.obs()

    def reward(self):
        pos, orn, vel, ang = self.env.get_base()
        return -pos[1] # reward for distance along -y direction

    def step(self, action):
        self.env.step(action)
        pos, orn, vel, ang = self.env.get_base()
        done = (pos[2] < .3) # rough threshold for falling
        return self.obs(), self.reward(), done, None

def env_maker(timestep, control_period):
    def make_env():    
        return Wrapper(timestep, control_period)
    return make_env

def initial_waypoints():

    # initial hand-picked walk waypoints from poppy-muffin/pybullet/tasks/check/strides.py
    env = PoppyErgoEnv(show=False)

    stand = env.get_position()
    stand_dict = env.angle_dict(stand)
    
    lift_dict = dict(stand_dict)
    lift_dict.update({'l_ankle_y': 30, 'l_knee_y': 80, 'l_hip_y': -50})
    lift = env.angle_array(lift_dict)
    
    step_dict = dict(stand_dict)
    step_dict.update({
        'l_ankle_y': 20, 'l_knee_y': 0, 'l_hip_y': -20,
        'r_ankle_y': -20, 'r_knee_y': 0, 'r_hip_y': 20})
    step = env.angle_array(step_dict)
    
    push_dict = dict(stand_dict)
    push_dict.update({
        'l_ankle_y': 0, 'l_knee_y': 0, 'l_hip_y': 0,
        'r_ankle_y': 50, 'r_knee_y': 90, 'r_hip_y': -40})
    push = env.angle_array(push_dict)
    
    settle_dict = dict(stand_dict)
    settle_dict.update({
        'l_ankle_y': 10, 'l_knee_y': 0, 'l_hip_y': -10,
        'r_ankle_y': -20, 'r_knee_y': 40, 'r_hip_y': -20})
    settle = env.angle_array(settle_dict)

    waypoints = [stand, lift, push, settle]
    for w in range(len(waypoints)):
        waypoints.append(env.mirror_position(waypoints[w]))

    env.close()

    return np.stack(waypoints)

def initial_policy():
    waypoints = initial_waypoints()
    waypoints = np.concatenate((waypoints, waypoints[:1]), axis=0) # return to stand

    # linear interpolation
    numpts = 6 # 24 env steps per second, 4 waypoints (one footstep) per second
    terp = np.linspace(0, 1, numpts)[:-1, np.newaxis]
    positions = []
    for w in range(len(waypoints)-1):
        positions.append(terp * waypoints[w] + (1 - terp) * waypoints[w+1])
    positions.append(waypoints[-1:])
    positions = np.concatenate(positions, axis=0)

    # sinusoidal time encoding
    t = np.linspace(0, 2*np.pi, len(positions))
    t_encoding = np.stack([np.sin(t), np.cos(t)]).T

    # policy examples
    obs = np.concatenate((positions[:-1], t_encoding[:-1]), axis=1)
    act = positions[1:]

    print(obs.shape, act.shape)

    # linear fit
    M = np.linalg.lstsq(obs, act, rcond=None)[0].T

    print(np.mean((M @ obs.T - act.T)**2))

    return M

if __name__ == "__main__":

    # # initial hand-picked walk waypoints from poppy-muffin/pybullet/tasks/check/strides.py
    # waypoints = initial_waypoints()
    # env = PoppyErgoEnv(show=True)
    # a = 0
    # while True:
    #     env.set_position(waypoints[a])
    #     waypoints[a] = env.mirror_position(waypoints[a])
    #     a = (a+1) % len(waypoints)
    #     input("...")

    # initial hand-picked walk waypoints from poppy-muffin/pybullet/tasks/check/strides.py
    M = initial_policy()

