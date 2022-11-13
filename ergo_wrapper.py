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

        self.goal_speed = .1 # units/second

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
            # show=True,
            step_hook=None,
            use_fixed_base=False,
            use_self_collision=False,
        )

        # initial position and orientation for assessing reward
        self.start_pos, self.start_orn, _, _ = self.env.get_base()
        self.num_steps = 0

        return self.obs()

    def reward(self):
        pos, orn, _, _ = self.env.get_base()

        # Target position
        dt = self.num_steps * self.control_period * self.timestep
        ty = - self.goal_speed * dt # move along -y direction
        tpos = (self.start_pos[0], ty, self.start_pos[2])

        # position error
        dpos = ((pos[0]-tpos[0])**2 + (pos[1]-tpos[1])**2 + (pos[2]-tpos[2])**2)**.5

        # orientation error
        quat = pb.getDifferenceQuaternion(orn, self.start_orn)
        _, angle = pb.getAxisAngleFromQuaternion(quat)
        dang = abs(angle)

        # penalize distance from goal position/orientation
        return -(dpos + dang)

    def step(self, action):
        self.env.step(action)
        self.num_steps += 1

        pos, orn, vel, ang = self.env.get_base()
        # done = abs(pos[2] - .43) > .1 # rough threshold for target waist height
        done = False # let negative rewards accumulate
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

    waypoints = [stand, lift, step, push, step, settle]
    for w in range(4,6):
        waypoints[w] = env.mirror_position(waypoints[w])

    env.close()

    return np.stack(waypoints)

def initial_policy():
    waypoints = initial_waypoints()
    waypoints = np.concatenate((waypoints, waypoints[:1]), axis=0) # return to stand

    # linear interpolation
    # numpts = 4 # 24 env steps per second, 1 second for cycle, 6 waypoints per cycle
    numpts = 32 # slower steps, more examples
    terp = np.linspace(0, 1, numpts+1)[:-1, np.newaxis] # traj towards next waypoint
    positions = []
    actions = []
    for w in range(len(waypoints)-1):
        positions.append((1 - terp) * waypoints[w] + terp * waypoints[w+1])
        actions.append(np.tile(waypoints[w+1], (numpts,1)))
    positions = np.concatenate(positions, axis=0)
    actions = np.concatenate(actions, axis=0)

    # sinusoidal time encoding
    t = np.linspace(0, 2*np.pi, len(positions))
    t_encoding = np.stack([np.sin(t), np.cos(t)]).T

    # policy examples
    # obs = np.concatenate((positions, t_encoding[:-1]), axis=1) # use time
    obs = positions # ignore time
    act = actions

    # linear fit
    M = np.linalg.lstsq(obs, act, rcond=None)[0].T

    return M

if __name__ == "__main__":

    # # initial hand-picked walk waypoints from poppy-muffin/pybullet/tasks/check/strides.py
    # waypoints = initial_waypoints()
    # env = PoppyErgoEnv(show=True)
    # a = 0
    # while True:
    #     env.set_position(waypoints[a])
    #     a = (a+1) % len(waypoints)
    #     input("...")

    # # goto waypoints in sim
    # waypoints = initial_waypoints()
    # env = PoppyErgoEnv(show=True)
    # a = 0
    # input("...")
    # while True:
    #     for _ in range(32):
    #         env.step(waypoints[a])
    #     a = (a+1) % len(waypoints)
    #     # input("...")

    # goto policy output in sim
    M = initial_policy()
    # import matplotlib.pyplot as pt
    # pt.imshow(M)
    # pt.show()

    # try policy in sim
    env = Wrapper(timestep=1/240, control_period=10)
    env.render("human")
    obs = env.reset()
    while True:
        act = M @ obs
        print(act)
        print(obs)
        obs, reward, done, _ = env.step(act)
        if done: break


