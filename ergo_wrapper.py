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

