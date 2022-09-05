import numpy as np

class BufferedEnv:

    def __init__(self, env, M, obs_pad=0., act_pad=0.):
        self.env = env
        self.M = M
        self.obs_pad = obs_pad
        self.act_pad = act_pad

        self.num_domains = env.num_domains
        self.act_size = env.act_size
        self.obs_size = M * (env.obs_size + env.act_size)

    def current_observation(self):
        cat = np.concatenate((self.observation, self.action), axis=-1) # (D, B, M, S+A)
        rot = cat[:,:,self.idx,:]
        rsh = rot.reshape(cat.shape[0], cat.shape[1], -1) # (D, B, M*(S+A))
        return rsh

    def reset(self, batch_size=1, state=None, seed=None, return_info=False):
        # assumes return_info is False
        observation, info = self.env.reset(batch_size, state, seed, return_info = True)
        D, B = observation.shape[:2]

        self.observation = np.full((D, B, self.M, self.env.obs_size), self.obs_pad)
        self.action = np.full((D, B, self.M, self.env.act_size), self.act_pad)
        self.observation[:,:,0,:] = observation

        self.idx = np.arange(self.M)

        observation = self.current_observation()
        if return_info: return (observation, info)
        else: return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.idx = (self.idx - 1) % self.M

        self.action[:,:,self.idx[0],:] = action
        self.observation[:,:,self.idx[0],:] = observation

        return self.current_observation(), reward, done, info

    def animate(self, *args, **kwargs):
        self.env.animate(*args, **kwargs)

def main():
    class MockEnv:
        def __init__(self, D, B, O, A):
            self.D = D
            self.B = B
            self.O = O
            self.A = A
            self.obs_size, self.act_size = O, A
        def reset(self, batch_size, state, seed, return_info):
            self.t = 1.
            return np.full((self.D, self.B, self.O), self.t), None
        def step(self, action):
            self.t += 1
            return np.full((self.D, self.B, self.O), self.t), 0, False, None

    env = BufferedEnv(MockEnv(1, 4, 2, 1), M=1)
    obs, _ = env.reset()
    print(obs)
    for t in range(4):
        obs, _, _, _ = env.step(np.full((1, 4, 1), t+.5))
        print(t)
        print(obs)


if __name__ == "__main__": main()
