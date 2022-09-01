import numpy as np
import matplotlib.pyplot as pt

class PointBotEnv(object):
    """
    An environment where a point bot is rewarded for reaching a goal position
    Motions are continuous in a bounded 2d plane
    The bot approaches its current target position like a damped spring
    Some areas of the environment exert external "gravity" force on the bot
    Supports domain variation (different physics parameters, mass, gravity, etc. in different episodes)
    Supports batch processing of multiple domains and episodes "in parallel" using numpy vectorization:
    
    observation: (num_domains, batch_size, 4) array of state observations
        obs[d,b,0:2] is current bot x,y position in episode b of domain d
        obs[d,b,2:4] is current bot x,y velocity in episode b of domain d
    action: (num_domains, batch_size, 2) array of actions
        act[d,b,:] is current target x,y position for bot in episode b in domain d
    reward: (num_domains, batch_size,) array of rewards
        rew[d,b] is current (negative) distance between bot and goal positions in episode b of domain d

    """

    def __init__(self, mass, gravity, restore, damping, control_rate, dt):
        """
        Physical domain parameters, each with shape (num_domains,):
            mass: mass of the point bot
            gravity: maximum magnitude of "gravity" acceleration
            restore: restoring constant of spring emulated by the bot motion
            damping: damping constant of spring emulated by the bot motion
            control_rate: number of simulation steps in between each action command
            dt: small time delta between simulation steps used for Euler's method

        Providing arrays of physical domain parameters allows different domains to be batched
        For example mass[d] is the mass of the bot in domain d, etc.
        
        For under-damped (oscillating) spring dynamics, use damping < (4*mass*restore)**.5
        For faster bot motion, make restore larger relative to mass

        """

        # expand dims to broadcast across batched episodes within each domain
        self.mass = np.expand_dims(mass, axis=(1, 2))
        self.gravity = np.expand_dims(gravity, axis=(1, 2))
        self.restore = np.expand_dims(restore / mass, axis=(1, 2))
        self.damping = np.expand_dims(damping / mass, axis=(1, 2))
        self.dt = np.expand_dims(dt, axis=(1, 2))

        self.control_rate = control_rate
        self.num_domains = len(mass)

        # self.goal = np.ones(2) * 0.9

        # # gravity field: sum of three Gaussians with equal variance
        # self.std = np.array([[.3, -.3], [.9, .9]]).T * 10 # (2, 2)
        # self.mus = np.array([[0, 1], [.5, 0], [1, .5]]) # (num_mus, 2)

        self.goal = np.array([.99, .01])

        # gravity field: sum of three Gaussians with equal variance
        self.std = np.array([[1., 0], [0, .15]]).T * 14 # (2, 2)
        self.mus = np.array([[.2, 0], [.5, 1], [.8, 0]]) # (num_mus, 2)
        self.drs = np.array([[-1, -1], [-1, 1], [-1, -1]]) # (num_mus, 2)

        self.reset()

    def current_observation(self):
        """
        Concatenate positions and velocities to form current observation
        """
        return np.concatenate((self.p, self.v), axis=-1)

    def gravity_field(self, position):
        """
        Calculate gravity field at given position, (broadcastable to) a corresponding (num_domains, batch_size, 2) array
        Returns g, a (num_domains, batch_size, 2) array of gravity acceleration vectors at each position
        batch_size refers to the number of positions given per domain
        """
        diffs = position - np.expand_dims(self.mus, axis=(1,2)) # (num_mus, num_domains, batch_size, 2)
        dirs = np.expand_dims(self.drs, axis=(1,2)) # (num_mus, num_domains, batch_size, 2)
        g = (np.exp(-((diffs @ self.std) ** 2).sum(axis=3, keepdims=True)) * dirs).sum(axis=0) * self.gravity
        # g = (np.exp(-((diffs @ self.std) ** 4).sum(axis=3, keepdims=True)) * dirs).sum(axis=0) * self.gravity
        return g

    def reset(self, batch_size=1, state=None, seed=None, return_info=False):
        """
        Reset the environment to a new initial state
        state is a (num_domains, batch_size, 2) array of initial states for each episode in each domain of the batch
        if state is None, batch_size argument is used to initialize all states near the origin
        """

        np.random.seed(seed)
        self.p = state[:,:,:2] if state is not None else np.ones((self.num_domains, batch_size, 2)) * 0.
        self.v = state[:,:,2:] if state is not None else np.zeros((self.num_domains, batch_size, 2))

        observation = self.current_observation()
        info = None

        if return_info: return (observation, info)
        else: return observation

    def bound(self, points):
        """
        Clip points to lie within the environment plane boundaries
        """
        return np.clip(points, 0, 1)

    def step(self, action):
        """
        Set the target bot position and update the environment with several simulation steps of Euler's method
        action: (num_domains, batch_size, 2) array of target positions for the bot in each episode of each domain
        """

        action = self.bound(action)

        # run several iterations of simulation for each "step" when actions are commanded
        for cr in range(self.control_rate):

            # acceleration from gravity and spring dynamics
            a = self.restore * (action - self.p) - self.damping * self.v
            a += self.gravity_field(self.p) / self.mass

            # Euler updates to position and velocity
            self.p = self.bound(self.p + self.v * self.dt) # don't go outside the bounds
            self.v = self.v + a * self.dt
            self.v = (self.bound(self.p + self.v * self.dt) - self.p) / self.dt # squash velocity at the bounds

        observation = self.current_observation()
        reward = -np.linalg.norm(self.goal - self.p, axis=-1)
        done = False
        info = None
        return observation, reward, done, info
    
    def plot(self, ax):
        """
        Render current state(s) with matplotlib
        ax: the matplotlib Axes object on which to draw
        
        Current states of all episodes in all domains are drawn on the same plot
        Blue is bot, red is gravity, green is goal
        """

        # gravity field magnitude
        spacing = np.linspace(0, 1, 100)
        xpt, ypt = np.meshgrid(spacing, spacing)
        pts = np.stack((xpt.flatten(), ypt.flatten()), axis=-1)
        g = self.gravity_field(np.expand_dims(pts, axis=0)).mean(axis=0) # broadcast then average over domains for visualization
        g = np.linalg.norm(g, axis=-1) # get magnitude from acceleration vectors
        g = g.reshape(xpt.shape) # reshape for contourf
        ax.contourf(xpt, ypt, g, levels = len(spacing), colors = np.array([1,1,1]) - spacing[:,np.newaxis] * np.array([0,1,1]))

        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')
        ax.plot(self.p[:,:,0].flatten(), self.p[:,:,1].flatten(), 'bo')
        ax.plot(self.goal[0], self.goal[1], 'go')
        
        # mark bot from episode 0 of domain 0
        ax.text(self.p[0,0,0], self.p[0,0,1], "0")

    def animate(self, policy, num_steps, ax, reset_batch_size=None):
        """
        Animate multiple steps of environment using provided policy
        policy(observation) should return (action, log_probability)
            log_probability can be None if not using a policy gradient method
        target positions are shown in magenta
        ax: the matplotlib Axes object for rendering
        resets unless reset_batch_size == None
            otherwise uses provided batch size for reset
        All episodes in batch are animated on the same plot
        """
        if reset_batch_size is not None:
            observation = self.reset(reset_batch_size)
        else:
            observation = self.current_observation()
        pt.ion()
        pt.show()
        reward = np.empty((num_steps,) + observation.shape[:2])
        for t in range(num_steps):
            action, _ = policy(observation)
            pt.cla()
            self.plot(ax)
            ax.plot(action[:,:,0].flatten(), action[:,:,1].flatten(), 'mo')
            pt.pause(.01)
            observation, reward[t], done, info = self.step(action)
        return reward

class FixedPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.reset()
    def reset(self):
        self.t = -1
    def __call__(self, observation):
        self.t += 1
        return (self.actions[self.t], None)

def ExpertPolicy(num_steps):
    actions = np.empty((num_steps, 1, 1, 2))

    actions[:50] = np.array([[[0, 1]]])
    actions[50:100] = np.array([[[.5, .25]]])
    actions[100:150] = np.array([[[1, 1]]])
    actions[150:] = np.array([[[1, 0]]])

    # actions[:50] = np.array([[[.1, .5]]])
    # actions[50:100] = np.array([[[1, .6]]])
    # actions[100:] = np.array([[[1, 0]]])

    return FixedPolicy(actions)

if __name__ == "__main__":

    # Set up spring parameters for bot motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = np.random.uniform(.25, .9)*critical # random underdamping

    num_domains = 3
    batch_size = 4

    mass = m + np.random.randn(num_domains) * 0.1
    gravity = 10 + np.random.randn(num_domains)
    restore = k + np.random.randn(num_domains) * 0.1
    damping = b + np.random.randn(num_domains) * 0.1

    control_rate = 10
    dt = 1/240 * np.ones(num_domains)

    env = PointBotEnv(mass, gravity, restore, damping, control_rate, dt)
    
    pt.figure(figsize=(5, 5), constrained_layout=True)
    # env.plot(pt.gca())
    # pt.show()

    # Some dummy policies (not optimized)
    
    # # bee-line to target at goal (may be outweighed by gravity)
    # policy = lambda obs: (np.broadcast_to(env.goal, (num_domains, batch_size, 2)), None)

    # # random targets near goal every step
    # policy = lambda obs: (env.bound(np.random.randn(num_domains, batch_size, 2)*0.1 + env.goal), None)

    # # bee-line to fixed random targets
    # targets = np.random.rand(num_domains, batch_size, 2)
    # policy = lambda obs: (targets, None)

    # "expert" policy crafted by hand
    policy = ExpertPolicy(num_steps=200)

    reward = env.animate(policy, num_steps=200, ax=pt.gca(), reset_batch_size=batch_size)
    print(reward)
    print(reward.sum(axis=0))

    # Save the ending state
    pt.savefig("pointbotenv.png")

