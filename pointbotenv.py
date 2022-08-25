import numpy as np
import matplotlib.pyplot as pt

class PointBotEnv(object):
    """
    An environment where a point bot is rewarded for reaching a goal position
    Motions are continuous in a bounded 2d plane
    The bot approaches its current target position like a damped spring
    Some areas of the environment exert external "gravity" force on the bot
    Several episodes can be batched and run "in parallel" using numpy vectorization:
    
    observation: (num_domains, batch_size, 4) array of state observations
        obs[d,b,0:2] is current bot x,y position in episode b of domain d
        obs[d,b,2:4] is current bot x,y velocity in episode b of domain d
    action: (num_domains, batch_size, 2) array of actions
        act[d, b,:] is current target x,y position for bot in episode b in domain d
    reward: (num_domains, batch_size,) array of rewards
        rew[d, b] is current (negative) distance between bot and goal positions in episode b of domain d

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
        
        For under-damped (oscillating) spring dynamics, use damping < (4*mass*restore)**.5
        For faster bot motion, make restore larger relative to mass

        """
        self.mass = np.expand_dims(mass, axis=(1, 2))
        self.gravity = np.expand_dims(gravity, axis=(1, 2))
        self.restore = np.expand_dims(restore / mass, axis=(1, 2))
        self.damping = np.expand_dims(damping / mass, axis=(1, 2))
        self.dt = np.expand_dims(dt, axis=(1, 2))

        self.control_rate = control_rate
        self.num_domains = len(mass)

        self.goal = np.ones(2) * 0.9

        # gravity fields
        self.std = np.array([[.3, -.3], [.9, .9]]).T * 10 # (2, 2)
        self.mus = np.array([[0, 1], [.5, 0], [1, .5]]) # (num_mus, 2)

        self.reset()

    def current_observation(self):
        """
        Concatenate positions and velocities to form current observation
        """
        return np.concatenate((self.p, self.v), axis=-1)

    def gravity_field(self, position):
        """
        Calculate gravity field at given position, (broadcastable to) a corresponding (num_domains, batch_size, 2) array
        Returns g, a (num_domains, batch_size, 1) array of gravity magnitude at each position
        """
        diffs = position - np.expand_dims(self.mus, axis=(1,2)) # (num_mus, num_domains, batch_size, 2)
        g = np.exp(-((diffs @ self.std) ** 2).sum(axis=3, keepdims=True)).sum(axis=0) * self.gravity
        return g

    def reset(self, batch_size=1, state=None, seed=None, return_info=False):
        """
        Reset the environment to a new initial state
        state is a (num_domains, batch_size, 2) array
        """

        np.random.seed(seed)
        self.p = state[:,:,:2] if state is not None else np.ones((self.num_domains, batch_size, 2)) * 0.1
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
        Set the target bot position and update the environment with Euler's method
        action: (num_domains, batch_size, 2) array of target positions for the bot in each episode
        """

        action = self.bound(action)

        for cr in range(self.control_rate):

            g = self.gravity_field(self.p)

            a = self.restore * (action - self.p) - self.damping * self.v
            a += np.array([0, -1]) * g / self.mass

            self.p = self.bound(self.p + self.v * self.dt)
            self.v = self.v + a * self.dt

        observation = self.current_observation()
        reward = -np.linalg.norm(self.goal - self.p, axis=-1)
        done = False
        info = None
        return observation, reward, done, info
    
    def plot(self, ax):
        """
        Render current state with matplotlib
        ax: the matplotlib Axes object on which to draw
        
        Current states of all episodes in batch are drawn on the same plot
        Blue is bot, red is gravity, green is goal
        """

        # gravity field
        spacing = np.linspace(0, 1, 100)
        xpt, ypt = np.meshgrid(spacing, spacing)
        pts = np.stack((xpt.flatten(), ypt.flatten()), axis=-1)
        g = self.gravity_field(np.expand_dims(pts, axis=0)).mean(axis=0) # broadcast then average over domains
        g = g.reshape(xpt.shape)
        ax.contourf(xpt, ypt, g, levels = len(spacing), colors = np.array([1,1,1]) - spacing[:,np.newaxis] * np.array([0,1,1]))

        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')
        ax.plot(self.p[:,:,0].flatten(), self.p[:,:,1].flatten(), 'bo')
        ax.plot(self.goal[0], self.goal[1], 'go')
        
        # mark bot from episode 0
        ax.text(self.p[0,0,0], self.p[0,0,1], "0")

    def animate(self, policy, num_steps, ax, reset_batch_size=None):
        """
        Animate multiple steps of environment using provided policy
        policy(observation) should return (action, log_probability)
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
        for i in range(num_steps):
            action, _ = policy(observation)
            pt.cla()
            self.plot(ax)
            ax.plot(action[:,0], action[:,1], 'mo')
            pt.pause(.01)
            observation, reward, done, info = self.step(action)

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

    policy = lambda obs: (np.broadcast_to(env.goal, (num_domains, batch_size, 2)), None)
    # policy = lambda obs: (env.bound(np.random.randn(num_domains, batch_size, 2)*0.1 + env.goal), None)

    env.animate(policy, num_steps=200, ax=pt.gca(), reset_batch_size=batch_size)

    # Save the ending state
    pt.savefig("pointbotenv.png")

