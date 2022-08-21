import numpy as np
import matplotlib.pyplot as pt

class PointBotEnv(object):
    """
    An environment where a point bot is rewarded for reaching a goal position
    Motions are continuous in a bounded 2d plane
    The bot approaches its current target position like a damped spring
    Some areas of the environment exert external "gravity" force on the bot
    Several episodes can be batched and run "in parallel" using numpy vectorization:
    
    observation: (batch_size, 4) array of state observations
        obs[b,0:2] is current bot x,y position in episode b
        obs[b,2:4] is current bot x,y velocity in episode b
    action: (batch_size, 2) array of actions
        act[b,:] is current target x,y position for bot in episode b
    reward: (batch_size,) array of rewards
        rew[b] is current distance between bot and goal positions

    """

    def __init__(self, width, height, gravity, mass, spring_constant, damping, dt, batch_size=1):
        """
        (width, height): environment extends from 0 to width in the x-direction and 0 to height in the y-direction
        gravity: maximum magnitude of "gravity" acceleration
        mass: mass of the point bot
        spring_constant: spring constant emulated by the bot motion
        damping: damping constant emulated by the bot motion
        dt: small delta between time-steps used for Euler's method
        batch_size: the number of episodes that are run in parallel
        
        For under-damped (oscillating) spring dynamics, use b < (4*m*k)**.5
        For faster bot motion, make k larger relative to m
        
        """
        self.width = width
        self.height = height
        self.gravity = gravity
        self.mass = mass
        self.spring_coef = spring_constant / mass
        self.damping_coef = damping / mass
        self.dt = dt
        self.batch_size = batch_size
        self.shape = np.array([width, height])
        self.goal = np.ones((1, 2)) * self.shape * 0.9

        # gravity fields
        self.std = (np.array([[.3, -.3], [.9, .9]]) * self.shape).T / 5
        self.mus = np.array([[0, 1], [.5, 0], [1, .5]])[:,np.newaxis,:] * self.shape

        self.reset()

    def current_observation(self):
        """
        Concatenate positions and velocities to form current observation
        """
        return np.concatenate((self.p, self.v), axis=1)

    def gravity_field(self, position):
        """
        Calculate gravity field at given position, a corresponding (batch_size, 2) array
        """
        diffs = position - self.mus
        g = np.exp(-((diffs @ self.std) ** 2).sum(axis=2)).sum(axis=0) * self.gravity
        return g[:,np.newaxis]

    def reset(self, p=None, v=None, seed=None, return_info=False):
        """
        Reset the environment to a new initial state
        Initial positions and velocities can be overwritten by passing any of:
            p: bot position
            v: bot velocity
        Each should be a corresponding (batch_size, 2) array
        """

        np.random.seed(seed)
        self.p = p if p is not None else np.ones((self.batch_size, 2))*self.shape*0.1
        self.v = v if v is not None else np.zeros((self.batch_size, 2))

        observation = self.current_observation()
        info = None

        if return_info: return (observation, info)
        else: return observation

    def bound(self, position):
        """
        Clip position to lie within the environment plane boundaries
        """
        return np.maximum(0, np.minimum(position, self.shape))

    def step(self, action):
        """
        Set the target bot position and update the environment with one Euler step
        action: (batch_size, 2) array of target positions for the bot in each episode
        """

        action = self.bound(action)
        a = self.spring_coef * (action - self.p) - self.damping_coef * self.v
        a += np.array([0, -1]) * self.gravity_field(self.p) / self.mass

        self.p = self.bound(self.p + self.v * self.dt)
        self.v = self.v + a * self.dt

        observation = self.current_observation()
        reward = np.linalg.norm(self.goal - self.p, axis=1)
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
        xpt, ypt = np.meshgrid(spacing*self.shape[0], spacing*self.shape[1])
        pts = np.stack((xpt.flatten(), ypt.flatten()), axis=-1)
        g = self.gravity_field(pts)
        g = g.reshape(xpt.shape)
        ax.contourf(xpt, ypt, g, levels = len(spacing), colors = np.array([1,1,1]) - spacing[:,np.newaxis] * np.array([0,1,1]))

        # b = 0 # only show first element of batch
        b = slice(self.batch_size) # show all elements of batch
        ax.plot([0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0], 'k-')
        ax.plot(self.p[b,0], self.p[b,1], 'bo')
        ax.plot(self.goal[0,0], self.goal[0,1], 'go')
        
        # mark cat and bot from episode 0
        ax.text(self.p[0,0], self.p[0,1], "0")

    def animate(self, policy, num_steps, ax, reset=True):
        """
        Animate multiple steps of environment using provided policy
        policy(observation) should return (action, log_probability)
        ax: the matplotlib Axes object for rendering
        randomly resets unless reset==False
        All episodes in batch are animated on the same plot
        """
        observation = self.reset() if reset else self.current_observation()
        pt.ion()
        pt.show()
        for i in range(num_steps):
            pt.cla()
            self.plot(ax)
            pt.pause(.01)            
            action, _ = policy(observation)
            observation, reward, done, info = self.step(action)

if __name__ == "__main__":

    # Set up spring parameters for mouse motion
    k = 2
    m = 1
    critical = (4*m*k)**.5 # critical damping point
    b = np.random.uniform(.25, .9)*critical # random underdamping

    width = 5
    height = 5
    gravity = 10
    mass = m
    spring_constant = k
    damping = b
    dt = 1/24
    batch_size=10

    env = PointBotEnv(width, height, gravity, mass, spring_constant, damping, dt, batch_size)
    
    pt.figure(figsize=env.shape, constrained_layout=True)
    # env.plot(pt.gca())
    # pt.show()

    # policy = lambda obs: (np.broadcast_to(env.goal, (batch_size, 2)), None)
    policy = lambda obs: (np.random.randn(batch_size, 2)*3 + env.goal, None)

    env.animate(policy, num_steps=100, ax=pt.gca(), reset=True)

    # Save the ending state
    pt.savefig("pointbotenv.png")

