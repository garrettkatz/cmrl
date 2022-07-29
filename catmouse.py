import numpy as np
import matplotlib.pyplot as pt

def random_directions(batch_size):
    """
    Create a (batch_size, 2) array of random 2d unit vectors
    """
    v = np.random.randn(batch_size, 2)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

class CatMouseEnv(object):
    """
    An environment where a mouse agent is rewarded for avoiding a cat
    Motions are continuous in a bounded 2d plane
    The mouse approaches its current target position like a damped spring
    The cat moves with constant speed in randomly changing directions
    Several episodes can be batched and run "in parallel" using numpy vectorization:
    
    observation: (batch_size, 8) array of state observations
        obs[b,0:2] is current cat x,y position in episode b
        obs[b,2:4] is current cat x,y velocity in episode b
        obs[b,4:6] is current mouse x,y position in episode b
        obs[b,6:8] is current mouse x,y velocity in episode b
    action: (batch_size, 2) array of actions
        act[b,:] is current target x,y position for mouse in episode b
    reward: (batch_size,) array of rewards
        rew[b] is current distance between cat and mouse positions

    """

    def __init__(self, width, height, cat_speed, spring, damping, dt, random_cat=True, batch_size=1):
        """
        (width, height): environment extends from 0 to width in the x-direction and 0 to height in the y-direction
        cat_speed: constant speed for cat (although direction changes)
        spring: k/m, where k is the spring constant and m is the spring mass emulated by the mouse motion
        damping: b/m, where b is the spring damping constant emulated by the mouse motion
        dt: small delta between time-steps used for Euler's method
        random_cat: whether to randomly update cat direction
        batch_size: the number of episodes that are run in parallel
        
        For under-damped (oscillating) spring dynamics, use b < (4*m*k)**.5
        For faster mouse motion, make k larger relative to m
        
        """
        self.width = width
        self.height = height
        self.cat_speed = cat_speed
        self.spring = spring
        self.damping = damping
        self.dt = dt
        self.random_cat = random_cat
        self.batch_size = batch_size
        self.shape = np.array([width, height])
        self.reset()

    def current_observation(self):
        """
        Concatenate positions and velocities to form current observation
        """
        return np.concatenate((self.cp, self.cv, self.mp, self.mv), axis=1)

    def reset(self, cp=None, cv=None, mp=None, mv=None, seed=None, return_info=False):
        """
        Reset the environment to a new random initial state
        Random initial positions and velocities can be overwritten by passing any of:
            cp: cat positions
            cv: cat velocity
            mp: mouse position
            mv: velocity position
        Each should be a corresponding (batch_size, 2) array
        """

        np.random.seed(seed)
        self.cp = cp if cp is not None else np.random.rand(self.batch_size, 2)*self.shape
        self.cv = cv if cv is not None else random_directions(self.batch_size)*self.cat_speed
        self.mp = mp if mp is not None else np.random.rand(self.batch_size, 2)*self.shape
        self.mv = mv if mv is not None else np.zeros((self.batch_size, 2))

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
        Set the target mouse position and update the environment with one Euler step
        action: (batch_size, 2) array of target positions for the mouse in each episode
        cat_walk: if True, cat's direction of motion changes randomly, otherwise remains fixed
        """

        action = self.bound(action)
        ma = self.spring * (action - self.mp) - self.damping * self.mv
        self.mp = self.bound(self.mp + self.mv * self.dt)
        self.mv += ma * self.dt

        self.cp = self.bound(self.cp + self.cv * self.dt)
        if self.random_cat:
            self.cv += np.random.randn(self.batch_size, 2)
            self.cv *= self.cat_speed / np.linalg.norm(self.cv, axis=1, keepdims=True)

        observation = self.current_observation()
        reward = np.linalg.norm(self.cp - self.mp, axis=1)
        done = False
        info = None
        return observation, reward, done, info
    
    def plot(self, ax):
        """
        Render current state with matplotlib
        ax: the matplotlib Axes object on which to draw
        
        Current states of all episodes in batch are drawn on the same plot
        Blue is mouse, red is cat
        """
        # b = 0 # only show first element of batch
        b = slice(self.batch_size) # show all elements of batch
        ax.plot([0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0], 'k-')
        ax.plot(self.cp[b,0], self.cp[b,1], 'ro')
        ax.plot(self.mp[b,0], self.mp[b,1], 'bo')
        
        # mark cat and mouse from episode 0
        ax.text(self.mp[0,0], self.mp[0,1], "0")
        ax.text(self.cp[0,0], self.cp[0,1], "0")

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

    # Initialize environment to run 10 episodes in parallel
    env = CatMouseEnv(
        width = 5,
        height = 5,
        cat_speed = 2,
        spring = k/m,
        damping = b/m,
        dt = 1/24,
        random_cat = True,
        batch_size=10)

    # Use same initial position and velocity for cat in all episodes
    env.reset(
        seed = 42,
        cp = 0.25*np.ones((env.batch_size, 2))*env.shape,
        cv = np.tile(np.array([1., 0.]), (env.batch_size,1))*env.cat_speed,
        # mp = 0.5*np.ones((env.batch_size, 2))*env.shape
    )

    # # random mouse policy
    # policy = lambda obs: (np.random.rand(env.batch_size, 2)*env.shape, None)

    # constant mouse policy, always uses (1,1) as target position
    action = np.ones((env.batch_size, 2))
    # action[:env.batch_size//2,0] = 0
    policy = lambda obs: (action, None)

    # Animate the environment
    pt.figure(figsize=(4,4), constrained_layout=True)
    env.animate(policy, num_steps=50, ax=pt.gca())

    # Save the ending state
    pt.savefig("catmouseenv.pdf")

