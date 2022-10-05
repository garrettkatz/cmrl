"""
Domain details:
https://www.gymlibrary.dev/environments/mujoco/walker2d/
"""
import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection, PatchCollection

def draw(states):

    num_steps = len(states)
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for t in range(len(states)):
        # TODO: use first two principle components instead of coordinates
        pt.scatter(states[t][:,0], states[t][:,1], color=colors[t], marker='.')

    # pt.xlim([-1.2, 0.6])
    # pt.ylim([-0.07, 0.07])
    # pt.xlabel("Position")
    # pt.ylabel("Velocity")

    pt.pause(0.01)
    pt.show()


def show():

    with open("pw_grbs.pkl","rb") as f: states = pk.load(f)

    pt.ioff()
    pt.cla()
    draw(states)


if __name__ == "__main__":
    show()


