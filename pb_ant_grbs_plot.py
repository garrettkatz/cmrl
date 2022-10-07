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

    # pca the states
    X = np.concatenate(states, axis=0)
    Xm = X.mean(axis=0)
    Xc = X - Xm
    w, v = np.linalg.eigh(Xc.T @ Xc)
    PC = v[:,-2:] # top-2 PCs
    

    num_steps = len(states)
    spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
    colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

    for t in range(len(states)):
        # project onto top-2 PCs
        proj = (states[t] - Xm) @ PC
        # TODO: use first two principle components instead of coordinates
        pt.scatter(proj[:,0], proj[:,1], color=colors[t], marker='.')

    pt.pause(0.01)
    pt.show()

def show():

    with open("pa_grbs.pkl","rb") as f: (states, plan) = pk.load(f)

    pt.ioff()
    pt.cla()
    draw(states)


if __name__ == "__main__":
    show()


