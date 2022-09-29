import time
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.collections import LineCollection, PatchCollection

import pickle as pk
with open("mcst.pkl","rb") as f: states = pk.load(f)

num_steps = len(states)
spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

for t in range(len(states)):
    pt.scatter(states[t][:,0], states[t][:,1], color=colors[t], marker='.')

pt.xlim([-1.2, 0.6])
pt.ylim([-0.07, 0.07])
pt.xlabel("Position")
pt.ylabel("Velocity")

pt.pause(0.01)
pt.show()


