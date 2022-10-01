import pickle as pk
import time
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp
import gym

with open("mcr.pkl","rb") as f: uniform_states = pk.load(f)
num_rollouts, num_steps = uniform_states.shape[:2]

with open("mcst.pkl","rb") as f: states = pk.load(f)

spacer = np.linspace(0, 1, num_steps)[:,np.newaxis]
colors = spacer * np.array([1,0,0]) + (1 - spacer) * np.array([0,0,1])

pt.figure(figsize=(4,6))

pt.subplot(2,1,1)

# goal rectangle
pt.gca().add_patch(mp.Rectangle((.45, 0), .15, .07, linewidth=0, facecolor='gray'))
for t in range(1,num_steps):
    sub = np.random.choice(np.arange(32), size=4, replace=False)
    pt.scatter(uniform_states[sub,t,0], uniform_states[sub,t,1], color=colors[t], marker='.')

pt.xlim([-1.2, 0.6])
pt.ylim([-0.07, 0.07])
pt.xlabel("Position")
pt.ylabel("Velocity")
pt.title("Blind rollouts")

pt.subplot(2,1,2)

# goal rectangle
pt.gca().add_patch(mp.Rectangle((.45, 0), .15, .07, linewidth=0, facecolor='gray'))
for t in range(1,len(states)):
    sub = np.random.choice(np.arange(len(states[t])), size=4, replace=False)
    pt.scatter(states[t][sub,0], states[t][sub,1], color=colors[t], marker='.')

pt.xlim([-1.2, 0.6])
pt.ylim([-0.07, 0.07])
pt.xlabel("Position")
pt.ylabel("Velocity")
pt.title("Gap Ratio Beam Search")

pt.tight_layout()
pt.show()

