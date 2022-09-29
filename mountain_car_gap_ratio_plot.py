import matplotlib.pyplot as pt
import numpy as np
import pickle as pk

with open("mcgr.pkl","rb") as f: (success_indicator, success, run_times) = pk.load(f)

num_reps = len(success_indicator)
print(success_indicator)

avg = success_indicator.mean(axis=0)
std = success_indicator.std(axis=0)
xpts = np.arange(len(avg))

pt.subplot(2,1,1)
ax = pt.gca()

ax.fill_between(xpts, avg-std, avg+std, color='r', alpha=0.2)

ax.plot(xpts, avg, color='r', linestyle='-', label="GR")
ax.legend(loc='lower right')
ax.set_ylabel("Pr(success)")
ax.set_xlabel("Num env steps")

pt.subplot(2,1,2)
xpts = np.concatenate(([0.], np.sort(run_times[success])))
uni = np.triu(np.ones((num_reps, len(xpts))), k=1)

avg = uni.mean(axis=0)
std = uni.std(axis=0)

ax = pt.gca()

ax.fill_between(xpts, avg-std, avg+std, color='r', alpha=0.2)

ax.plot(xpts, avg, color='r', linestyle='-', marker='.', label="GR")
ax.legend(loc='lower right')
ax.set_ylabel("Pr(success)")
ax.set_xlabel("Wall Time (s)")

pt.tight_layout()
pt.show()


