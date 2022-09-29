import matplotlib.pyplot as pt
import numpy as np
import pickle as pk

with open("mcb.pkl","rb") as f: (success_indicator, success, run_times) = pk.load(f)

num_reps = len(success_indicator)

uni_avg, rw_avg = success_indicator[:,0,:].mean(axis=0), success_indicator[:,1,:].mean(axis=0)
uni_std, rw_std = success_indicator[:,0,:].std(axis=0), success_indicator[:,1,:].std(axis=0)
xpts = np.arange(len(uni_avg))

pt.subplot(2,1,1)
ax = pt.gca()

ax.fill_between(xpts, uni_avg-uni_std, uni_avg+uni_std, color='r', alpha=0.2)
ax.fill_between(xpts, rw_avg-rw_std, rw_avg+rw_std, color='b', alpha=0.2)

ax.plot(xpts, uni_avg, color='r', linestyle='-', label="Uniform")
ax.plot(xpts, rw_avg, color='b', linestyle='-', label="Random walk")
ax.legend(loc='lower right')
ax.set_ylabel("Pr(success)")
ax.set_xlabel("Num env steps")

pt.subplot(2,1,2)
uni_xpts = np.concatenate(([0.], np.sort(run_times[success[:,0],0])))
rw_xpts = np.concatenate(([0.], np.sort(run_times[success[:,1],1])))
uni = np.triu(np.ones((num_reps, len(uni_xpts))), k=1)
rw = np.triu(np.ones((num_reps, len(rw_xpts))), k=1)
if rw_xpts[-1] > uni_xpts[-1]:
    uni_xpts = np.append(uni_xpts, [rw_xpts[-1]])
    uni = np.concatenate((uni, uni[:,-1:]), axis=1)
elif rw_xpts[-1] < uni_xpts[-1]:
    rw_xpts = np.append(rw_xpts, [uni_xpts[-1]])
    rw = np.concatenate((rw, rw[:,-1:]), axis=1)


uni_avg, rw_avg = uni.mean(axis=0), rw.mean(axis=0)
uni_std, rw_std = uni.std(axis=0), rw.std(axis=0)

ax = pt.gca()

ax.fill_between(uni_xpts, uni_avg-uni_std, uni_avg+uni_std, color='r', alpha=0.2)
ax.fill_between(rw_xpts, rw_avg-rw_std, rw_avg+rw_std, color='b', alpha=0.2)

ax.plot(uni_xpts, uni_avg, color='r', linestyle='-', marker='.', label="Uniform")
ax.plot(rw_xpts, rw_avg, color='b', linestyle='-', marker='.', label="Random walk")
ax.legend(loc='lower right')
ax.set_ylabel("Pr(success)")
ax.set_xlabel("Wall Time (s)")

pt.tight_layout()
pt.show()


