import os
import numpy as np
import matplotlib.pyplot as pt

def plot_metrics(root_path):
    with open(os.path.join(root_path, 'progress.pkl'), 'rb') as f:
        (metrics, M, mean, var, nx) = pk.load(f)

    updates = np.arange(len(metrics['lifetime']))
    walltime = np.cumsum(metrics['runtime']) / (60*60)

    fig, ax = pt.subplots(2, 2, layout='constrained')
    for c,key in enumerate(['lifetime','reward']):
        for r,xticks in enumerate([updates, walltime]):
            ax[r,c].plot(xticks, metrics[key])
            ax[r,c].set_ylabel(key)
            ax[r,c].set_xlabel(["Update","Wall time (hrs)"][r])

    pt.show()

