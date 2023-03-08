#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from matplotlib import pyplot as plt

parent = os.path.abspath('../')
sys.path.insert(1, parent)

from mfRegression import MFRegress
import fonts
from analyticalFunc import Surface

np.random.seed(4)

surf1 = Surface('Sine', 'GPR')
# surf1.k_hf *= DotProduct()
# surf1.regression()

# PLOTTING FOR TESTING
fig = plt.figure(figsize=(15, 10))

axs = [None] * 6

# Plot exact solution
for i in range(len(axs)):
    axs[i] = fig.add_subplot(2, 3, i + 1, projection='3d')
    # axs[i].axes(projection='3d')
    x, y = np.meshgrid(surf1.X, surf1.Y)
    axs[i].plot_surface(x, y, surf1.hf(surf1.X),  # rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', alpha=0.6)

    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')
    axs[i].set_zlabel('z')

# Plot LF 0D samples
for i in [1, 3, 5]:
    axs[i].scatter3D(surf1.X_lf, np.zeros(len(surf1.X_lf)), surf1.lf(surf1.X_lf), c='tab:red', depthshade=False)

# Plot HF 1D samples
for i in [2, 4, 5]:
    for j in range(len(surf1.X_hf)):
        axs[i].plot3D(surf1.X_hf[j] * np.ones(len(surf1.Y)), surf1.Y, surf1.hf(surf1.X_hf)[:, j], 'b')

surf1.regression()

x, y = np.meshgrid(surf1.X, surf1.Y)
axs[5].plot_surface(x, y, surf1.pred_mf_mean.T,  # rstride=1, cstride=1,
                    cmap='inferno', edgecolor='none', alpha=0.8)
axs[4].plot_surface(x, y, surf1.pred_hf_mean.T,  # rstride=1, cstride=1,
                    cmap='inferno', edgecolor='none', alpha=0.8)
axs[3].plot_surface(x, y, surf1.pred_lf_mean.T,  # rstride=1, cstride=1,
                    cmap='inferno', edgecolor='none', alpha=0.8)

axs[0].set_title('Exact Solution')
axs[1].set_title('LF 0D Samples')
axs[2].set_title('HF 1D Samples')
axs[3].set_title('LF Model')
axs[4].set_title('HF Model')
axs[5].set_title('MF Model')


def on_move(event):
    if event.inaxes == axs[0]:
        for ax in axs[1:]:
            ax.view_init(elev=axs[0].elev, azim=axs[0].azim)
    else:
        return
    fig.canvas.draw_idle()


c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()
