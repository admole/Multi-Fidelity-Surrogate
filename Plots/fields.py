#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import fonts
import json
import glob
import pprint


def get_surface(case, u_inf=1, surface="zNormal", field="UMean"):
    file = glob.glob(f'../Data/{case}/postProcessing/surfaces/*/{field}_{surface}.raw')
    file = file[0]
    with open(file) as f:
        line = f.readline()
        cnt = 0
        while line.startswith('#'):
            prev_line = line
            line = f.readline()
            cnt += 1
            # print(prev_line)
    header = prev_line.strip().lstrip('# ').split()
    data = pd.read_csv(file, comment='#', sep=r'\s+', names=header, header=None, engine='python')

    return data


def add_cubes(ax, normal='y'):
    if normal=='y':
        cube1 = patches.Rectangle((2, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
        cube2 = patches.Rectangle((7, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
    elif normal=='z':
        cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
        cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
    else:
        print('normal must be y or z')
    ax.add_patch(cube1)
    ax.add_patch(cube2)


def plot_surface(ax, data, angle):
    magnitude = np.sqrt(data['UMean_x']**2 + data['UMean_y']**2 + data['UMean_z']**2)
    # contour = ax.tricontourf(data['x'], data['z'], data['UMean_x'], cmap='seismic', levels=np.arange(-1, 1.5, 0.1))
    contour = ax.tricontourf(data['x'], data['z'], magnitude, cmap='inferno', levels=np.arange(0, 1.5, 0.05))
    contour = ax.tricontour(data['x'], data['z'], data['UMean_x'], colors='w', linewidths=1, levels=[0])
    ax.set_aspect('equal')
    add_cubes(ax)
    ax.set_title(r'$\theta=%i$' %angle)


fig, ax = plt.subplots(3, 7, figsize=(20, 7), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
for angle, i in zip(range(0, 41, 2), range(3*7)):
# for angle, i in zip([0], [0]):
    print(angle)
    case = f'RANS/Yaw/a{angle}'
    data = get_surface(case, field='UMean', surface='yNormal')
    position = ax[int(np.floor(i/7)), int(i%7)]
    plot_surface(position, data, angle)

plt.show()
fig.savefig('velocityslices.png', bbox_inches='tight')

