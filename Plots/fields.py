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


def get_line(case, position, field):
    pos = '%.1f' % position
    file = glob.glob(f'../Data/{case}/postProcessing/lines/*/x{pos}_*{field}*.xy')
    file = file[0]
    data = np.loadtxt(file)
    return data


def get_probe(case, position, field):
    line = get_line(case, position[0], field)
    probe_location = np.abs(line[:, 0] - position[1]).argmin()
    probe_values = line[probe_location]
    probe = np.linalg.norm(probe_values[-3:-1])  # magnitude of velocity vector
    return probe


def add_cubes(ax, normal='y'):
    if normal == 'y':
        cube1 = patches.Rectangle((2, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
        cube2 = patches.Rectangle((7, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
    elif normal == 'z':
        cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
        cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////')
    else:
        print('normal must be y or z')
    ax.add_patch(cube1)
    ax.add_patch(cube2)


def plot_surface(ax, data, field, angle):
    magnitude = np.sqrt(data[f'{field}_x']**2 + data[f'{field}_y']**2 + data[f'{field}_z']**2)
    # contour = ax.tricontourf(data['x'], data['z'], data['UMean_x'], cmap='seismic', levels=np.arange(-1, 1.5, 0.1))
    # contour = ax.tricontourf(data['x'], data['z'], magnitude, cmap='bwr', levels=np.arange(0, 2, 0.05))
    contour = ax.tricontourf(data['x'], data['z'], magnitude, cmap='inferno', levels=np.arange(0, 1.5, 0.05))
    contour = ax.tricontour(data['x'], data['z'], data[f'{field}_x'], colors='w', linewidths=1, levels=[0])
    ax.set_aspect('equal')
    add_cubes(ax)
    ax.set_title(r'$\theta=%i$' % angle)


if __name__ == "__main__":
    # RANS
    print('Plotting RANS velocity')
    fig, axes = plt.subplots(7, 3, figsize=(12, 20), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
    for ang, i in zip(range(0, 41, 2), range(3*7)):
        print('angle', ang)
        acase = f'RANS/Yaw/a{ang}'
        velocity = get_surface(acase, field='UMean', surface='yNormal')
        plot_position = axes[int(np.floor(i/3)), int(i % 3)]
        plot_surface(plot_position, velocity, 'UMean', ang)
    for i in range(7):
        axes[i, 0].set_ylabel(r'$z$')
    for i in range(3):
        axes[-1, i].set_xlabel(r'$x$')

    # LES
    print('Plotting LES velocity')
    fig2, axes2 = plt.subplots(7, 1, figsize=(6, 20), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
    for ang, i in zip(range(0, 31, 5), range(1*7)):
        print('angle', ang)
        acase = f'LES/Yaw/a{ang}'
        velocity = get_surface(acase, field='U', surface='yNormal')
        plot_position = axes2[i, 0]
        plot_surface(plot_position, velocity, 'U', ang)
        axes2[i, 0].set_ylabel(r'$z$')
        axes2[-1, 0].set_xlabel(r'$x$')

    plt.show()
    # fig.savefig('velocityslicesRANS.png', bbox_inches='tight')
    fig2.savefig('velocityslicesLES.png', bbox_inches='tight')


