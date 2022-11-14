#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import fonts
import glob
import argparse
import PIL


def get_surface(case, u_inf=1, surface="zNormal", field="UMean"):
    file = glob.glob(f'../Data/{case}/postProcessing/surfaces/*/{field}_{surface}.raw')
    file = file[-1]
    with open(file) as f:
        line = f.readline()
        cnt = 0
        while line.startswith('#'):
            prev_line = line
            line = f.readline()
            cnt += 1
    header = prev_line.strip().lstrip('# ').split()
    data = pd.read_csv(file, comment='#', sep=r'\s+', names=header, header=None, engine='python')

    return data


def get_line(case, position, field):
    pos = '%.1f' % position
    file = glob.glob(f'../Data/{case}/postProcessing/lines/*/x{pos}_*{field}*.xy')
    file = file[-1]
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
        cube1 = patches.Rectangle((2, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
        cube2 = patches.Rectangle((7, -0.5), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
    elif normal == 'z':
        cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
        cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='/////', zorder=10)
    else:
        print('normal must be y or z')
    ax.add_patch(cube1)
    ax.add_patch(cube2)


def plot_surface(ax, data, field, angle):
    normal = 'z' if SURFACE[0] == 'y' else 'y'
    if field == 'UMean' or field == 'U':
        magnitude = np.sqrt(data[f'{field}_x']**2 + data[f'{field}_y']**2 + data[f'{field}_z']**2)
        loc1, loc2 = data['x'], data[normal]
    elif field == 'kMean':
        magnitude = data[f'{field}']
        loc1, loc2 = data['x'], data[normal]
    elif field == 'UPrime2Mean':
        magnitude = 0.5 * (data[f'yy'] + data[f'{field}_2'] + data[f'{field}_2'])
        loc1, loc2 = data['xx'], data[f'x{normal}']

    contour = ax.tricontourf(loc1, loc2, magnitude, cmap='inferno', levels=np.arange(0, 1.5, 0.01), antialiased=False)
    ax.set_aspect('equal')
    ax.set_ylim(-4.25, 4.25)
    add_cubes(ax, SURFACE[0])
    ax.annotate(fr'$\alpha = {angle}^\circ$', xy=(0.2, 3), xytext=(0.2, 3), size=fonts.BIG_SIZE)
    return contour


def draw(it, ax1, ax2, angles):
    ang = angles[it]
    print('\nangle', ang)

    if any(x == ang for x in RANS_CASES):
        print('Plotting RANS velocity')
        acase = f'RANS/Yaw/a{ang}'
        velocity = get_surface(acase, field='UMean', surface=SURFACE)
        contour = plot_surface(ax1, velocity, 'UMean', ang)

    if any(x == ang for x in LES_CASES):
        print('Plotting LES velocity')
        acase = f'LES/Yaw/a{ang}'
        velocity = get_surface(acase, field='U', surface=SURFACE)
        contour = plot_surface(ax2[0], velocity, 'U', ang)
        velocity = get_surface(acase, field='UMean', surface=SURFACE)
        contour = plot_surface(ax2[1], velocity, 'UMean', ang)

    return contour


RANS_CASES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
LES_CASES = [0, 5, 10, 15, 20, 25, 30]
ALL_CASES = list(set(RANS_CASES) | set(LES_CASES))
SURFACE = 'yNormal'  # 'yNormal' 'yHalf'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Create animation')
    args = parser.parse_args()
    if args.a:
        # Animation
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), squeeze=False, constrained_layout=True)
        axes[0, 0].set_ylabel(r'$z$')
        axes[0, 0].set_xlabel(r'$x$')
        n_cases = len(ALL_CASES)
        anim = animation.FuncAnimation(fig, draw, fargs=(axes[0, 0], axes[0, 0], ALL_CASES), frames=n_cases, interval=1, blit=False)
        anim.save('animations/yaw_animation.mp4', fps=1, dpi=400)
    else:
        # Matrix Figure
        fig, axes = plt.subplots(7, 3, figsize=(12, 18), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
        fig2, axes2 = plt.subplots(7, 2, figsize=(8, 18), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
        for i in range(len(ALL_CASES)):
            rans_loc = np.argmin(np.abs(np.array(RANS_CASES)-ALL_CASES[i]))
            les_loc = np.argmin(np.abs(np.array(LES_CASES)-ALL_CASES[i]))
            rans_plot_position = axes[int(np.floor(rans_loc / 3)), int(rans_loc % 3)]
            les_plot_position = axes2[les_loc, :]
            contour = draw(i, rans_plot_position, les_plot_position, ALL_CASES)
        for i in range(7):
            normal = 'z' if SURFACE[0] == 'y' else 'y'
            axes[i, 0].set_ylabel(rf'${normal}$')
            axes2[i, 0].set_ylabel(rf'${normal}$')
        for i in range(3):
            axes[-1, i].set_xlabel(r'$x$')
        axes2[-1, 0].set_xlabel(r'$x$')
        u_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        cbar1 = fig.colorbar(contour, ax=axes[-1, :], location='bottom', shrink=0.9, aspect=75, ticks=u_ticks, format='%.1f')
        cbar2 = fig.colorbar(contour, ax=axes2[-1, :], location='bottom', shrink=0.7, aspect=50, ticks=u_ticks, format='%.1f')
        cbar1.set_label(r'$U_{mag}/U_0$')
        cbar2.set_label(r'$u_{mag}/U_0$')

        plt.show()
        fig.savefig(f'figures/velocityslices-{SURFACE}-RANS-300.tiff', bbox_inches='tight', dpi=400)
        fig2.savefig(f'figures/velocityslices-{SURFACE}-LES-300.tiff', bbox_inches='tight', dpi=400)


if __name__ == "__main__":
    main()
