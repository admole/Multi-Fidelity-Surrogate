#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fonts
import json
import fields
from mfRegression import MFRegress
from matplotlib import animation
import argparse
from scipy import interpolate


class Slices:
    def __init__(self, model, plane):
        self.model = model
        self.plane = plane
        self.path = f"{self.model}/Yaw/"
        self.json_file = open(os.path.join(os.getcwd(), f"../Data/{self.path}inlet_sweep_extra.json"))
        self.case = json.load(self.json_file)
        self.n_cases = len(self.case)
        self.alphas = []
        self.grid = []
        self.u = []
        self.u_interp = []

    def collect_slices(self):
        for ci in range(self.n_cases):
            file = os.path.join(self.path, self.case[ci]["Name"])
            slice = fields.get_surface(file, surface=self.plane, field='UMean')
            self.alphas.append(self.case[ci]['FlowAngle'])
            self.grid.append((slice['x'], slice['z']))
            self.u.append(slice['UMean_x'].to_numpy())
        print(f'Collected {self.n_cases} slices from {self.model}')

    def interpolate_velocity(self, interp_to):
        for i in range(self.n_cases):
            # self.u_interp.append(np.interp(interp_to, self.grid[i], self.u[i]))
            fun = interpolate.NearestNDInterpolator(self.grid[i], self.u[i])
            self.u_interp.append(fun(interp_to[0], interp_to[1]))

    def training_split(self):
        alpha_train = []
        velocity_train = []
        for i in range(len(self.alphas)):
            if self.alphas[i] % 10 == 0:
                alpha_train.append(self.alphas[i])
                velocity_train.append(self.u_interp[i])
        return alpha_train, velocity_train


def regress_slice(sample_location,):
    rans_slices = Slices('RANS', sample_location)
    rans_slices.collect_slices()
    les_slices = Slices('LES', sample_location)
    les_slices.collect_slices()

    rans_slices.interpolate_velocity(interp_to=rans_slices.grid[0])
    les_slices.interpolate_velocity(interp_to=rans_slices.grid[0])

    les_training_alpha, les_training_velocity = les_slices.training_split()

    print(np.shape(np.array(rans_slices.u_interp)))
    print(np.shape(np.array(les_training_velocity)))

    regress = MFRegress(np.array(rans_slices.alphas),
                        np.array(rans_slices.u_interp),
                        np.array(les_training_alpha),
                        np.array(les_training_velocity),
                        embedding_theory=True,)

    from sklearn.gaussian_process.kernels import (Matern, DotProduct)
    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfgp(kernel_lf=Matern(),
                                                                                  kernel_hf=DotProduct()**2*Matern())
    return alpha, rans_slices, les_slices, mf_mean, rans_slices.grid[0]


def draw(it, fig, ax, angles, hf, lf, mf, grid, animation):
    angle = angles[int(it*10)]
    print(f'Plotting mfr slice at angle {angle}')
    for a in ax.flatten():
        a.clear()

    hf_loc = np.argmin(np.abs(hf.alphas-angle))
    lf_loc = np.argmin(np.abs(lf.alphas-angle))

    labels = [r'RANS',
              rf'LES (Test {hf.alphas[hf_loc]})',
              rf'LES (Train $-5^\circ$)',
              rf'LES (Train $+5^\circ$)',
              rf'MF-GPR']
    if any(hf.alphas[hf_loc] == [0, 10, 20, 30]):
        sty1 = 'dotted'
        sty2 = 'solid'

    lf_plot1 = ax[0, 0].tricontour(grid[0], grid[1], lf.u_interp[lf_loc],
                                   colors='r', levels=[0.5], linewidths=2)
    lf_plot1.collections[0].set_label(labels[0])

    hf_plot1 = ax[0, 0].tricontour(grid[0], grid[1], hf.u_interp[hf_loc],
                                   colors='b', levels=[0.5], linewidths=2)
    hf_plot1.collections[0].set_label(labels[1])

    if hf_loc > 0:
        hf_plot2 = ax[0, 0].tricontour(grid[0], grid[1], hf.u_interp[hf_loc-1],
                                       colors='b', levels=[0.5], linewidths=2,
                                       alpha=0.2, linestyles='dotted')
        hf_plot2.collections[0].set_label(labels[2])

    if hf_loc < len(hf.alphas - 1):
        hf_plot3 = ax[0, 0].tricontour(grid[0], grid[1], hf.u_interp[hf_loc+1],
                                       colors='b', levels=[0.5], linewidths=2,
                                       alpha=0.2, linestyles='dashed')
        hf_plot3.collections[0].set_label(labels[3])

    mf_plot1 = ax[0, 0].tricontour(grid[0], grid[1], mf[int(it*10)],
                                   colors='k', levels=[0.5], linewidths=2)
    mf_plot1.collections[0].set_label(labels[4])

    ax[0, 0].legend(loc='lower left', ncol=3, columnspacing=0.5, frameon=False)

    for ai in ax[:]:
        for a in ai[:]:
            fields.add_cubes(a, 'y')
            a.set_aspect('equal', adjustable='box')
    for a in ax[:, 0]:
        a.set_ylabel(r'$z/H$')
    for a in ax[-1, :]:
        a.set_xlabel(r'$x/H$')

    if animation:
        fig.suptitle(fr'$\theta = {int(angle)}$')

    return mf_plot1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Create animation')
    parser.add_argument('angle', type=float, help='Angle to sample slices at', default=[5, 15, 25], nargs='*')
    args = parser.parse_args()
    sample_angle = args.angle
    print(sample_angle)

    plane = 'yNormal'

    print(f'\nSlice {plane}')
    alpha, lf, hf, mf, grid = regress_slice(plane)

    if args.a:
        fig1, axes1 = plt.subplots(1, 1, figsize=(8, 6),
                                   squeeze=False, constrained_layout=True,
                                   sharex='all', sharey='all')
        anim = animation.FuncAnimation(fig1, draw, fargs=(fig1, axes1, alpha, hf, lf, mf, grid, args.a),
                                       frames=int(len(alpha)/10), interval=1, blit=False)
        anim.save('animations/slices_animation.mp4', fps=5, dpi=400)
    else:
        fig1, axes1 = plt.subplots(1, 1, figsize=(7, 6.5),
                                   squeeze=False, constrained_layout=True,
                                   sharex='all', sharey='all')
        for ang in sample_angle:
            draw(ang * len(alpha) / (10 * alpha[-1]), fig1, axes1, alpha, hf, lf, mf, grid, args.a)
            plt.show()
            fig1.savefig(f'figures/slices_{int(ang)}.svg', bbox_inches='tight')


if __name__ == "__main__":
    main()
