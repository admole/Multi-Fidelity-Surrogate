#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
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
    
    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp()
    return alpha, rans_slices, les_slices, mf_mean, rans_slices.grid[0]


def draw(it, ax, angles, hf, lf, mf, grid):
    angle = angles[int(it*10)]
    print(f'Plotting mfr slice at angle {angle}')

    hf_loc = np.argmin(np.abs(hf.alphas-angle))
    lf_loc = np.argmin(np.abs(lf.alphas-angle))

    hf_plot = ax[0, 0].tricontourf(grid[0], grid[1], hf.u_interp[hf_loc],
                                   cmap='seismic', levels=np.arange(-1, 1.5, 0.05))
    mf_plot = ax[1, 0].tricontourf(grid[0], grid[1], mf[int(it*10)],
                                   cmap='seismic', levels=np.arange(-1, 1.5, 0.05))

    # hf.u_interp[hf_loc], grid
    # lf.u_interp[lf_loc], grid
    # mf[int(it*10)], grid

    for a in ax[:, 0]:
        # a.clear()
        fields.add_cubes(a, 'y')
        a.set_xlabel(r'$x/H$')
        a.set_ylabel(r'$z/H$')
        # a.set_title(fr'$\angle = {sample_angle} $', fontsize=40)
        a.set_aspect('equal', adjustable='box')

    return mf_plot


def main():
    fig1, axes1 = plt.subplots(2, 1, figsize=(11, 7), squeeze=False, constrained_layout=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Create animation')
    parser.add_argument('angle', type=float, help='Angle to sample slices at', default=15, nargs='?')
    args = parser.parse_args()
    sample_angle = args.angle

    plane = 'yNormal'

    print(f'\nSlice {plane}')
    alpha, lf, hf, mf, grid = regress_slice(plane)


    if args.a:
        anim = animation.FuncAnimation(fig1, draw, fargs=(axes1, alpha, hf, lf, mf, grid,), frames=len(alpha), interval=1, blit=False)
        plt.show()
        anim.save('animations/slices_animation.mp4', fps=5, dpi=400)
    else:
        draw(sample_angle * len(alpha) / (10 * alpha[-1]), axes1, alpha, hf, lf, mf, grid, )
        plt.show()
        fig1.savefig(f'figures/slices_{sample_angle}.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
