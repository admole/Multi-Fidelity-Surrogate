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


class Profiles:
    def __init__(self, model, x):
        self.model = model
        self.x = x
        self.path = f"{self.model}/Yaw/"
        self.json_file = open(os.path.join(os.getcwd(), f"../Data/{self.path}inlet_sweep_extra.json"))
        self.case = json.load(self.json_file)
        self.n_cases = len(self.case)
        self.alphas = []
        self.y = []
        self.u = []
        self.u_interp = []

    def collect_profiles(self):
        for ci in range(self.n_cases):
            file = os.path.join(self.path, self.case[ci]["Name"])
            line = fields.get_line(file, position=self.x, field='UMean')
            self.alphas.append(self.case[ci]['FlowAngle'])
            self.y.append(line[:, 0])
            self.u.append(line[:, -3])
        print(f'Collected {self.n_cases} profiles from {self.model}')

    def interpolate_velocity(self, interp_to):
        for i in range(self.n_cases):
            self.u_interp.append(np.interp(interp_to, self.y[i], self.u[i]))

    def training_split(self):
        alpha_train = []
        velocity_train = []
        for i in range(len(self.alphas)):
            if self.alphas[i] % 10 == 0:
                alpha_train.append(self.alphas[i])
                velocity_train.append(self.u_interp[i])
        return alpha_train, velocity_train


def regress_profile(sample_location, gpr):
    rans_profiles = Profiles('RANS', sample_location)
    rans_profiles.collect_profiles()
    les_profiles = Profiles('LES', sample_location)
    les_profiles.collect_profiles()

    rans_profiles.interpolate_velocity(interp_to=rans_profiles.y[0])
    les_profiles.interpolate_velocity(interp_to=rans_profiles.y[0])

    les_training_alpha, les_training_velocity = les_profiles.training_split()

    regress = MFRegress(np.array(rans_profiles.alphas),
                        np.array(rans_profiles.u_interp),
                        np.array(les_training_alpha),
                        np.array(les_training_velocity),
                        embedding_theory=True,)

    if gpr:
        alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfgp()
    else:
        alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp()

    return alpha, rans_profiles, les_profiles, mf_mean, rans_profiles.y[0]


def draw(it, ax, angles, hf, lf, mf, y, model):
    angle = angles[int(it*10)]
    ax[0, 0].clear()

    cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='///')
    cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='///')
    ax[0, 0].add_patch(cube1)
    ax[0, 0].add_patch(cube2)
    ax[0, 0].set_ylim(0, 2)
    ax[0, 0].set_xlim(0, 15)
    ax[0, 0].set_xlabel(r'$U/U_0$')
    ax[0, 0].set_ylabel(r'$y/H$')
    # ax[0, 0].set_title(fr'$\angle = {sample_angle} $', fontsize=40)
    ax[0, 0].set_aspect('equal', adjustable='box')
    
    print(f'Plotting mfr profile at angle {angle}')

    hf_loc = np.argmin(np.abs(hf[0].alphas-angle))
    lf_loc = np.argmin(np.abs(lf[0].alphas-angle))

    for sample_loc, i in zip(np.arange(0, 14, 1), range(len(mf))):
        hf_plot, = ax[0, 0].plot(hf[i].u_interp[hf_loc]+sample_loc, y[i], 'b',
                                    label=fr'LES Profile at ${int(hf[i].alphas[hf_loc])}^\circ$')
        lf_plot, = ax[0, 0].plot(lf[i].u_interp[lf_loc]+sample_loc, y[i], 'r',
                                    label=fr'RANS Profile at ${int(lf[i].alphas[lf_loc])}^\circ$')
        mf_plot, = ax[0, 0].plot(mf[i][int(it*10)]+sample_loc, y[i], 'k',
                                    label=fr'MF-{model} Profile at ${int(angle)}^\circ$')
    
    ax[0, 0].legend(handles=[mf_plot, hf_plot, lf_plot],
                       frameon=False, ncol=3, loc='lower left',
                       bbox_to_anchor=(0.05, 1.01))

    return mf_plot


def main():
    fig1, axes1 = plt.subplots(1, 1, figsize=(11, 3), squeeze=False, constrained_layout=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Create animation')
    parser.add_argument('-gpr', action='store_true', help='Use GPR (default is MLP)')
    parser.add_argument('angle', type=float, help='Angle to sample profiles at', default=15, nargs='?')
    args = parser.parse_args()
    sample_angle = args.angle
    if args.gpr:
        model = 'GPR'
    else:
        model = 'MLP'

    lfs = []
    hfs = []
    mfs = []
    ys = []

    for sample_location in np.arange(0, 14, 1):
        print(f'\nProfile at x = {sample_location}')
        alpha, lf, hf, mf, y = regress_profile(sample_location, args.gpr)
        lfs.append(lf)
        hfs.append(hf)
        mfs.append(mf)
        ys.append(y)

    if args.a:
        anim = animation.FuncAnimation(fig1, draw, fargs=(axes1, alpha, hfs, lfs, mfs, ys, model), frames=int(len(alpha)/10), blit=False)
        plt.show()
        anim.save(f'animations/profiles_{model}_animation.mp4', fps=5, dpi=400)
    else:
        draw(sample_angle * len(alpha) / (10 * alpha[-1]), axes1, alpha, hfs, lfs, mfs, ys, model)
        plt.show()
        fig1.savefig(f'figures/profiles_{model}_{int(sample_angle)}.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
