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


class Profiles:
    def __init__(self, model, x):
        self.model = model
        self.x = x
        self.path = f"{self.model}/Yaw/"
        self.json_file = open(os.path.join(os.getcwd(), f"../Data/{self.path}inlet_sweep.json"))
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
        for i, y, u in zip(range(self.n_cases), self.y, self.u):
            self.u_interp.append(np.interp(interp_to, self.y[i], self.u[i]))

    def training_split(self):
        alpha_train = []
        velocity_train = []
        for i in range(len(self.alphas)):
            if self.alphas[i] % 10 == 0:
                alpha_train.append(self.alphas[i])
                velocity_train.append(self.u_interp[i])
        return alpha_train, velocity_train


def plot_profile(sample_location, ax):
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
                        np.array(les_training_velocity))

    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp()

    angle_location = np.abs(alpha - sample_angle).argmin()
    new_profile = mf_mean[angle_location]
    print(f'Plotting mfr profile at angle {alpha[angle_location]}')

    rans_loc_f = int(np.floor(sample_angle/2))
    rans_loc_c = int(np.ceil(sample_angle/2))

    rans_plot, = ax.plot(rans_profiles.u[rans_loc_f]+sample_location, rans_profiles.y[rans_loc_f], 'r', label=fr'RANS Profile at ${sample_angle}^\circ$')
    ax.plot(rans_profiles.u[rans_loc_c]+sample_location, rans_profiles.y[rans_loc_c], 'r')
    rans_fill = ax.fill_betweenx(rans_profiles.y[0],
                                 rans_profiles.u_interp[rans_loc_f-2]+sample_location,
                                 rans_profiles.u_interp[rans_loc_c+2]+sample_location,
                                 alpha=0.2, color='r', label=r"RANS $\pm 4^{\circ}$")

    les_loc_f = int(np.floor(sample_angle/5))
    les_loc_c = int(np.ceil(sample_angle/5))

    les_plot, = ax.plot(les_profiles.u[les_loc_f]+sample_location, les_profiles.y[les_loc_f], 'b', label=fr'LES Profile at ${sample_angle}^\circ$')
    ax.plot(les_profiles.u[les_loc_c]+sample_location, les_profiles.y[les_loc_c], 'b')
    les_fill = ax.fill_betweenx(rans_profiles.y[0],
                                les_profiles.u_interp[les_loc_f-1]+sample_location,
                                les_profiles.u_interp[les_loc_c+1]+sample_location,
                                alpha=0.2, color='b', label=r"LES $\pm 5^{\circ}$")

    mf_plot, = ax.plot(new_profile+sample_location, rans_profiles.y[0], 'k', label=fr'MF-MLP Profile at ${sample_angle}^\circ$')

    return rans_plot, rans_fill, les_plot, les_fill, mf_plot


sample_angle = 5
fig1, axes1 = plt.subplots(1, 1, figsize=(11, 3),
                           squeeze=False, constrained_layout=True)

for sample_location in np.arange(0, 14, 1):
    print(f'\nProfile at x = {sample_location}')
    rans_plot, rans_fill, les_plot, les_fill, mf_plot = plot_profile(sample_location, axes1[0, 0])

cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='///')
cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='///')
axes1[0, 0].add_patch(cube1)
axes1[0, 0].add_patch(cube2)
axes1[0, 0].set_ylim(0, 2)
axes1[0, 0].set_xlim(0, 15)
axes1[0, 0].set_xlabel(r'$U/U_0$')
axes1[0, 0].set_ylabel(r'$y/H$')
# axes1[0, 0].set_title(fr'$\alpha = {sample_angle} $', fontsize=40)
axes1[0, 0].legend(handles=[rans_plot, rans_fill, les_plot, les_fill, mf_plot],
                   frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0.05, 1.01))
axes1[0, 0].set_aspect('equal', adjustable='box')

plt.show()
