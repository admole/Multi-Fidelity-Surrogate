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
                        np.array(les_training_velocity),
                        embedding_theory=True,)
    
    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp()
    
    return mf_mean, rans_profiles.y[0]


def draw(it, angle, mf_mean, y):
    axes1[0, 0].clear()

    cube1 = patches.Rectangle((2, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='///')
    cube2 = patches.Rectangle((7, 0), 1, 1, linewidth=1, edgecolor='k', fc='lightgrey', hatch='///')
    axes1[0, 0].add_patch(cube1)
    axes1[0, 0].add_patch(cube2)
    axes1[0, 0].set_ylim(0, 2)
    axes1[0, 0].set_xlim(0, 15)
    axes1[0, 0].set_xlabel(r'$U/U_0$')
    axes1[0, 0].set_ylabel(r'$y/H$')
    # axes1[0, 0].set_title(fr'$\angle = {sample_angle} $', fontsize=40)
    axes1[0, 0].set_aspect('equal', adjustable='box')
    
    print(f'Plotting mfr profile at angle {angle[it*10]}')
    for sample_loc, i in zip(np.arange(0, 14, 1), range(len(mf_mean))):
        mf_plot, = axes1[0, 0].plot(mf_mean[i][it*10]+sample_loc, y[i], 'k', label=fr'MF-MLP Profile at ${int(angle[it*10])}^\circ$')
    
    axes1[0, 0].legend(handles=[mf_plot], frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0.05, 1.01))

    return mf_plot


sample_angle = 15
fig1, axes1 = plt.subplots(1, 1, figsize=(11, 3),
                           squeeze=False, constrained_layout=True)
mfs = []
ys = []
alpha = np.linspace(0, 40, 1000)

for sample_location in np.arange(0, 14, 1):
    print(f'\nProfile at x = {sample_location}')
    mf, y = plot_profile(sample_location, axes1[0, 0])
    mfs.append(mf)
    ys.append(y)
    
anim = animation.FuncAnimation(fig1, draw, fargs=(alpha, mfs, ys,), frames=100, interval=1, blit=False)
# plt.show()
anim.save('animations/profiles_animation.mp4', fps=10, dpi=400)
