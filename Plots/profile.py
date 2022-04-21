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


def collect_profiles(model, x):
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case = json.load(j)
    numcases = len(case)
    path = f"{model}/Yaw/"
    for ci in range(numcases):
        file = os.path.join(path, case[ci]["Name"])
        line = fields.get_line(file, position=x, field='U')
        case[ci]['y'] = line[:, 0]
        case[ci]['UMean'] = line[:, -3]
        print(file)
        print(len(case[ci]['UMean']))
    print(f'Collected {numcases} profiles from {model}')
    return case, numcases


def plot_profile(sample_location, ax):
    RANS_Profiles, RANS_ncases = collect_profiles('RANS', sample_location)
    LES_Profiles, LES_ncases = collect_profiles('LES', sample_location)

    # add field for LES result interpolated onto RANS mesh
    les_alpha = np.zeros(LES_ncases)
    for i, case in zip(range(LES_ncases), LES_Profiles):
        les_alpha[i] = case['FlowAngle']
        case['UMean_interp'] = np.interp(RANS_Profiles[0]['y'],  # Using RANS y values not at alpha 0 (0 case differs?)
                                         case['y'],              # due to the mesh and needs to be updated
                                         case['UMean'])

    rans_alpha = np.zeros(RANS_ncases)
    for i, case in zip(range(RANS_ncases), RANS_Profiles):
        rans_alpha[i] = case['FlowAngle']
        case['UMean_interp'] = np.interp(RANS_Profiles[0]['y'],  # Using RANS y values not at alpha 0 (0 case differs?)
                                         case['y'],              # due to the mesh and needs to be updated
                                         case['UMean'])

    rans_velocity_old = np.zeros(RANS_ncases)
    les_velocity_old = np.zeros(3)
    mf_old = np.zeros(1000)
    Nlocs = len(RANS_Profiles[0]['y'])
    new_profile = np.zeros(Nlocs)
    print(Nlocs)
    for yi in range(Nlocs):
        rans_velocity = np.zeros(RANS_ncases)
        for i, case in zip(range(RANS_ncases), RANS_Profiles):
            rans_velocity[i] = case['UMean_interp'][yi]
        les_velocity = np.zeros(LES_ncases)
        for i, case in zip(range(LES_ncases), LES_Profiles):
            les_velocity[i] = case['UMean_interp'][yi]

        les_velocity_train = les_velocity[les_alpha % 10 != 0]
        les_velocity_test = les_velocity[les_alpha % 10 == 0]
        les_alpha_train = les_alpha[les_alpha % 10 != 0]
        les_alpha_test = les_alpha[les_alpha % 10 == 0]

        regress = MFRegress(rans_alpha, rans_velocity, les_alpha_train, les_velocity_train,)
        alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp2(rans_velocity_old,
                                                                                        les_velocity_old,
                                                                                        mf_old)
        angle_location = np.abs(alpha - sample_angle).argmin()
        new_profile[yi] = mf_mean[angle_location]
        print(f'Actual alpha = {alpha[angle_location]}')
        rans_velocity_old = rans_velocity
        les_velocity_old = les_velocity_train
        mf_old = mf_mean

    print(f'Plotting mfr profile at angle {alpha[angle_location]}')

    rans_loc_f = int(np.floor(sample_angle/2))
    rans_loc_c = int(np.ceil(sample_angle/2))
    print(rans_loc_f)
    print(rans_loc_c)
    rans_plot, = ax.plot(RANS_Profiles[rans_loc_f]['UMean']+sample_location, RANS_Profiles[rans_loc_f]['y'], 'r', label=fr'RANS Profile at ${sample_angle}^\circ$')
    ax.plot(RANS_Profiles[rans_loc_c]['UMean']+sample_location, RANS_Profiles[rans_loc_c]['y'], 'r')
    rans_fill = ax.fill_betweenx(RANS_Profiles[0]['y'],
                                 RANS_Profiles[rans_loc_f-2]['UMean_interp']+sample_location,
                                 RANS_Profiles[rans_loc_c+2]['UMean_interp']+sample_location,
                                 alpha=0.2, color='r', label=r"RANS $\pm 4^{\circ}$")

    les_loc_f = int(np.floor(sample_angle/5))
    les_loc_c = int(np.ceil(sample_angle/5))
    print(les_loc_f)
    print(les_loc_c)
    les_plot, = ax.plot(LES_Profiles[les_loc_f]['UMean']+sample_location, LES_Profiles[les_loc_f]['y'], 'b', label=fr'LES Profile at ${sample_angle}^\circ$')
    ax.plot(LES_Profiles[les_loc_c]['UMean']+sample_location, LES_Profiles[les_loc_c]['y'], 'b')
    les_fill = ax.fill_betweenx(RANS_Profiles[0]['y'],
                                LES_Profiles[les_loc_f-1]['UMean_interp']+sample_location,
                                LES_Profiles[les_loc_c+1]['UMean_interp']+sample_location,
                                alpha=0.2, color='b', label=r"LES $\pm 5^{\circ}$")

    mf_plot, = ax.plot(new_profile+sample_location, RANS_Profiles[0]['y'], 'k', label=fr'MF-MLP Profile at ${sample_angle}^\circ$')

    return rans_plot, rans_fill, les_plot, les_fill, mf_plot


sample_angle = 20
sample_location = 9.0
fig1, axes1 = plt.subplots(1, 1, figsize=(11, 3),
                           squeeze=False, constrained_layout=True)

for sample_location in np.arange(0, 14, 1):
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
