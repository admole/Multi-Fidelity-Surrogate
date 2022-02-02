#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import fonts
import json
import fields
import mfRegression as mfr


def collect_profiles(model, ax):
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case = json.load(j)
    numcases = len(case)
    path = f"{model}/Yaw/"
    for i in range(numcases):
        file = os.path.join(path, case[i]["Name"])
        line = fields.get_line(file, position=4.0, field='U')
        case[i]['y'] = line[:, 0]
        case[i]['UMean'] = line[:, -3]
        # if model == 'RANS':
        #     ax.plot(case[i]['UMean'], case[i]['y'], 'r')
        # else:
        #     ax.plot(case[i]['UMean'], case[i]['y'], 'b')
    print(f'Collected {numcases} profiles from {model}')
    return case, numcases


fig1, axes1 = plt.subplots(1, 1, figsize=(5, 10),
                           squeeze=False, constrained_layout=True)


RANS_Profiles, RANS_ncases = collect_profiles('RANS', axes1[0, 0])
LES_Profiles, LES_ncases = collect_profiles('LES', axes1[0, 0])

# add field for LES result interpolated onto LES mesh
les_alpha = np.zeros(LES_ncases)
for i, case in zip(range(LES_ncases), LES_Profiles):
    les_alpha[i] = case['FlowAngle']
    case['UMean_interp'] = np.interp(RANS_Profiles[1]['y'],  # Using RANS y values not at alpha 0 as this case differs
                                     case['y'],              # due to the mesh and needs to be updated
                                     case['UMean'])

rans_alpha = np.zeros(RANS_ncases)
for i, case in zip(range(RANS_ncases), RANS_Profiles):
    rans_alpha[i] = case['FlowAngle']


Nlocs = len(RANS_Profiles[1]['y'])
new_profile = np.zeros(Nlocs)
print(Nlocs)
for yi in range(Nlocs):
    rans_velocity = np.zeros(RANS_ncases)
    for i, case in zip(range(RANS_ncases), RANS_Profiles):
        rans_velocity[i] = case['UMean'][yi]
    les_velocity = np.zeros(LES_ncases)
    for i, case in zip(range(LES_ncases), LES_Profiles):
        les_velocity[i] = case['UMean_interp'][yi]

    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = mfr.mfmlp(rans_alpha,
                                                                               rans_velocity,
                                                                               les_alpha,
                                                                               les_velocity)
    sample_angle = 4
    angle_location = np.abs(alpha - sample_angle).argmin()
    new_profile[yi] = mf_mean[angle_location]
    print(f'Actual alpha = {alpha[angle_location]}')


print(f'Plotting mfr profile at angle {alpha[angle_location]}')

axes1[0][0].plot(RANS_Profiles[2]['UMean'], RANS_Profiles[2]['y'], 'r')
axes1[0][0].plot(LES_Profiles[1]['UMean'], LES_Profiles[1]['y'], 'b')
axes1[0][0].plot(LES_Profiles[1]['UMean_interp'], RANS_Profiles[1]['y'], 'b--')
axes1[0, 0].plot(new_profile, RANS_Profiles[1]['y'], 'k')
plt.show()
