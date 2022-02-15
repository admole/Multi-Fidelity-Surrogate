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


def collect_profiles(model, ax, x):
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case = json.load(j)
    numcases = len(case)
    path = f"{model}/Yaw/"
    for ci in range(numcases):
        file = os.path.join(path, case[ci]["Name"])
        line = fields.get_line(file, position=x, field='U')
        case[ci]['y'] = line[:, 0]
        case[ci]['UMean'] = line[:, -3]
    print(f'Collected {numcases} profiles from {model}')
    return case, numcases


fig1, axes1 = plt.subplots(1, 1, figsize=(5, 10),
                           squeeze=False, constrained_layout=True)


RANS_Profiles, RANS_ncases = collect_profiles('RANS', axes1[0, 0], 4.0)
LES_Profiles, LES_ncases = collect_profiles('LES', axes1[0, 0], 4.0)
sample_angle = 25


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


rans_velocity_old = np.zeros(RANS_ncases)
les_velocity_old = np.zeros(4)
mf_old = np.zeros(1000)
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

    les_velocity_train = les_velocity[les_alpha % 10 == 0]
    les_velocity_test = les_velocity[les_alpha % 10 != 0]
    les_alpha_train = les_alpha[les_alpha % 10 == 0]
    les_alpha_test = les_alpha[les_alpha % 10 != 0]

    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = mfr.mfmlp(rans_alpha,
                                                                               rans_velocity,
                                                                               rans_velocity_old,
                                                                               les_alpha_train,
                                                                               les_velocity_train,
                                                                               les_velocity_old,
                                                                               mf_old)
    angle_location = np.abs(alpha - sample_angle).argmin()
    new_profile[yi] = mf_mean[angle_location]
    print(f'Actual alpha = {alpha[angle_location]}')
    rans_velocity_old = rans_velocity
    les_velocity_old = les_velocity_train
    mf_old = mf_mean


print(f'Plotting mfr profile at angle {alpha[angle_location]}')

axes1[0][0].plot(RANS_Profiles[5]['UMean'], RANS_Profiles[10]['y'], 'r--')
axes1[0][0].plot(RANS_Profiles[7]['UMean'], RANS_Profiles[12]['y'], 'r', label='RANS profile')
axes1[0][0].plot(RANS_Profiles[8]['UMean'], RANS_Profiles[13]['y'], 'r')
axes1[0][0].plot(RANS_Profiles[10]['UMean'], RANS_Profiles[15]['y'], 'r--')

axes1[0][0].plot(LES_Profiles[2]['UMean'], LES_Profiles[4]['y'], 'b--')
axes1[0][0].plot(LES_Profiles[3]['UMean'], LES_Profiles[5]['y'], 'b', label='LES profile')
axes1[0][0].plot(LES_Profiles[4]['UMean'], LES_Profiles[6]['y'], 'b--')

axes1[0, 0].plot(new_profile, RANS_Profiles[1]['y'], 'k', label='MF-MLP profile')

axes1[0, 0].set_xlabel(r'$U/U_0$')
axes1[0, 0].set_ylabel(r'$y/H$')
axes1[0, 0].legend(frameon=False, ncol=1)

plt.show()
