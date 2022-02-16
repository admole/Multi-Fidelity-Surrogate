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
    print(f'Collected {numcases} profiles from {model}')
    return case, numcases


sample_angle = 20
sample_location = 9.0
fig1, axes1 = plt.subplots(1, 1, figsize=(4, 6),
                           squeeze=False, constrained_layout=True)


RANS_Profiles, RANS_ncases = collect_profiles('RANS', sample_location)
LES_Profiles, LES_ncases = collect_profiles('LES', sample_location)


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
les_velocity_old = np.zeros(3)
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

    les_velocity_train = les_velocity[les_alpha % 10 != 0]
    les_velocity_test = les_velocity[les_alpha % 10 == 0]
    les_alpha_train = les_alpha[les_alpha % 10 != 0]
    les_alpha_test = les_alpha[les_alpha % 10 == 0]

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

rans_loc_f = int(np.floor(sample_angle/2))
rans_loc_c = int(np.ceil(sample_angle/2))
print(rans_loc_f)
print(rans_loc_c)
axes1[0][0].plot(RANS_Profiles[rans_loc_f]['UMean'], RANS_Profiles[rans_loc_f]['y'], 'r', label='RANS profile')
axes1[0][0].plot(RANS_Profiles[rans_loc_c]['UMean'], RANS_Profiles[rans_loc_c]['y'], 'r')
axes1[0][0].fill_betweenx(RANS_Profiles[rans_loc_f-2]['y'],
                          RANS_Profiles[rans_loc_f-2]['UMean'],
                          RANS_Profiles[rans_loc_c+2]['UMean'],
                          alpha=0.2, color='r', label="+/- 4")

les_loc_f = int(np.floor(sample_angle/5))
les_loc_c = int(np.ceil(sample_angle/5))
print(les_loc_f)
print(les_loc_c)
axes1[0][0].plot(LES_Profiles[les_loc_f]['UMean'], LES_Profiles[les_loc_f]['y'], 'b', label='LES profile')
axes1[0][0].plot(LES_Profiles[les_loc_c]['UMean'], LES_Profiles[les_loc_c]['y'], 'b')
axes1[0][0].fill_betweenx(LES_Profiles[les_loc_f-1]['y'],
                          LES_Profiles[les_loc_f-1]['UMean'],
                          LES_Profiles[les_loc_c+1]['UMean'],
                          alpha=0.2, color='b', label="+/- 5")

axes1[0, 0].plot(new_profile, RANS_Profiles[1]['y'], 'k', label='MF-MLP profile')

axes1[0, 0].set_xlabel(r'$U/U_0$')
axes1[0, 0].set_ylabel(r'$y/H$')
axes1[0, 0].legend(frameon=False, ncol=1)
axes1[0, 0].set_title(fr'$\alpha = {sample_angle} \; \; x = {sample_location}$')

plt.show()
