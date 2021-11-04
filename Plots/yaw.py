#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import fonts
import json
import forces
import cf
import mfGP


def get_yaw(model):
    # Opening JSON file
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case_settings = json.load(j)
    # case_settings = [{"Name": "test11"}]
    numcases = len(case_settings)
    path = f"{model}/Yaw/"
    cd1 = np.zeros(numcases)
    cd2 = np.zeros(numcases)
    # recirc = np.zeros(numcases)
    alpha = np.zeros(numcases)
    for i in range(numcases):
        case = os.path.join(path, case_settings[i]["Name"])
        alpha[i] = case_settings[i]["FlowAngle"]
        cd1[i] = forces.get_cd(case, 'cube1')
        cd2[i] = forces.get_cd(case, 'cube2')
        # recirc[i] = cf.recirculation(case)
        # print(recirc[i])
    data = {r'$\alpha$': alpha, r'$Cd_1$': cd1, r'$Cd_2$': cd2}
    df = pd.DataFrame(data=data)
    return df


def plot_yaw(ax, rans, les, variable):
    ax.plot(rans[r'$\alpha$'], rans[variable], 'r.', markersize=12, label=f'RANS Sample')
    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = mfGP.mfgp(rans[r'$\alpha$'].to_numpy(), rans[variable].to_numpy(),
                                                                               les[r'$\alpha$'].to_numpy(), les[variable].to_numpy())
    ax.plot(alpha, rans_mean, 'r', label='RANS GP Mean')
    ax.fill_between(alpha[:, 0], rans_mean[:, 0] - rans_std, rans_mean[:, 0] + rans_std, alpha=0.2, color='r')
    ax.plot(les[r'$\alpha$'], les[variable], 'b.', markersize=12, label=f'LES Sample')
    ax.plot(alpha, les_mean, 'b', label='LES GP Mean')
    ax.fill_between(alpha[:, 0], les_mean[:, 0] - les_std, les_mean[:, 0] + les_std, alpha=0.2, color='b')
    ax.plot(alpha, mf_mean, 'k', label='Multi-fidelity GP Mean')
    ax.fill_between(alpha[:, 0], mf_mean[:, 0] - mf_std, mf_mean[:, 0] + mf_std,
                    alpha=0.2, color='k', label="Model +/- 1 std")

    if ax == axes1[-1, 0]:
        ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(variable)
    ax.set_ylim(0.9*min(mf_mean), 1.1*max(mf_mean))
    ax.set_xlim(0, max(alpha))
    ax.legend(frameon=False, ncol=3)


variables = [r'$Cd_1$', r'$Cd_2$']
fig1, axes1 = plt.subplots(len(variables), 1, figsize=(10, 3*len(variables)),
                           squeeze=False, constrained_layout=True, sharex=True, sharey=False)
RANS_data = get_yaw('RANS')
LES_data = get_yaw('LES')
LES_data = LES_data[LES_data[r"$\alpha$"] % 10 != 0]
for it in range(len(variables)):
    quantity = variables[it]
    plot_yaw(axes1[it, 0], RANS_data, LES_data, quantity)

plt.show()
# fig1.savefig('figures/Ubulk_time.pdf', bbox_inches='tight')
