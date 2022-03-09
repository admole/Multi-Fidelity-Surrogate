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
import fields
import cf
from mfRegression import MFRegress


def get_yaw(model):
    # Opening JSON file
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case_settings = json.load(j)
    # case_settings = [{"Name": "test11"}]
    numcases = len(case_settings)
    path = f"{model}/Yaw/"
    cd1 = cd2 = np.zeros(numcases)
    cd2 = np.zeros(numcases)
    cz1 = np.zeros(numcases)
    cz2 = np.zeros(numcases)
    recirc = np.zeros(numcases)
    probe = np.zeros(numcases)
    alpha = np.zeros(numcases)
    for i in range(numcases):
        case = os.path.join(path, case_settings[i]["Name"])
        alpha[i] = case_settings[i]["FlowAngle"]
        cd1[i] = forces.get_cd(case, 'cube1')
        cz1[i] = forces.get_cl(case, 'cube1')
        cd2[i] = forces.get_cd(case, 'cube2')
        cz2[i] = forces.get_cl(case, 'cube2')
        if model == 'remove':
            recirc[i] = cf.recirculation(case)
        probe[i] = fields.get_probe(case, position=[6.0, 0.6], field='U')

    data = {r'$\alpha$': alpha,
            r'$Cd_1$': cd1, r'$Cz_1$': cz1,
            r'$Cd_2$': cd2, r'$Cz_2$': cz2,
            r'$A_{recirc}$': recirc,
            'Probe': probe}
    df = pd.DataFrame(data=data)
    return df


def plot_yaw(ax, ax2, rans, les, variable):
    les_train = les[les[r"$\alpha$"] % 10 == 0]
    les_test = les[les[r"$\alpha$"] % 10 != 0]
    ax.scatter(rans[r'$\alpha$'], rans[variable], edgecolors='b', facecolors='none', label=f'RANS Sample')

    regress = MFRegress(rans[r'$\alpha$'].to_numpy(),
                        rans[variable].to_numpy(),
                        les_train[r'$\alpha$'].to_numpy(),
                        les_train[variable].to_numpy())
    alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp()
    ax.plot(alpha, rans_mean, 'b--', label='RANS Only GPR')
    ax.fill_between(alpha[:, 0], rans_mean[:, 0] - rans_std, rans_mean[:, 0] + rans_std, alpha=0.2, color='b')
    ax.scatter(les_test[r'$\alpha$'], les_test[variable], c='r', label='LES Sample (testing)')
    ax.scatter(les_train[r'$\alpha$'], les_train[variable], edgecolors='r', facecolors='none', label=f'LES Sample (training)')
    ax.plot(alpha, les_mean, 'r--', label='LES Only GPR')
    ax.fill_between(alpha[:, 0], les_mean[:, 0] - les_std, les_mean[:, 0] + les_std, alpha=0.2, color='r')
    ax.plot(alpha, mf_mean, 'k', label='Multi-fidelity GPR')
    ax.fill_between(alpha[:, 0], mf_mean[:, 0] - mf_std, mf_mean[:, 0] + mf_std,
                    alpha=0.2, color='k')# , label="Model +/- 1 std")

    if ax == axes1[-1, 0]:
        ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(variable)
    ax.set_xlim(0, max(alpha))
    ax.legend(frameon=False, ncol=2)

    ax2.plot(rans_mean, les_mean, 'r', label='LES Only')
    ax2.plot(rans_mean, mf_mean, 'k', label='Multi-fidelity')
    if ax2 == axes1[-1, 1]:
        ax2.set_xlabel(f'Low-Fidelity')
    ax2.set_ylabel(f'High-Fidelity')
    ax2.legend(frameon=False)


def plot_profile(ax, rans):
    for alpha in rans[r'$\alpha$']:
        y = rans['profile'][rans[r'$\alpha$'] == alpha][0]
        u = rans['profile'][rans[r'$\alpha$'] == alpha][1]
        ax.plot(alpha+u, y)


variables = [r'$Cd_1$', r'$Cd_2$', r'$A_{recirc}$']
variables = [r'$Cd_1$', r'$Cd_2$']
variables = [r'$Cd_1$', r'$Cz_1$', r'$Cd_2$', r'$Cz_2$']
variables = [r'$Cd_2$', r'$Cz_2$']
variables = [r'$Cd_2$', r'$Cz_2$', 'Probe']

fig1, axes1 = plt.subplots(len(variables), 2, figsize=(15, 3*len(variables)),
                           squeeze=False, constrained_layout=True, sharex='col', sharey='none',
                           gridspec_kw={'width_ratios': [2, 1]})
RANS_data = get_yaw('RANS')
LES_data = get_yaw('LES')
for it in range(len(variables)):
    quantity = variables[it]
    plot_yaw(axes1[it, 0], axes1[it, 1], RANS_data, LES_data, quantity)

# fig2, axes2 = plt.subplots(1, 1, figsize=(12, 10), squeeze=False, constrained_layout=True)
# plot_profile(axes2[0, 0], RANS_data)

plt.show()
fig1.savefig('yaw.pdf', bbox_inches='tight')
