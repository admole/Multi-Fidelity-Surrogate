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


def plot_yaw(model, ax, col):
    # Opening JSON file
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case_settings = json.load(j)
    # case_settings = [{"Name": "test11"}]
    numcases = len(case_settings)
    path = f"{model}/Yaw/"
    cd = np.zeros(shape=(2, numcases))
    recirc = np.zeros(numcases)
    alpha = np.zeros(numcases)
    for i in range(numcases):
        case = os.path.join(path, case_settings[i]["Name"])
        alpha[i] = case_settings[i]["FlowAngle"]
        cd[0, i] = forces.get_cd(case, 'cube1')
        cd[1, i] = forces.get_cd(case, 'cube2')
        # recirc[i] = cf.recirculation(case)
        # print(recirc[i])

    ax.plot(alpha, cd[0], f'{col}.', markersize=12, label=f'{model} cube1')
    ax.plot(alpha, cd[1], f'{col}+', markersize=12, label=f'{model} cube2')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'Cd')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, ncol=2)


fig1, axes1 = plt.subplots(1, 1, figsize=(6, 3), squeeze=False, constrained_layout=True, sharex=True, sharey=True)
plot_yaw("RANS", axes1[0, 0], 'r')
plot_yaw("LES", axes1[0, 0], 'b')

plt.show()
# fig1.savefig('figures/Ubulk_time.pdf', bbox_inches='tight')
