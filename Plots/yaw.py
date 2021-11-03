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


# Opening JSON file
j = open(os.path.join(os.getcwd(), "../Cases/inlet_sweep.json"))
case_settings = json.load(j)
# case_settings = [{"Name": "test11"}]
numcases = len(case_settings)
path = "RANS/Yaw/"


fig1, axes1 = plt.subplots(1, 1, figsize=(6, 3), squeeze=False, constrained_layout=True, sharex=True, sharey=True)


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

axes1[0, 0].plot(alpha, cd[0], 'k.', label='cube1')
axes1[0, 0].plot(alpha, cd[1], 'k+', label='cube2')
axes1[0, 0].set_xlabel(r'$\alpha$')
axes1[0, 0].set_ylabel(r'Cd')
axes1[0, 0].set_ylim(bottom=0)
axes1[0, 0].set_xlim(left=0)
axes1[0, 0].legend(frameon=False, ncol=1)

plt.show()
# fig1.savefig('figures/Ubulk_time.pdf', bbox_inches='tight')
