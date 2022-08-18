#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import re
import fonts
import forces
import cf


def get_ncells(c):
    file = f'../Data/{c}/log.checkMesh'
    regexp = re.compile(r'^\s*cells.*?(\d+)')
    n = 0
    with open(file) as f:
        for line in f:
            match = regexp.match(line)
            if match:
                n = float(match.group(1))
                print(f'Retrieved number of cells = {n}')
    return n


# Opening JSON file
j = open(os.path.join(os.getcwd(), "../Cases/mesh.json"))
case_settings = json.load(j)
# case_settings = [{"Name": "test11"}]
numcases = len(case_settings)
path = "RANS/Mesh/"


fig1, axes1 = plt.subplots(1, numcases, figsize=(numcases*5, 3.5),
                           squeeze=False, constrained_layout=True, sharex=True, sharey=True)

fig2, axes2 = plt.subplots(1, numcases, figsize=(numcases*5, 3.5),
                           squeeze=False, constrained_layout=True, sharex=True, sharey=True)

fig3, axes3 = plt.subplots(1, 1, figsize=(5, 3),
                           squeeze=False, constrained_layout=True, sharex=True, sharey=True)


ncells = np.zeros(numcases)
cd1 = np.zeros(numcases)
cd2 = np.zeros(numcases)
for i in range(numcases):
    case = os.path.join(path, case_settings[i]["Name"])
    forces.plot_forces(case, axes1[0, i])
    cf.plot_cf(case, axes2[0, i])
    ncells[i] = get_ncells(case)
    cd1[i] = forces.get_cd(case, 'cube1')
    cd2[i] = forces.get_cd(case, 'cube2')
    for ax in [axes1, axes2]:
        ax[0, i].set_title(fr'$N_c \approx {int(ncells[i])/1000000:.1f} \times 10 ^6$')

axes3[0, 0].plot(ncells, cd1, color='k', marker='o', mfc='r', mec='r', ms=10, label='cube1')
axes3[0, 0].plot(ncells, cd2, color='k', marker='o', mfc='b', mec='b', ms=10, label='cube2')
axes3[0, 0].set_xlabel('Number of cells')
axes3[0, 0].set_ylabel('Cd')
axes3[0, 0].set_ylim(bottom=0)
axes3[0, 0].legend(frameon=False)

plt.show()
fig2.savefig('figures/meshCf.pdf', bbox_inches='tight')
fig3.savefig('figures/mesh.pdf', bbox_inches='tight')
