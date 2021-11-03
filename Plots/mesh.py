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
j = open(os.path.join(os.getcwd(), "../Cases/mesh.json"))
case_settings = json.load(j)
# case_settings = [{"Name": "test11"}]
numcases = len(case_settings)
path = "RANS/Mesh/"


fig1, axes1 = plt.subplots(1, numcases, figsize=(numcases*5, 3.5),
                           squeeze=False, constrained_layout=True, sharex=True, sharey=True)

fig2, axes2 = plt.subplots(1, numcases, figsize=(numcases*5, 3.5),
                           squeeze=False, constrained_layout=True, sharex=True, sharey=True)

for i in range(numcases):
    case = os.path.join(path, case_settings[i]["Name"])
    forces.plot_forces(case, axes1[0, i])
    cf.plot_cf(case, axes2[0, i])

plt.show()
# fig1.savefig('figures/Ubulk_time.pdf', bbox_inches='tight')
