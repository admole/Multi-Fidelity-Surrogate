#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import fonts
import glob


def get_forces(case, cube):
    file = glob.glob(f'../Data/{case}/postProcessing/{cube}-forces/*/coefficient.dat')
    file = file[0]
    with open(file) as f:
        line = f.readline()
        cnt = 0
        while line.startswith('#'):
            prev_line = line
            line = f.readline()
            cnt += 1
            # print(prev_line)
    header = prev_line.strip().lstrip('# ').split()
    data = pd.read_csv(file, comment='#', sep=r'\s+', names=header, header=None, engine='python')
    return data


def get_cd(case, cube):
    forces = get_forces(case, cube)
    cd = np.mean(forces['Cd'].tail(500))
    return cd


def get_cl(case, cube):
    forces = get_forces(case, cube)
    cl = np.mean(forces['Cl'].tail(500))
    return cl


def plot_forces(case, ax):
    data1 = get_forces(case, 'cube1')
    data2 = get_forces(case, 'cube2')
    # Plotting Cd
    data1.plot(x='Time', y='Cd', ax=ax, style='r')
    data2.plot(x='Time', y='Cd', ax=ax, style='b')
    ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'Cd')
    ax.set_title(case)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 2000)
    ax.legend(["cube1", "cube2"], frameon=False, ncol=1, loc='upper right')

