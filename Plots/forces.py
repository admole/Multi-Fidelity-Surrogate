#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import fonts


def get_forces(case, cube):
    cols = ['time', 'Cm', 'Cd', 'Cl', 'Clf', 'Clr']
    data = pd.read_csv(f'../Data/{case}/postProcessing/{cube}-forces/0/coefficient.dat',
                       skiprows=10, sep=r'\s+', names=cols, header=None, engine='python')

    return data


def get_cd(case, cube):
    forces = get_forces(case, cube)
    cd = np.mean(forces['Cd'].tail(500))
    return cd


def plot_forces(case, ax):
    data1 = get_forces(case, 'cube1')
    data2 = get_forces(case, 'cube2')
    # Plotting Cd
    data1.plot(x='time', y='Cd', ax=ax, style='r')
    data2.plot(x='time', y='Cd', ax=ax, style='b')
    ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'Cd')
    ax.set_title(case)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 2000)
    ax.legend(["cube1", "cube2"], frameon=False, ncol=1, loc='upper right')

