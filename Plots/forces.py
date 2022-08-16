#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import fonts
import glob
import argparse
import matplotlib.pyplot as plt


def get_forces(case, cube):
    files = glob.glob(f'../Data/{case}/postProcessing/{cube}-forces/*/coefficient.dat')
    file = files[0]
    data = pd.DataFrame()
    for file in files:
        with open(file) as f:
            line = f.readline()
            cnt = 0
            while line.startswith('#'):
                prev_line = line
                line = f.readline()
                cnt += 1
                # print(prev_line)
        header = prev_line.strip().lstrip('# ').split()
        new_data = pd.read_csv(file, comment='#', sep=r'\s+', names=header, header=None, engine='python')
        data = data.append(new_data, ignore_index=True)
    data.sort_values(by='Time')
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
    data1.plot(x='Time', y='Cl', ax=ax, style='r--')
    data2.plot(x='Time', y='Cl', ax=ax, style='b--')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Force Coefficient')
    ax.set_title(case)
    ax.set_ylim(0, 2)
    ax.set_xlim(xmin=0)
    ax.legend(["cube1", "cube2"], frameon=False, ncol=1, loc='upper right')


def main():
    fig1, axes1 = plt.subplots(1, 1, figsize=(6, 4), squeeze=False, constrained_layout=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='Case name', default='2Dtest')
    args = parser.parse_args()
    plot_forces(args.case, axes1[0, 0])
    plt.show()
    fig1.savefig(f'figures/force_{args.case}.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
