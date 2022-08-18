#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import fonts
import glob
import argparse
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq


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
    datasort = data.sort_values(by='Time')
    return datasort


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
    ax.plot(data1['Time'], data1['Cd'], 'r', label=r'$Cd_1$')
    ax.plot(data2['Time'], data2['Cd'], 'b', label=r'$Cd_2$')
    ax.plot(data1['Time'], data1['Cl'], 'r--', label=r'$Cz_1$')
    ax.plot(data2['Time'], data2['Cl'], 'b--', label=r'$Cz_2$')
    ax.set_ylabel(r'Force Coefficient')
    ax.set_ylim(-1, 2)
    ax.set_xlim(data1['Time'].iloc[-1]/2, data1['Time'].iloc[-1])
    # ax.legend(frameon=False, ncol=2, loc='upper center')


def plot_forces_angle(base, angle, ax):
    case = f'{base}{angle}'
    plot_forces(case, ax)
    pad = 5
    ax.annotate(fr'$\theta = {angle}^\circ$', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=28)


def plot_fft(base, angle, ax, log=False):
    case = f'{base}{angle}'
    data1 = get_forces(case, 'cube1')
    data2 = get_forces(case, 'cube2')
    n = len(data2['Cl'])

    pcd1 = fft(data1['Cd'][n // 2:])
    pcd2 = fft(data2['Cd'][n // 2:])
    pcl1 = fft(data1['Cl'][n // 2:])
    pcl2 = fft(data2['Cl'][n // 2:])
    n = n // 2
    # get the sampling rate
    ts = (data2['Time'][100] - data2['Time'][99])
    freq = np.fft.fftfreq(n, d=ts)

    # Get the one-sided specturm
    n_oneside = n // 2
    # get the one side frequency
    f_oneside = freq[1:n_oneside]
    ax.plot(f_oneside, np.abs(pcd1[1:n_oneside]), 'r', label=r'$Cd_1$')
    ax.plot(f_oneside, np.abs(pcd2[1:n_oneside]), 'b', label=r'$Cd_2$')
    ax.plot(f_oneside, np.abs(pcl1[1:n_oneside]), 'r--', label=r'$Cz_1$')
    ax.plot(f_oneside, np.abs(pcl2[1:n_oneside]), 'b--', label=r'$Cz_2$')
    ax.set_ylabel(r'PSD')
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
        x = np.array([10, 100])
        y = 10 ** 4 * x ** (-5 / 3)
        ax.plot(x, y, 'k--', label=r'$-5/3$')
    else:
        ax.set_xlim(0, 1)
    legend_location = (1, 1)
    ax.legend(frameon=False, bbox_to_anchor=legend_location, loc='upper left')
    annot_max(f_oneside, np.abs(pcl2[1:n_oneside]), ax)


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = fr"$f_p = {xmax:.3f}$"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def main():
    base = 'LES/Yaw/a'
    angles = [0, 5, 10, 15, 20, 25, 30]
    fig1, axes1 = plt.subplots(len(angles), 2, figsize=(12, 18),
                               squeeze=False, constrained_layout=True, sharex='col', sharey='col')

    for i in range(len(angles)):
        angle = angles[i]
        print(angle)
        plot_forces_angle(base, angle, axes1[i, 0])
        axes1[-1, 0].set_xlabel(r'$t\;\frac{U_0}{H}$')
        plot_fft(base, angle, axes1[i, 1])
        axes1[-1, 1].set_xlabel(r'$f\;\frac{H}{U_0}$')

    plt.show()
    fig1.savefig(f'figures/forces2.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
