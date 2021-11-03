#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import fonts
import json


def get_cf(case, u_inf=1, surface="zNormal"):
    cols = ['x', 'y', 'z', 'tx', 'ty', 'tz']
    # TODO: glob * for time directory
    data = pd.read_csv(f'../Data/{case}/postProcessing/WSS-surfaces/2000/wallShearStress_{surface}.raw',
                       skiprows=10, sep=r'\s+', names=cols, header=None, engine='python')
    data['cfx'] = -data['tx']/(0.5*u_inf)
    data['cfy'] = -data['ty']/(0.5*u_inf)
    data['cfz'] = -data['tz']/(0.5*u_inf)

    return data


def recirculation(case):
    data = get_cf(case, surface="walls")
    print(np.shape(data['x']))
    print(np.shape(data['z']))
    print(np.shape(data['cfx']))
    df.sort_values(by=['col1', 'col2'])

    recirc_int = np.trapz(np.trapz(data['cfx'], data['z']), data['x'])
    return recirc_int


def plot_cf(case, ax):
    data = get_cf(case, u_inf=1)
    data = data.sort_values(by='x')
    center = abs(data["z"]) <= 0.1
    top = abs(data["y"]) == 1
    val = data['cfx'][center] != 0
    # inter = data['x'][center][val] > 3
    ax.plot(data['x'][center][val], data['cfx'][center][val], 'k')
    # ax.plot([0, 14], [0, 0], 'k--', alpha=0.4)
    ax.fill_between(data['x'][center][val], data['cfx'][center][val], 0,
                    interpolate=True, where=data['cfx'][center][val] < 0,
                    facecolor='r', alpha=0.7)
    ax.set_xlabel(r'x/H')
    ax.set_ylabel(r'Cf')
    ax.axvspan(2, 3, alpha=0.2, color='k')
    ax.axvspan(7, 8, alpha=0.2, color='k')
    ax.set_xlim(0, 14)
    print(f"Cfx max = {data['cfx'].max(axis=0)}")
    print(f"Cfx min = {data['cfx'].min(axis=0)}")

    m = data['cfx'][center][val].lt(0)
    recirc = np.sum(data['x'][center][val][m] - data['x'][center][val][m].shift())
    print(f"Recirc length = {recirc}")
    switches = data['x'][center][val][m != m.shift()]
    print(switches[switches > 3][switches < 7])

    # switch_positions = []
    # for i in data[center][val]:
    #     switch = i['cfx']/data['cfx'][center][val][i+1] < 0
    #     if switch:
    #         switch_positions.append(data['x'][center][val][i+1])
    #
    # print(switch_positions)

    # dataslice.plot(x='x', y='cfx', ax=ax, style='r')


# # Opening JSON file
# j = open(os.path.join(os.getcwd(), "../Cases/cases.json"))
# case_settings = json.load(j)
# numcases = len(case_settings)


# fig1, axes1 = plt.subplots(1, 1, figsize=(7, 3), squeeze=False, constrained_layout=True)
# plot_cf("test10", axes1[0, 0])
#
# plt.show()
# # fig1.savefig('figures/Ubulk_time.pdf', bbox_inches='tight')
