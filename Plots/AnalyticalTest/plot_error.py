#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from matplotlib import pyplot as plt

parent = os.path.abspath('../')
sys.path.insert(1, parent)

from mfRegression import MFRegress
import fonts
from analyticalFunc import Line
from sklearn.gaussian_process.kernels import (RBF, Matern, DotProduct)

np.random.seed(2)


def reset_constants(line):
    line.constants[0] = 1
    line.constants[1] = 0
    line.constants[2] = 1
    line.constants[3] = 0
    line.constants[4] = 0
    line.constants[5] = 0


def collect_error(regress, constant, crange):
    reset_constants(regress)
    if constant == 5:
        regress.constants[4] = 1
    mf_error = np.zeros(len(crange))
    hf_error = np.zeros(len(crange))
    lf_error = np.zeros(len(crange))
    for i in range(len(crange)):
        regress.constants[constant] = crange[i]
        regress.regression()
        mf_error[i], hf_error[i], lf_error[i] = regress.errors()
    return mf_error, hf_error, lf_error


legend_location = (1, 1)
line = Line('Sine', 'MLP')
reset_constants(line)
line.k_hf *= DotProduct()**1

print('c1')
c1s = np.arange(-1, 2.1, 0.1)
c1_mf_error, c1_hf_error, c1_lf_error = collect_error(line, 0, c1s)
print('c2')
c2s = np.arange(-1, 2.1, 0.1)
c2_mf_error, c2_hf_error, c2_lf_error = collect_error(line, 1, c2s)
print('c3')
c3s = np.arange(-1, 2.1, 0.1)
c3_mf_error, c3_hf_error, c3_lf_error = collect_error(line, 2, c3s)
print('c4')
c4s = np.arange(-1, 2.1, 0.1)
c4_mf_error, c4_hf_error, c4_lf_error = collect_error(line, 3, c4s)
print('c5')
c5s = np.arange(-1, 2.1, 0.1)
c5_mf_error, c5_hf_error, c5_lf_error = collect_error(line, 4, c5s)
print('c6')
c6s = np.arange(-1, 2.1, 0.1)
c6_mf_error, c6_hf_error, c6_lf_error = collect_error(line, 5, c6s)


fig, axs = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=True, sharex='none', sharey='none')

axs[0, 0].plot(c1s, c1_lf_error, 'tab:red', label=f'LF-{line.Model}')
axs[0, 1].plot(c2s, c2_lf_error, 'tab:red', label=f'LF-{line.Model}')
axs[1, 0].plot(c3s, c3_lf_error, 'tab:red', label=f'LF-{line.Model}')
axs[1, 1].plot(c4s, c4_lf_error, 'tab:red', label=f'LF-{line.Model}')
axs[2, 0].plot(c5s, c5_lf_error, 'tab:red', label=f'LF-{line.Model}')
axs[2, 1].plot(c6s, c6_lf_error, 'tab:red', label=f'LF-{line.Model}')

axs[0, 0].plot(c1s, c1_hf_error, 'blue', label=f'HF-{line.Model}')
axs[0, 1].plot(c2s, c2_hf_error, 'blue', label=f'HF-{line.Model}')
axs[1, 0].plot(c3s, c3_hf_error, 'blue', label=f'HF-{line.Model}')
axs[1, 1].plot(c4s, c4_hf_error, 'blue', label=f'HF-{line.Model}')
axs[2, 0].plot(c5s, c5_hf_error, 'blue', label=f'HF-{line.Model}')
axs[2, 1].plot(c6s, c6_hf_error, 'blue', label=f'HF-{line.Model}')

axs[0, 0].plot(c1s, c1_mf_error, 'k', label=f'MF-{line.Model}')
axs[0, 1].plot(c2s, c2_mf_error, 'k', label=f'MF-{line.Model}')
axs[1, 0].plot(c3s, c3_mf_error, 'k', label=f'MF-{line.Model}')
axs[1, 1].plot(c4s, c4_mf_error, 'k', label=f'MF-{line.Model}')
axs[2, 0].plot(c5s, c5_mf_error, 'k', label=f'MF-{line.Model}')
axs[2, 1].plot(c6s, c6_mf_error, 'k', label=f'MF-{line.Model}')

axs[0, 0].set_ylabel('MSE')
axs[0, 1].set_ylabel('MSE')
axs[1, 0].set_ylabel('MSE')
axs[1, 1].set_ylabel('MSE')
axs[2, 0].set_ylabel('MSE')
axs[2, 1].set_ylabel('MSE')

axs[0, 0].set_xlabel('$c_1$')
axs[0, 1].set_xlabel('$c_2$')
axs[1, 0].set_xlabel('$c_3$')
axs[1, 1].set_xlabel('$c_4$')
axs[2, 0].set_xlabel('$c_5$')
axs[2, 1].set_xlabel('$c_6$')

axs[0, 0].legend(frameon=False)

axs[0, 0].set_yscale('log')
axs[0, 1].set_yscale('log')
axs[1, 0].set_yscale('log')
axs[1, 1].set_yscale('log')
axs[2, 0].set_yscale('log')
axs[2, 1].set_yscale('log')


plt.show()
fig.savefig(f'figures/error_{line.Func}_{line.Model}.pdf', bbox_inches='tight')
