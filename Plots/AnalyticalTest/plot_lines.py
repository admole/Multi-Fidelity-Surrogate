#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox

parent = os.path.abspath('../')
sys.path.insert(1, parent)

from mfRegression import MFRegress
import fonts
from analyticalFunc import Line

np.random.seed(4)


# Plotting --
legend_location = (1, 1)
line = Line('Sine', 'MLP')
line.regression()

fig, axs = plt.subplots(5, figsize=(10, 12), constrained_layout=True, sharex='none', sharey='none')

axs[0].plot(line.X, line.hf(line.X), 'mediumseagreen', label="Exact")
lf_scatter, = axs[0].plot(line.X_lf, line.lf(line.X_lf), 'tab:red', linestyle='None', marker='o', label="Low-Fidelity samples")
axs[0].plot(line.X_hf, line.hf(line.X_hf), 'blue', linestyle='None', marker='o', label="High-Fidelity samples")

axs[1].plot(line.X, line.hf(line.X), 'mediumseagreen', label="Exact")
axs[1].plot(line.X, line.pred_hf_mean, 'k', lw=3, label=f"High-Fidelity {line.Model} \n(trained on blue dots)")
axs[1].plot(line.X_hf, line.hf(line.X_hf), 'blue', linestyle='None', marker='o', label="High-Fidelity samples")

axs[2].plot(line.X, line.hf(line.X), 'mediumseagreen', label="Exact")
lf_prediction_line, = axs[2].plot(line.X, line.pred_lf_mean, 'k', lw=3, label=f"Low-Fidelity {line.Model} \n(trained on red dots)")
lf_scatter2, = axs[2].plot(line.X_lf, line.lf(line.X_lf), 'tab:red', linestyle='None', marker='o', label="Low-Fidelity samples")

axs[3].plot(line.X, line.hf(line.X), 'mediumseagreen', label="Exact")
mf_prediction_line, = axs[3].plot(line.X, line.pred_mf_mean, 'k', lw=3, label=f"Multi-Fidelity {line.Model} \n(trained on all dots)")

if line.Model == 'GP Mean' or line.Model == 'GPR':
    axs[1].fill_between(line.X[:, 0], line.pred_hf_mean[:, 0] - 2 * line.pred_hf_std, line.pred_hf_mean[:, 0] + 2 * line.pred_hf_std, alpha=0.2,
                        color='k', label="+/- 2 std")
    lf_fill = axs[2].fill_between(line.X[:, 0], line.pred_lf_mean[:, 0] - 2 * line.pred_lf_std, line.pred_lf_mean[:, 0] + 2 * line.pred_lf_std, alpha=0.2,
                        color='k', label="+/- 2 std")
    mf_fill = axs[3].fill_between(line.X[:, 0], line.pred_mf_mean[:, 0] - 2 * line.pred_mf_std, line.pred_mf_mean[:, 0] + 2 * line.pred_mf_std, alpha=0.2,
                        color='k', label="+/- 2 std")

axs[0].legend(bbox_to_anchor=legend_location, loc='upper left', frameon=False)
axs[1].legend(bbox_to_anchor=legend_location, loc='upper left', frameon=False)
axs[2].legend(bbox_to_anchor=legend_location, loc='upper left', frameon=False)
axs[3].legend(bbox_to_anchor=legend_location, loc='upper left', frameon=False)


axs[3].set_xlabel('$x$')
for i in range(4):
    axs[i].set_ylabel('$y=f(x)$')

correlation_line2, = axs[4].plot(line.pred_lf_mean[:, 0], line.hf(line.X), 'mediumseagreen', label='Exact')
correlation_line, = axs[4].plot(line.pred_lf_mean[:, 0], line.pred_mf_mean[:, 0], 'k', lw=3, label=f'MF{line.Model}')
axs[4].set_xlabel(r"$y_{lf}$")
axs[4].set_ylabel(r"$y_{hf}$")
axs[4].legend()


fig.text(0.71, 0.20, r'$f(x)_{lf} = c_1 x^{c_2} \; f(c_3x+c_4)$')
fig.text(0.80, 0.17, r'$+ \; c_5x^{c6}$')
axc1 = plt.axes([0.77, 0.12, 0.06, 0.025])
sc1 = TextBox(axc1, r'$c_1\,=\,$', initial=f'{line.constants[0]}')
axc2 = plt.axes([0.92, 0.12, 0.06, 0.025])
sc2 = TextBox(axc2, r'$c_2\,=\,$', initial=f'{line.constants[1]}')
axc3 = plt.axes([0.77, 0.08, 0.06, 0.025])
sc3 = TextBox(axc3, r'$c_3\,=\,$', initial=f'{line.constants[2]}')
axc4 = plt.axes([0.92, 0.08, 0.06, 0.025])
sc4 = TextBox(axc4, r'$c_4\,=\,$', initial=f'{line.constants[3]}')
axc5 = plt.axes([0.77, 0.04, 0.06, 0.025])
sc5 = TextBox(axc5, r'$c_5\,=\,$', initial=f'{line.constants[4]}')
axc6 = plt.axes([0.92, 0.04, 0.06, 0.025])
sc6 = TextBox(axc6, r'$c_6\,=\,$', initial=f'{line.constants[5]}')


def update(val):
    line.constants[0] = float(sc1.text)
    line.constants[1] = float(sc2.text)
    line.constants[2] = float(sc3.text)
    line.constants[3] = float(sc4.text)
    line.constants[4] = float(sc5.text)
    line.constants[5] = float(sc6.text)
    line.regression()

    correlation_line.set_data(line.pred_lf_mean[:, 0], line.pred_mf_mean[:, 0])
    correlation_line2.set_data(line.pred_lf_mean[:, 0], line.hf(line.X))
    lf_scatter.set_data(line.X_lf, line.lf(line.X_lf))
    lf_scatter2.set_data(line.X_lf, line.lf(line.X_lf))
    lf_prediction_line.set_data(line.X, line.pred_lf_mean)
    mf_prediction_line.set_data(line.X, line.pred_mf_mean)

    if line.Model == 'GP Mean':
        global lf_fill
        lf_fill.remove()
        lf_fill = axs[2].fill_between(line.X[:, 0], line.pred_lf_mean[:, 0] - 2 * line.pred_lf_std,
                                      line.pred_lf_mean[:, 0] + 2 * line.pred_lf_std, alpha=0.2,
                                      color='k', label="+/- 2 std")
        global mf_fill
        mf_fill.remove()
        mf_fill = axs[3].fill_between(line.X[:, 0], line.pred_mf_mean[:, 0] - 2 * line.pred_mf_std,
                                      line.pred_mf_mean[:, 0] + 2 * line.pred_mf_std, alpha=0.2,
                                      color='k', label="+/- 2 std")

    for ax in axs:
        ax.relim()
        ax.autoscale_view(True, True, True)


sc1.on_submit(update)
sc2.on_submit(update)
sc3.on_submit(update)
sc4.on_submit(update)
sc5.on_submit(update)
sc6.on_submit(update)

plt.show()
fig.savefig(f'figures/analytical_{line.Func}_{line.Model}.pdf', bbox_inches='tight')
