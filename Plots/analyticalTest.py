#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from mfRegression import MFRegress
import fonts

np.random.seed(2)


def sine(x):
    return 1.8 * np.sin(8.0 * np.pi * x) * 2 * x


def step(x):
    y1 = 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5
    y2 = 3 + 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 2
    y = np.zeros(np.shape(x))
    y[x <= 0.5] = y1[x <= 0.5]
    y[x > 0.5] = y2[x > 0.5]
    return y


class Line:
    def __init__(self, func, model):
        from sklearn.gaussian_process.kernels import (RBF, Matern)
        self.Func = func
        self.Model = model
        self.X = np.linspace(0, 1, 1000)
        nhf = 8
        nlf = 50
        if self.Func == "Step":
            extrax = True
        else:
            extrax = False
        self.X_lf = np.random.permutation(self.X)[0:nlf]
        self.X_hf = np.random.permutation(self.X_lf)[0:nhf]
        self.X_hf[0] = 0.81
        if extrax:
            xfocus = np.linspace(0.4, 0.6, 20)
            self.X_lf = np.concatenate((self.X_lf, xfocus))
        if self.Func == 'Sine':
            self.c1 = 0.5
            self.c2 = 1
            self.c3 = 0.04
            self.c4 = -1.0
            self.activ = 'tanh'
            self.k_lf = RBF()
        elif self.Func == 'Step':
            self.c1 = 0.5
            self.c2 = 1.1
            self.c3 = -0.05
            self.c4 = -5.0
            self.activ = 'relu'
            self.k_lf = Matern()
        self.k_hf = self.k_lf ** 2 + self.k_lf
        self.pred_lf_mean = None
        self.pred_lf_std = None
        self.pred_hf_mean = None
        self.pred_hf_std = None
        self.pred_mf_mean = None
        self.pred_mf_std = None

    def hf(self, x):
        if self.Func == "Sine":
            y = sine(x)
        elif self.Func == "Step":
            y = step(x)
        else:
            sys.exit("function should be Sine or Step")
        return y

    def lf(self, x):
        y = self.c1*self.hf(self.c2*x+self.c3)+self.c4
        return y

    def regression(self):
        regress = MFRegress(self.X_lf, self.lf(self.X_lf), self.X_hf, self.hf(self.X_hf))
        if self.Model == 'MLP':
            self.X, self.pred_lf_mean, self.pred_lf_std,\
                self.pred_hf_mean, self.pred_hf_std,\
                self.pred_mf_mean, self.pred_mf_std = regress.mfmlp(hidden_layers1=(8, 32, 32, 8),
                                                                    hidden_layers2=(8, 32, 32, 8),
                                                                    activation=self.activ)
        else:
            self.X, self.pred_lf_mean, self.pred_lf_std,\
                self.pred_hf_mean, self.pred_hf_std,\
                self.pred_mf_mean, self.pred_mf_std = regress.mfgp(self.k_lf, self.k_hf)


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

if line.Model == 'GP Mean':
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


fig.text(0.7, 0.18, r'$f(x)_{lf} = c_1 f(c_2x+c_3)_{hf} + c_4$')
axc1 = plt.axes([0.75, 0.14, 0.06, 0.025])
sc1 = TextBox(axc1, r'$c_1\,=\,$', initial=f'{line.c1}')
axc2 = plt.axes([0.90, 0.14, 0.06, 0.025])
sc2 = TextBox(axc2, r'$c_2\,=\,$', initial=f'{line.c2}')
axc3 = plt.axes([0.75, 0.10, 0.06, 0.025])
sc3 = TextBox(axc3, r'$c_3\,=\,$', initial=f'{line.c3}')
axc4 = plt.axes([0.90, 0.10, 0.06, 0.025])
sc4 = TextBox(axc4, r'$c_4\,=\,$', initial=f'{line.c4}')


def update(val):
    line.c1 = float(sc1.text)
    line.c2 = float(sc2.text)
    line.c3 = float(sc3.text)
    line.c4 = float(sc4.text)
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


sc1.on_submit(update)
sc2.on_submit(update)
sc3.on_submit(update)
sc4.on_submit(update)

plt.show()
fig.savefig(f'figures/analytical_{line.Func}_{line.Model}.pdf', bbox_inches='tight')
