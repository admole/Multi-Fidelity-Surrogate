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

