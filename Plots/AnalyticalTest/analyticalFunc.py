#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from mpl_toolkits import mplot3d


parent = os.path.abspath('../')
sys.path.insert(1, parent)

from mfRegression import MFRegress
import fonts

np.random.seed(2)


def sine(x):
    return 1.8 * np.sin(8.0 * np.pi * x) * 2 * x


def sine2d(x, y):
    x, y = np.meshgrid(x, y)
    return (1.8 * np.sin(8.0 * np.pi * x) * 2 * x) * (1 + 2*y**3)
    # return (1.8 * np.sin(8.0 * np.pi * x) * 2 * x) * (1 + 2*np.sin(2.0 * np.pi * y))


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
            self.constants = [0.5, 0.0, 1.0, 0.04, -1.0, 0.0]
            self.activ = 'tanh'
            self.k_lf = RBF()
        elif self.Func == 'Step':
            self.constants = [0.5, 0.0, 1.1, -0.05, -5.0, 0.0]
            self.activ = 'relu'
            self.k_lf = Matern()
        self.k_hf = self.k_lf ** 2 + self.k_lf
        self.hl_lf = (150, 145, 115, 85, 45, 150, 15)
        self.hl_hf = (25, 120)
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
        y = self.constants[0]*x**self.constants[1]\
            * self.hf(self.constants[2]*x+self.constants[3])\
            + self.constants[4]*x**self.constants[5]
        return y

    def regression(self):
        regress = MFRegress(self.X_lf, self.lf(self.X_lf), self.X_hf, self.hf(self.X_hf))
        if self.Model == 'MLP':
            self.X, self.pred_lf_mean, self.pred_lf_std,\
                self.pred_hf_mean, self.pred_hf_std,\
                self.pred_mf_mean, self.pred_mf_std = regress.mfmlp(hidden_layers1=self.hl_lf,
                                                                    hidden_layers2=self.hl_hf,
                                                                    activation=self.activ)
        else:
            self.X, self.pred_lf_mean, self.pred_lf_std,\
                self.pred_hf_mean, self.pred_hf_std,\
                self.pred_mf_mean, self.pred_mf_std = regress.mfgp(self.k_lf, self.k_hf)

    def optimise(self):
        if self.Model == 'MLP':
            import random
            n_runs = 60
            best_mf_error = float("inf")
            for i in range(n_runs):
                # generate random architectures
                self.hl_lf = []
                self.hl_hf = []
                for layer in range(random.randint(1, 8)):
                    self.hl_lf.append(5 * random.randint(1, 32))
                for layer in range(random.randint(1, 8)):
                    self.hl_hf.append(5 * random.randint(1, 32))

                self.regression()
                mf_error, hf_error, lf_error = self.errors()
                if mf_error < best_mf_error:
                    best_lf_hl = self.hl_lf
                    best_hf_hl = self.hl_hf
                    best_mf_error = mf_error
            self.hl_lf = best_lf_hl
            self.hl_hf = best_hf_hl
            print(f'Optimised architecture is LF:{self.hl_lf} and HF:{self.hl_hf} \n with error of: {best_mf_error}')

    def errors(self):
        from sklearn.metrics import mean_squared_error
        mf_error = mean_squared_error(self.hf(self.X), self.pred_mf_mean)
        hf_error = mean_squared_error(self.hf(self.X), self.pred_hf_mean)
        lf_error = mean_squared_error(self.hf(self.X), self.pred_lf_mean)
        return mf_error, hf_error, lf_error


class Surface:
    def __init__(self, func, model):
        from sklearn.gaussian_process.kernels import (RBF, Matern)
        self.Func = func
        self.Model = model
        self.Y = np.linspace(0, 1, 20)
        self.X = np.linspace(0, 1, 1000)
        # self.X, self.Y = np.meshgrid(self.X, self.Y)
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
            self.constants = [0.5, 0.0, 1.0, 0.0, 4.0, 0.0]
            self.activ = 'tanh'
            self.k_lf = RBF()
        elif self.Func == 'Step':
            self.constants = [0.5, 0.0, 1.1, -0.05, -5.0, 0.0]
            self.activ = 'relu'
            self.k_lf = Matern()
        self.k_hf = self.k_lf ** 2 + self.k_lf
        self.hl_lf = (150, 145, 115, 85, 45, 150, 15)
        self.hl_hf = (25, 120)
        self.pred_lf_mean = None
        self.pred_lf_std = None
        self.pred_hf_mean = None
        self.pred_hf_std = None
        self.pred_mf_mean = None
        self.pred_mf_std = None

    def hf(self, x):
        if self.Func == "Sine":
            func = sine2d(x, self.Y)
        elif self.Func == "Step":
            func = step(x)
        else:
            sys.exit("function should be Sine or Step")
        return func

    def lf(self, x):
        func = self.constants[0]*x**self.constants[1]\
            * self.hf(self.constants[2]*x+self.constants[3])\
            + self.constants[4]*x**self.constants[5]
        av = np.average(func, 0)
        return av

    def regression(self):
        print(f'hf_length = {np.shape(self.hf(self.X_hf))}')
        regress = MFRegress(self.X_lf, self.lf(self.X_lf),
                            np.array(self.X_hf), np.array(self.hf(self.X_hf)).T, embedding_theory=False)

        if self.Model == 'MLP':
            self.X, self.pred_lf_mean, self.pred_lf_std,\
                self.pred_hf_mean, self.pred_hf_std,\
                self.pred_mf_mean, self.pred_mf_std = regress.mfmlp(hidden_layers1=self.hl_lf,
                                                                    hidden_layers2=self.hl_hf,
                                                                    activation=self.activ)
        else:
            self.X, self.pred_lf_mean, self.pred_lf_std,\
                self.pred_hf_mean, self.pred_hf_std,\
                self.pred_mf_mean, self.pred_mf_std = regress.mfgp(self.k_lf, self.k_hf)

    # def errors(self):
    #     from sklearn.metrics import mean_squared_error
    #     mf_error = mean_squared_error(self.hf(self.X), self.pred_mf_mean)
    #     hf_error = mean_squared_error(self.hf(self.X), self.pred_hf_mean)
    #     lf_error = mean_squared_error(self.hf(self.X), self.pred_lf_mean)
    #     return mf_error, hf_error, lf_error

