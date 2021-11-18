from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel,
                                              CompoundKernel, Kernel,
                                              Product, Sum)
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import numpy as np
import pprint


def mfgp(x_lf, lf, x_hf, hf):
    xmin = min(min(x_lf), min(x_hf))
    xmax = max(max(x_lf), max(x_hf))
    x = np.linspace(xmin, xmax, 1000)[:, np.newaxis]
    x_lf = x_lf.reshape(-1, 1)
    x_hf = x_hf.reshape(-1, 1)
    lf = lf.T
    hf = hf.T
    lf = lf.reshape(-1, 1)
    hf = hf.reshape(-1, 1)

    solver = 'lbfgs'
    activation = 'relu'
    hidden_layers = (100)

    mlpr_lf = MLPRegressor(activation=activation,
                           hidden_layer_sizes=hidden_layers,
                           solver=solver,
                           random_state=1,
                           shuffle=True,
                           max_iter=1000).fit(x_lf, lf)
    mlpr_hf = MLPRegressor(activation=activation,
                           hidden_layer_sizes=hidden_layers,
                           solver=solver,
                           random_state=1,
                           shuffle=True,
                           max_iter=1000).fit(x_hf, hf)

    l1mean = mlpr_lf.predict(x_hf)
    l1mean = l1mean.reshape(-1, 1)
    l2_train = np.hstack((x_hf, l1mean))
    mlpr_mf_l2 = MLPRegressor(activation=activation,
                              hidden_layer_sizes=hidden_layers,
                              solver=solver,
                              random_state=1,
                              shuffle=True,
                              max_iter=1000).fit(l2_train, hf)

    pred_hf_mean = mlpr_hf.predict(x)
    pred_lf_mean = mlpr_lf.predict(x)

    pred_lf_mean = pred_lf_mean.reshape(-1, 1)
    pred_hf_mean = pred_hf_mean.reshape(-1, 1)
    l2_test = np.hstack((x, pred_lf_mean))
    pred_mf_mean = mlpr_mf_l2.predict(l2_test)
    pred_mf_mean = pred_mf_mean.reshape(-1, 1)

    pred_lf_std = np.zeros(len(pred_lf_mean))
    pred_hf_std = np.zeros(len(pred_hf_mean))
    pred_mf_std = np.zeros(len(pred_mf_mean))

    return x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std
