from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn import preprocessing
import numpy as np


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

    gpr_hf = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=2000, normalize_y=True).fit(x_hf, hf)
    gpr_lf = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=200, normalize_y=True).fit(x_lf, lf)

    l1mean = gpr_lf.predict(x_hf)
    l2_train = np.hstack((x_hf, l1mean))
    gpr_mf_l2 = GaussianProcessRegressor(kernel=RBF() * RBF() + RBF(), n_restarts_optimizer=200, normalize_y=True).fit(l2_train, hf)

    pred_hf_mean, pred_hf_std = gpr_hf.predict(x, return_std=True)
    pred_lf_mean, pred_lf_std = gpr_lf.predict(x, return_std=True)

    l2_test = np.hstack((x, pred_lf_mean))
    pred_mf_mean, pred_mf_std = gpr_mf_l2.predict(l2_test, return_std=True)

    return x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std
