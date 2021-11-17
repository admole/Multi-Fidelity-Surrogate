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

    # k_lf = RBF()
    # k_lf = RationalQuadratic()

    gpr_lf = MLPRegressor(random_state=1, solver='lbfgs', max_iter=2000).fit(x_lf, lf)
    # pprint.pprint(gpr_lf.kernel_.get_params())
    # length = gpr_lf.kernel_.get_params()['length_scale']
    # k_hf = RBF(length, length_scale_bounds='fixed')
    gpr_hf = MLPRegressor(random_state=1, solver='lbfgs', max_iter=2000).fit(x_hf, hf)

    l1mean = gpr_lf.predict(x_hf)
    l1mean = l1mean.reshape(-1, 1)
    l2_train = np.hstack((x_hf, l1mean))
    # k_mf = k_hf  # * k_hf + k_hf
    gpr_mf_l2 = MLPRegressor(random_state=1, solver='lbfgs', max_iter=2000).fit(l2_train, hf)

    pred_hf_mean = gpr_hf.predict(x)
    pred_lf_mean = gpr_lf.predict(x)

    pred_lf_mean = pred_lf_mean.reshape(-1, 1)
    pred_hf_mean = pred_hf_mean.reshape(-1, 1)
    l2_test = np.hstack((x, pred_lf_mean))
    pred_mf_mean = gpr_mf_l2.predict(l2_test)
    pred_mf_mean = pred_mf_mean.reshape(-1, 1)

    pred_lf_std = np.zeros(len(pred_lf_mean))
    pred_hf_std = np.zeros(len(pred_hf_mean))
    pred_mf_std = np.zeros(len(pred_mf_mean))

    return x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std
