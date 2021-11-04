from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import matplotlib
import numpy as np

from matplotlib import pyplot as plt

np.random.seed(2)


def hf(x):
    return 1.8 * np.sin(8.0 * np.pi * x) * 2 * x


def lf(x):
    return np.sin(8.0 * np.pi * x) * x


# X = np.linspace(-np.pi, np.pi, 1000)[:, np.newaxis]
X = np.linspace(0, 1, 1000)[:, np.newaxis]

Nhf = 8
Nlf = 50

# sample LF model randomly
X_lf = np.random.permutation(X)[0:Nlf]
X_hf = np.random.permutation(X_lf)[0:Nhf]

X_hf[0] = 0.81

gpr_hf = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=2000).fit(X_hf, hf(X_hf))
gpr_lf = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=200).fit(X_lf, lf(X_lf))

L1mean = gpr_lf.predict(X_hf)

L2_train = np.hstack((X_hf, L1mean))

gpr_mf_l2 = GaussianProcessRegressor(kernel=RBF() * RBF() + RBF(), n_restarts_optimizer=200).fit(L2_train, hf(X_hf))

# Plotting --

fig, axs = plt.subplots(4)
axs[0].plot(X, hf(X), label="High Fidelity / Exact")

axs[0].plot(X_lf, lf(X_lf), 'bo', label="Low fidelity samples")
axs[0].plot(X_hf, hf(X_hf), 'ro', label="High fidelity samples")

axs[0].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

pred_hf_mean, pred_hf_std = gpr_hf.predict(X, return_std=True)

axs[1].plot(X, hf(X), label="High Fidelity / Exact")
axs[1].plot(X, pred_hf_mean, 'k', lw=3, label="GP mean (trained on red dots)")
axs[1].plot(X_hf, hf(X_hf), 'ro', label="High fidelity samples")
axs[1].fill_between(X[:, 0], pred_hf_mean[:, 0] - 2 * pred_hf_std, pred_hf_mean[:, 0] + 2 * pred_hf_std, alpha=0.2,
                    color='k', label="+/- 2 std")
axs[1].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

pred_lf_mean, pred_lf_std = gpr_lf.predict(X, return_std=True)

axs[2].plot(X, hf(X), label="High Fidelity / Exact")
axs[2].plot(X, pred_lf_mean, 'k', lw=3, label="GP mean (trained on blue dots)")
axs[2].plot(X_lf, lf(X_lf), 'bo', label="Low fidelity samples")
axs[2].fill_between(X[:, 0], pred_lf_mean[:, 0] - 2 * pred_lf_std, pred_lf_mean[:, 0] + 2 * pred_lf_std, alpha=0.2,
                    color='k', label="+/- 2 std")
axs[2].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

L2_test = np.hstack((X, pred_lf_mean))
pred_mf_mean, pred_mf_std = gpr_mf_l2.predict(L2_test, return_std=True)

axs[3].plot(X, hf(X), label="High Fidelity / Exact")
axs[3].plot(X, pred_mf_mean, 'k', lw=3, label="Deep GP mean (trained on all dots)")
axs[3].fill_between(X[:, 0], pred_mf_mean[:, 0] - 2 * pred_mf_std, pred_mf_mean[:, 0] + 2 * pred_mf_std, alpha=0.2,
                    color='k', label="+/- 2 std")
axs[3].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

fig.text(0.5, 0.03, '$x$', ha='center')
fig.text(0.03, 0.5, '$y=f(x)$', va='center', rotation='vertical')
plt.show()
