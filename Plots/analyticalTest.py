import numpy as np
from matplotlib import pyplot as plt
import mfRegression as mfr

np.random.seed(2)


def hf(x):
    # return 1.8*np.sin(8.0*np.pi*x)*2*x
    return 1.8 * np.sin(8.0 * np.pi * x) * 2 * x


def lf(x):
    # return np.sin(8.0*np.pi*x)*x
    return np.sin(8.0 * np.pi * x) * x


X = np.linspace(0, 1, 1000)

Nhf = 8
Nlf = 50

# sample LF model randomly
X_lf = np.random.permutation(X)[0:Nlf]
X_hf = np.random.permutation(X_lf)[0:Nhf]

X_hf[0] = 0.81

X, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std = mfr.mfgp(X_lf, lf(X_lf), X_hf, hf(X_hf))

# Plotting --

fig, axs = plt.subplots(4)
axs[0].plot(X, hf(X), label="High Fidelity / Exact")

axs[0].plot(X_lf, lf(X_lf), 'bo', label="Low fidelity samples")
axs[0].plot(X_hf, hf(X_hf), 'ro', label="High fidelity samples")

axs[0].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

axs[1].plot(X, hf(X), label="High Fidelity / Exact")
axs[1].plot(X, pred_hf_mean, 'k', lw=3, label="GP mean (trained on red dots)")
axs[1].plot(X_hf, hf(X_hf), 'ro', label="High fidelity samples")
axs[1].fill_between(X[:, 0], pred_hf_mean[:, 0] - 2 * pred_hf_std, pred_hf_mean[:, 0] + 2 * pred_hf_std, alpha=0.2,
                    color='k', label="+/- 2 std")
axs[1].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

axs[2].plot(X, hf(X), label="High Fidelity / Exact")
axs[2].plot(X, pred_lf_mean, 'k', lw=3, label="GP mean (trained on blue dots)")
axs[2].plot(X_lf, lf(X_lf), 'bo', label="Low fidelity samples")
axs[2].fill_between(X[:, 0], pred_lf_mean[:, 0] - 2 * pred_lf_std, pred_lf_mean[:, 0] + 2 * pred_lf_std, alpha=0.2,
                    color='k', label="+/- 2 std")
axs[2].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

axs[3].plot(X, hf(X), label="High Fidelity / Exact")
axs[3].plot(X, pred_mf_mean, 'k', lw=3, label="Deep GP mean (trained on all dots)")
axs[3].fill_between(X[:, 0], pred_mf_mean[:, 0] - 2 * pred_mf_std, pred_mf_mean[:, 0] + 2 * pred_mf_std, alpha=0.2,
                    color='k', label="+/- 2 std")
axs[3].legend(bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='x-small')

fig.text(0.5, 0.03, '$x$', ha='center')
fig.text(0.03, 0.5, '$y=f(x)$', va='center', rotation='vertical')
plt.show()
