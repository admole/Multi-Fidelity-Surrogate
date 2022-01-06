import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import mfRegression as mfr
import fonts

np.random.seed(2)

Func = "Sine"
# Func = "Step"

model = 'GP Mean'
model = 'MLP'

if Func == "Step":
    extraX = True
else:
    extraX = False


def hf(x):
    if Func == "Sine":
        y = 1.8 * np.sin(8.0 * np.pi * x) * 2 * x
    elif Func =="Step":
        y1 = 0.5*(6*x-2)**2*np.sin(12*x-4)+10*(x-0.5)-5
        y2 = 3+0.5*(6*x-2)**2*np.sin(12*x-4)+10*(x-0.5)-2
        y = np.zeros(np.shape(x))
        y[x<=0.5] = y1[x<=0.5]
        y[x>0.5] = y2[x>0.5]
    return y


def lf(x, c1, c2, c3):
    y = c1*hf(x+c2)+c3
    return y


X = np.linspace(0, 1, 1000)

Nhf = 8
Nlf = 50

# sample LF model randomly
X_lf = np.random.permutation(X)[0:Nlf]
X_hf = np.random.permutation(X_lf)[0:Nhf]
if extraX:
    Xfocus = np.linspace(0.4, 0.6, 20)
    X_lf = np.concatenate((X_lf, Xfocus))

X_hf[0] = 0.81

c1 = 1
c2 = 0
c3 =0

if model == 'MLP':
    X, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std = mfr.mfmlp(X_lf,
                                                                                                   lf(X_lf, c1, c2, c3),
                                                                                                   X_hf,
                                                                                                   hf(X_hf))
else:
    X, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std = mfr.mfgp(X_lf,
                                                                                                  lf(X_lf, c1, c2, c3),
                                                                                                  X_hf,
                                                                                                  hf(X_hf))
# Plotting --

legend_location = (1, 1)

fig, axs = plt.subplots(5, figsize=(12, 11), constrained_layout=True, sharex='none', sharey='none')

axs[0].plot(X, hf(X), label="High Fidelity / Exact")
lf_scatter, = axs[0].plot(X_lf, lf(X_lf, c1, c2, c3), 'bo', label="Low fidelity samples")
axs[0].plot(X_hf, hf(X_hf), 'ro', label="High fidelity samples")
axs[0].legend(bbox_to_anchor=legend_location, loc='upper left')

axs[1].plot(X, hf(X), label="High Fidelity / Exact")
axs[1].plot(X, pred_hf_mean, 'k', lw=3, label=f"High fidelity {model} \n(trained on red dots)")
axs[1].plot(X_hf, hf(X_hf), 'ro', label="High fidelity samples")

axs[2].plot(X, hf(X), label="High Fidelity / Exact")
lf_prediction_line, = axs[2].plot(X, pred_lf_mean, 'k', lw=3, label=f"Low fidelity {model} \n(trained on blue dots)")
lf_scatter2, = axs[2].plot(X_lf, lf(X_lf, c1, c2, c3), 'bo', label="Low fidelity samples")

axs[3].plot(X, hf(X), label="High Fidelity / Exact")
mf_prediction_line, = axs[3].plot(X, pred_mf_mean, 'k', lw=3, label=f"Multi fidelity {model} \n(trained on all dots)")

if model == 'GP Mean':
    axs[1].fill_between(X[:, 0], pred_hf_mean[:, 0] - 2 * pred_hf_std, pred_hf_mean[:, 0] + 2 * pred_hf_std, alpha=0.2,
                        color='k', label="+/- 2 std")
    lf_fill = axs[2].fill_between(X[:, 0], pred_lf_mean[:, 0] - 2 * pred_lf_std, pred_lf_mean[:, 0] + 2 * pred_lf_std, alpha=0.2,
                        color='k', label="+/- 2 std")
    mf_fill = axs[3].fill_between(X[:, 0], pred_mf_mean[:, 0] - 2 * pred_mf_std, pred_mf_mean[:, 0] + 2 * pred_mf_std, alpha=0.2,
                        color='k', label="+/- 2 std")

axs[1].legend(bbox_to_anchor=legend_location, loc='upper left')
axs[2].legend(bbox_to_anchor=legend_location, loc='upper left')
axs[3].legend(bbox_to_anchor=legend_location, loc='upper left')


axs[3].set_xlabel('$x$')
for i in range(4):
    axs[i].set_ylabel('$y=f(x)$')
# only works with plt version 3.4+ (requires python3.7+)
# fig.supxlabel('$x$')
# fig.supylabel('$y=f(x)$')


# correlation_3dline, = ax[4].plot3D(pred_lf_mean[:, 0], pred_mf_mean[:, 0], X[:, 0], 'gray')
# ax1.set_xlabel(r"$y_{lf}$")
# ax1.set_ylabel(r"$y_{hf}$")
# ax1.set_zlabel(r"$x$")

correlation_line2, = axs[4].plot(pred_lf_mean[:, 0], pred_hf_mean[:, 0], label='Exact')
correlation_line, = axs[4].plot(pred_lf_mean[:, 0], pred_mf_mean[:, 0], 'k', lw=3, label=f'MF{model}')
axs[4].set_xlabel(r"$y_{lf}$")
axs[4].set_ylabel(r"$y_{hf}$")
axs[4].legend()


fig.text(0.75, 0.18, r'$f(x)_{lf} = c_1 f(x+c_2)_{hf} + c_3$')
axc1 = plt.axes([0.8, 0.14, 0.15, 0.03])
sc1 = Slider(axc1, 'c1', -1, 2, valinit=c1)
axc2 = plt.axes([0.8, 0.1, 0.15, 0.03])
sc2 = Slider(axc2, 'c2', -0.5, 0.5, valinit=c2)
axc3 = plt.axes([0.8, 0.06, 0.15, 0.03])
sc3 = Slider(axc3, 'c3', -10, 10, valinit=c3)


def update(val):
    if model == 'MLP':
        X, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std = mfr.mfmlp(X_lf,
                                                                                                       lf(X_lf,
                                                                                                          sc1.val,
                                                                                                          sc2.val,
                                                                                                          sc3.val),
                                                                                                       X_hf,
                                                                                                       hf(X_hf))
    else:
        X, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std = mfr.mfgp(X_lf,
                                                                                                       lf(X_lf,
                                                                                                          sc1.val,
                                                                                                          sc2.val,
                                                                                                          sc3.val),
                                                                                                       X_hf,
                                                                                                       hf(X_hf))

    correlation_line.set_data(pred_lf_mean[:, 0], pred_mf_mean[:, 0])
    correlation_line2.set_data(pred_lf_mean[:, 0], hf(X))
    # correlation_3dline.set_data(pred_lf_mean[:, 0], pred_mf_mean[:, 0], X[:, 0])
    lf_scatter.set_data(X_lf, lf(X_lf, sc1.val, sc2.val, sc3.val))
    lf_scatter2.set_data(X_lf, lf(X_lf, sc1.val, sc2.val, sc3.val))
    lf_prediction_line.set_data(X, pred_lf_mean)
    mf_prediction_line.set_data(X, pred_mf_mean)

    if model == 'GP Mean':
        global lf_fill
        lf_fill.remove()
        lf_fill = axs[2].fill_between(X[:, 0], pred_lf_mean[:, 0] - 2 * pred_lf_std,
                                      pred_lf_mean[:, 0] + 2 * pred_lf_std, alpha=0.2,
                                      color='k', label="+/- 2 std")
        global mf_fill
        mf_fill.remove()
        mf_fill = axs[3].fill_between(X[:, 0], pred_mf_mean[:, 0] - 2 * pred_mf_std,
                                      pred_mf_mean[:, 0] + 2 * pred_mf_std, alpha=0.2,
                                      color='k', label="+/- 2 std")


sc1.on_changed(update)
sc2.on_changed(update)
sc3.on_changed(update)

plt.show()
