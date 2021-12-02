
def mfgp(x_lf, lf, x_hf, hf):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                                  ExpSineSquared, DotProduct,
                                                  ConstantKernel, WhiteKernel,
                                                  CompoundKernel, Kernel,
                                                  Product, Sum)
    import numpy as np
    import pprint

    xmin = min(min(x_lf), min(x_hf))
    xmax = max(max(x_lf), max(x_hf))
    x = np.linspace(xmin, xmax, 1000)[:, np.newaxis]
    x_lf = x_lf.reshape(-1, 1)
    x_hf = x_hf.reshape(-1, 1)
    lf = lf.T
    hf = hf.T
    lf = lf.reshape(-1, 1)
    hf = hf.reshape(-1, 1)

    k_lf = RBF()

    gpr_lf = GaussianProcessRegressor(kernel=k_lf, n_restarts_optimizer=200, normalize_y=True).fit(x_lf, lf)
    pprint.pprint(gpr_lf.kernel_.get_params())
    k_hf = RBF()
    gpr_hf = GaussianProcessRegressor(kernel=k_hf, n_restarts_optimizer=2000, normalize_y=True).fit(x_hf, hf)

    l1mean = gpr_lf.predict(x_hf)
    l1mean_shift1 = gpr_lf.predict(x_hf+0.02)
    l1mean_shift2 = gpr_lf.predict(x_hf-0.02)
    l2_train = np.hstack((x_hf, l1mean, l1mean_shift1, l1mean_shift2))
    k_mf = k_hf * k_hf + k_hf
    gpr_mf_l2 = GaussianProcessRegressor(kernel=k_mf, n_restarts_optimizer=200, normalize_y=True).fit(l2_train, hf)

    pred_hf_mean, pred_hf_std = gpr_hf.predict(x, return_std=True)
    pred_lf_mean, pred_lf_std = gpr_lf.predict(x, return_std=True)
    pred_lf_mean_shift1, pred_lf_std_shift1 = gpr_lf.predict(x+0.02, return_std=True)
    pred_lf_mean_shift2, pred_lf_std_shift2 = gpr_lf.predict(x-0.02, return_std=True)

    l2_test = np.hstack((x, pred_lf_mean, pred_lf_mean_shift1, pred_lf_mean_shift2))
    pred_mf_mean, pred_mf_std = gpr_mf_l2.predict(l2_test, return_std=True)

    return x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std


def mfmlp(x_lf, lf, x_hf, hf):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import StackingRegressor
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    xmin = min(min(x_lf), min(x_hf))
    xmax = max(max(x_lf), max(x_hf))
    x = np.linspace(xmin, xmax, 1000)[:, np.newaxis]
    x_lf = x_lf.reshape(-1, 1)
    x_hf = x_hf.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(x)
    x_lf = scaler.transform(x_lf)
    x_hf = scaler.transform(x_hf)
    x = scaler.transform(x)

    lf = lf.T
    hf = hf.T
    lf = lf.reshape(-1, 1)
    hf = hf.reshape(-1, 1)

    datascaler = MinMaxScaler()
    datascaler.fit(lf)
    lf = datascaler.transform(lf)
    hf = datascaler.transform(hf)
    print(np.min(lf), np.max(lf))

    solver = 'lbfgs'
    activation = 'tanh'
    # activation = 'relu'
    hidden_layers = (20, 50, 20)
    hidden_layers = (20, 20, 20, 20)

    mlpr_lf = MLPRegressor(activation=activation,
                           hidden_layer_sizes=hidden_layers,
                           solver=solver,
                           random_state=1,
                           max_iter=1000).fit(x_lf, lf)
    mlpr_hf = MLPRegressor(activation=activation,
                           hidden_layer_sizes=hidden_layers,
                           solver=solver,
                           random_state=1,
                           max_iter=1000).fit(x_hf, hf)

    l1mean = mlpr_lf.predict(x_hf).reshape(-1, 1)
    l1mean_shift1 = mlpr_lf.predict(x_hf+0.02).reshape(-1, 1)
    l1mean_shift2 = mlpr_lf.predict(x_hf-0.02).reshape(-1, 1)
    # l2_train = np.hstack((x_hf, l1mean))
    l2_train = np.hstack((x_hf, l1mean, l1mean_shift1, l1mean_shift2))
    # l2_train = np.hstack((x_hf, l1mean))

    mlpr_mf_nlin = MLPRegressor(activation=activation,
                                hidden_layer_sizes=hidden_layers,
                                solver=solver,
                                random_state=1,
                                # alpha=0.0001,
                                max_iter=1000).fit(l2_train, hf)

    pred_hf_mean = mlpr_hf.predict(x).reshape(-1, 1)
    pred_lf_mean = mlpr_lf.predict(x).reshape(-1, 1)
    pred_lf_mean_shift1 = mlpr_lf.predict(x+0.02).reshape(-1, 1)
    pred_lf_mean_shift2 = mlpr_lf.predict(x-0.02).reshape(-1, 1)

    l2_test = np.hstack((x, pred_lf_mean, pred_lf_mean_shift1, pred_lf_mean_shift2))
    # l2_test = np.hstack((x, pred_lf_mean))
    pred_mf_mean = mlpr_mf_nlin.predict(l2_test)
    pred_mf_mean = pred_mf_mean.reshape(-1, 1)

    pred_lf_std = np.zeros(len(pred_lf_mean))
    pred_hf_std = np.zeros(len(pred_hf_mean))
    pred_mf_std = np.zeros(len(pred_mf_mean))

    x = scaler.inverse_transform(x)
    pred_lf_mean = datascaler.inverse_transform(pred_lf_mean)
    pred_hf_mean = datascaler.inverse_transform(pred_hf_mean)
    pred_mf_mean = datascaler.inverse_transform(pred_mf_mean)

    return x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std
