class MFRegress:
    def __init__(self, x_lf, lf, x_hf, hf,
                 embedding_theory=True,
                 gradient=False):
        import numpy as np

        self.x_lf = x_lf
        self.lf = lf
        self.x_hf = x_hf
        self.hf = hf
        self.embedding_theory = embedding_theory
        self.gradient = gradient
        xmin = min(min(self.x_lf), min(self.x_hf))
        xmax = max(max(self.x_lf), max(self.x_hf))
        self.x = np.linspace(xmin, xmax, 1001)[:, np.newaxis]

    def prep(self):
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        self.x_lf = self.x_lf.reshape(-1, 1)
        self.x_hf = self.x_hf.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(self.x)
        self.x_lf = scaler.transform(self.x_lf)
        self.x_hf = scaler.transform(self.x_hf)
        self.x = scaler.transform(self.x)
        if len(np.shape(self.lf)) == 1:
            self.lf = self.lf.T
            self.hf = self.hf.T
            self.lf = self.lf.reshape(-1, 1)
            self.hf = self.hf.reshape(-1, 1)
        datascaler = MinMaxScaler()
        datascaler.fit(self.lf)
        self.lf = datascaler.transform(self.lf)
        self.hf = datascaler.transform(self.hf)

        return scaler, datascaler

    def mfgp(self, kernel_lf, kernel_hf):
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor

        if len(np.shape(self.lf)) == 1:
            single = True
            print('single')
        else:
            single = False

        scaler, datascaler = self.prep()

        gpr_lf = GaussianProcessRegressor(kernel=kernel_lf,
                                          n_restarts_optimizer=200,
                                          normalize_y=True).fit(self.x_lf, self.lf)
        gpr_hf = GaussianProcessRegressor(kernel=kernel_hf,
                                          n_restarts_optimizer=200,
                                          normalize_y=True).fit(self.x_hf, self.hf)

        l1mean = gpr_lf.predict(self.x_hf)

        if self.embedding_theory:
            l1mean_shift1 = gpr_lf.predict(self.x_hf+0.02)
            l1mean_shift2 = gpr_lf.predict(self.x_hf+0.04)
            l2_train = np.hstack((self.x_hf, l1mean, l1mean_shift1, l1mean_shift2))
        else:
            l2_train = np.hstack((self.x_hf, l1mean))

        k_mf = kernel_hf * kernel_hf + kernel_hf
        gpr_mf_l2 = GaussianProcessRegressor(kernel=k_mf,
                                             n_restarts_optimizer=200,
                                             normalize_y=True).fit(l2_train, self.hf)

        pred_hf_mean = gpr_hf.predict(self.x, return_std=single)
        pred_lf_mean = gpr_lf.predict(self.x, return_std=single)

        if single:
            pred_lf_std = pred_lf_mean[1]
            pred_lf_mean = pred_lf_mean[0]
            pred_hf_std = pred_hf_mean[1]
            pred_hf_mean = pred_hf_mean[0]
        else:
            pred_lf_std = np.zeros(len(pred_lf_mean))
            pred_hf_std = np.zeros(len(pred_hf_mean))

        if self.embedding_theory:
            pred_lf_mean_shift1 = gpr_lf.predict(self.x + 0.02, return_std=False)
            pred_lf_mean_shift2 = gpr_lf.predict(self.x + 0.04, return_std=False)
            l2_test = np.hstack((self.x, pred_lf_mean, pred_lf_mean_shift1, pred_lf_mean_shift2))
        else:
            l2_test = np.hstack((self.x, pred_lf_mean))

        pred_mf_mean = gpr_mf_l2.predict(l2_test, return_std=single)

        if single:
            pred_mf_std = pred_mf_mean[1]
            pred_mf_mean = pred_mf_mean[0]

        else:
            pred_mf_std = np.zeros(len(pred_mf_mean))

        self.x = scaler.inverse_transform(self.x)
        pred_lf_mean = datascaler.inverse_transform(pred_lf_mean)
        pred_hf_mean = datascaler.inverse_transform(pred_hf_mean)
        pred_mf_mean = datascaler.inverse_transform(pred_mf_mean)
        if single:
            pred_lf_std *= datascaler.data_range_
            pred_hf_std *= datascaler.data_range_
            pred_mf_std *= datascaler.data_range_

        return self.x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std

    def mfmlp(self,
              hidden_layers1=(20, 50, 50, 50, 20),
              hidden_layers2=(20, 20, 20, 20),
              solver='lbfgs',
              activation='tanh'
              ):
        from sklearn.neural_network import MLPRegressor
        import numpy as np
        import warnings
        from sklearn.exceptions import DataConversionWarning
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)

        if len(np.shape(self.lf)) == 1:
            single = True
        else:
            single = False
            
        scaler, datascaler = self.prep()

        mlpr_lf = MLPRegressor(activation=activation,
                               hidden_layer_sizes=hidden_layers1,
                               solver=solver,
                               random_state=1,
                               alpha=0.0001,
                               max_iter=5000).fit(self.x_lf, self.lf)
        mlpr_hf = MLPRegressor(activation=activation,
                               hidden_layer_sizes=hidden_layers2,
                               solver=solver,
                               random_state=1,
                               alpha=0.0001,
                               max_iter=5000).fit(self.x_hf, self.hf)

        l1mean = mlpr_lf.predict(self.x_hf)
        if single:
            l1mean = l1mean.reshape(-1, 1)
            
        l2_train = np.hstack((self.x_hf, l1mean))
        
        if self.embedding_theory:
            l1mean_shift1 = mlpr_lf.predict(self.x_hf + 0.02)
            l1mean_shift2 = mlpr_lf.predict(self.x_hf + 0.04)
            if single:
                l1mean_shift1 = l1mean_shift1.reshape(-1, 1)
                l1mean_shift2 = l1mean_shift2.reshape(-1, 1)
            l2_train = np.concatenate((l2_train, l1mean_shift1, l1mean_shift2), axis=1)
            
        if self.gradient:
            l1mean_gradient = np.gradient(l1mean, axis=1)
            if single:
                l1mean_gradient = l1mean_gradient.reshape(-1, 1)
            l2_train = np.concatenate((l2_train, l1mean_gradient), axis=1)
                        
        mlpr_mf_nlin = MLPRegressor(activation=activation,
                                    hidden_layer_sizes=hidden_layers2,
                                    solver=solver,
                                    random_state=1,
                                    alpha=0.0001,
                                    max_iter=5000).fit(l2_train, self.hf)

        pred_hf_mean = mlpr_hf.predict(self.x)
        pred_lf_mean = mlpr_lf.predict(self.x)
        if single:
            pred_hf_mean = pred_hf_mean.reshape(-1, 1)
            pred_lf_mean = pred_lf_mean.reshape(-1, 1)

        l2_test = np.hstack((self.x, pred_lf_mean))

        if self.embedding_theory:
            pred_lf_mean_shift1 = mlpr_lf.predict(self.x + 0.02)
            pred_lf_mean_shift2 = mlpr_lf.predict(self.x + 0.04)
            if single:
                pred_lf_mean_shift1 = pred_lf_mean_shift1.reshape(-1, 1)
                pred_lf_mean_shift2 = pred_lf_mean_shift2.reshape(-1, 1)
            l2_test = np.concatenate((l2_test, pred_lf_mean_shift1, pred_lf_mean_shift2), axis=1)
            
        if self.gradient:
            pred_lf_mean_gradient = np.gradient(pred_lf_mean, axis=1)
            if single:
                pred_lf_mean_gradient = pred_lf_mean_gradient.reshape(-1, 1)
            l2_test = np.concatenate((l2_test, pred_lf_mean_gradient), axis=1)

        pred_mf_mean = mlpr_mf_nlin.predict(l2_test)
        if single:
            pred_mf_mean = pred_mf_mean.reshape(-1, 1)

        pred_lf_std = np.zeros(len(pred_lf_mean))
        pred_hf_std = np.zeros(len(pred_hf_mean))
        pred_mf_std = np.zeros(len(pred_mf_mean))

        self.x = scaler.inverse_transform(self.x)
        pred_lf_mean = datascaler.inverse_transform(pred_lf_mean)
        pred_hf_mean = datascaler.inverse_transform(pred_hf_mean)
        pred_mf_mean = datascaler.inverse_transform(pred_mf_mean)

        return self.x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std

    def mfmlp2(self, lf2, hf2, old_result):
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import StackingRegressor
        import numpy as np
        
        scaler, datascaler = self.prep()

        lf2 = lf2.T
        hf2 = hf2.T
        old_result = old_result.T
        lf2 = lf2.reshape(-1, 1)
        hf2 = hf2.reshape(-1, 1)
        old_result = old_result.reshape(-1, 1)
        lf2 = datascaler.transform(lf2)
        hf2 = datascaler.transform(hf2)
        old_result = datascaler.transform(old_result)

        solver = 'lbfgs'
        activation = 'tanh'
        # activation = 'relu'
        hidden_layers1 = (20, 50, 50, 50, 20)
        hidden_layers2 = (20, 20, 20, 20)

        mlpr_lf = MLPRegressor(activation=activation,
                               hidden_layer_sizes=hidden_layers1,
                               solver=solver,
                               random_state=1,
                               max_iter=1000).fit(np.hstack((self.x_lf, lf2)), self.lf)
        mlpr_hf = MLPRegressor(activation=activation,
                               hidden_layer_sizes=hidden_layers2,
                               solver=solver,
                               random_state=1,
                               max_iter=1000).fit(self.x_hf, self.hf)

        l1mean = mlpr_lf.predict(np.hstack((self.x_hf, hf2))).reshape(-1, 1)
        l1mean_shift1 = mlpr_lf.predict(np.hstack((self.x_hf+0.02, hf2))).reshape(-1, 1)
        l1mean_shift2 = mlpr_lf.predict(np.hstack((self.x_hf+0.04, hf2))).reshape(-1, 1)
        l2_train = np.hstack((self.x_hf, hf2, l1mean, l1mean_shift1, l1mean_shift2))
        # l2_train = np.hstack((x_hf, l1mean))

        mlpr_mf_nlin = MLPRegressor(activation=activation,
                                    hidden_layer_sizes=hidden_layers2,
                                    solver=solver,
                                    random_state=1,
                                    # alpha=0.0001,
                                    max_iter=1000).fit(l2_train, self.hf)

        pred_hf_mean = mlpr_hf.predict(self.x).reshape(-1, 1)
        pred_lf_mean = mlpr_lf.predict(np.hstack((self.x, old_result))).reshape(-1, 1)
        pred_lf_mean_shift1 = mlpr_lf.predict(np.hstack((self.x + 0.02, old_result))).reshape(-1, 1)
        pred_lf_mean_shift2 = mlpr_lf.predict(np.hstack((self.x + 0.04, old_result))).reshape(-1, 1)

        l2_test = np.hstack((self.x, old_result, pred_lf_mean, pred_lf_mean_shift1, pred_lf_mean_shift2))
        # l2_test = np.hstack((x, pred_lf_mean))
        pred_mf_mean = mlpr_mf_nlin.predict(l2_test)
        pred_mf_mean = pred_mf_mean.reshape(-1, 1)

        pred_lf_std = np.zeros(len(pred_lf_mean))
        pred_hf_std = np.zeros(len(pred_hf_mean))
        pred_mf_std = np.zeros(len(pred_mf_mean))

        self.x = scaler.inverse_transform(self.x)
        pred_lf_mean = datascaler.inverse_transform(pred_lf_mean)
        pred_hf_mean = datascaler.inverse_transform(pred_hf_mean)
        pred_mf_mean = datascaler.inverse_transform(pred_mf_mean)

        return self.x, pred_lf_mean, pred_lf_std, pred_hf_mean, pred_hf_std, pred_mf_mean, pred_mf_std
