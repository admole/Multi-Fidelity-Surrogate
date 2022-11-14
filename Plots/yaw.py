#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fonts
import json
import forces
import fields
import cf
from mfRegression import MFRegress


def get_yaw(model):
    # Opening JSON file
    j = open(os.path.join(os.getcwd(), f"../Data/{model}/Yaw/inlet_sweep.json"))
    case_settings = json.load(j)
    numcases = len(case_settings)
    path = f"{model}/Yaw/"
    cd1 = cd2 = np.zeros(numcases)
    cd2 = np.zeros(numcases)
    cz1 = np.zeros(numcases)
    cz2 = np.zeros(numcases)
    recirc = np.zeros(numcases)
    probe = np.zeros(numcases)
    alpha = np.zeros(numcases)
    for i in range(numcases):
        case = os.path.join(path, case_settings[i]["Name"])
        alpha[i] = case_settings[i]["FlowAngle"]
        cd1[i] = forces.get_cd(case, 'cube1')
        cz1[i] = forces.get_cl(case, 'cube1')
        cd2[i] = forces.get_cd(case, 'cube2')
        cz2[i] = forces.get_cl(case, 'cube2')
        if model == 'remove':
            recirc[i] = cf.recirculation(case)
        probe[i] = fields.get_probe(case, position=[6.0, 0.6], field='U')

    data = {r'$\alpha$': alpha,
            r'$Cd_1$': cd1, r'$Cz_1$': cz1,
            r'$Cd_2$': cd2, r'$Cz_2$': cz2,
            r'$A_{recirc}$': recirc,
            'Probe': probe}
    df = pd.DataFrame(data=data)
    return df


def plot_yaw(ax, ax2, rans, les, variable, model):
    print(f'\nRunning for {variable}')
    les_train = les[les[r"$\alpha$"] % 10 == 0]
    les_test = les[les[r"$\alpha$"] % 10 == 5]
    ax.scatter(rans[r'$\alpha$'], rans[variable], edgecolors='r', facecolors='none', label=f'RANS Sample')

    regress = MFRegress(rans[r'$\alpha$'].to_numpy(),
                        rans[variable].to_numpy(),
                        les_train[r'$\alpha$'].to_numpy(),
                        les_train[variable].to_numpy())
    if model == 'GPR':
        from sklearn.gaussian_process.kernels import (Matern, DotProduct)
        alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfgp(kernel_lf=Matern(),
                                                                                      kernel_hf=DotProduct()*Matern())
    else:
        from sklearn.metrics import mean_squared_error
        import random
        n_runs = 60
        scores = []
        scores_hf = []
        architectures = []
        while len(scores) < n_runs:

            lf_hidden_layers = []
            hf_hidden_layers = []
            for layer in range(random.randint(1, 8)):
                lf_hidden_layers.append(10*random.randint(1, 8))
            for layer in range(random.randint(1, 8)):
                hf_hidden_layers.append(10*random.randint(1, 8))

            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            score_hf = 0
            score = 0
            for train_index, test_index in loo.split(les_train[r'$\alpha$']):

                regress = MFRegress(rans[r'$\alpha$'].to_numpy(),
                                    rans[variable].to_numpy(),
                                    les_train[r'$\alpha$'].to_numpy()[train_index],
                                    les_train[variable].to_numpy()[train_index])

                alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp(hidden_layers1=tuple(lf_hidden_layers),
                                                                                               hidden_layers2=tuple(hf_hidden_layers),)

                kscore_hf = mean_squared_error(les_train[variable].to_numpy()[test_index],
                                              les_mean[alpha == les_train[r'$\alpha$'].to_numpy()[test_index]])
                kscore = mean_squared_error(les_train[variable].to_numpy()[test_index],
                                           mf_mean[alpha == les_train[r'$\alpha$'].to_numpy()[test_index]])
                score_hf += kscore_hf
                score += kscore
            scores.append(score)
            scores_hf.append(score)
            architectures.append((lf_hidden_layers, hf_hidden_layers))

        best_run = np.argmin(np.abs(scores))
        worst_run = np.argmax(np.abs(scores))
        print('Best configuration')
        print(f'iteration= {best_run}')
        print(f'score = {scores[best_run]}')
        print(f'architecture = {architectures[best_run]}')
        print('worst configuration')
        print(f'iteration= {worst_run}')
        print(f'score = {scores[worst_run]}')
        print(f'architecture = {architectures[worst_run]}')

        # rerun regression with optimal MF set-up
        regress = MFRegress(rans[r'$\alpha$'].to_numpy(),
                            rans[variable].to_numpy(),
                            les_train[r'$\alpha$'].to_numpy(),
                            les_train[variable].to_numpy())

        alpha, rans_mean, rans_std, les_mean, les_std, mf_mean, mf_std = regress.mfmlp(
            hidden_layers1=tuple(architectures[best_run][0]),
            hidden_layers2=tuple(architectures[best_run][1]), )

        # rerun regression with optimal HF set-up
        regress = MFRegress(rans[r'$\alpha$'].to_numpy(),
                            rans[variable].to_numpy(),
                            les_train[r'$\alpha$'].to_numpy(),
                            les_train[variable].to_numpy())

        les_mean = regress.mfmlp(
            hidden_layers1=tuple(architectures[best_run][0]),
            hidden_layers2=tuple(architectures[np.argmin(np.abs(scores_hf))][1]), )[3]

    ax.plot(alpha, rans_mean, 'r--', label=f'RANS Only {model}')
    ax.fill_between(alpha[:, 0], rans_mean[:, 0] - rans_std, rans_mean[:, 0] + rans_std, alpha=0.2, color='r')
    ax.scatter(les_test[r'$\alpha$'], les_test[variable], color=[0.2, 0.7, 1], label='LES Sample (testing)')
    ax.scatter(les_train[r'$\alpha$'], les_train[variable], edgecolors='b', facecolors='none', label=f'LES Sample (training)')
    ax.plot(alpha, les_mean, 'b--', label=f'LES Only {model}')
    ax.fill_between(alpha[:, 0], les_mean[:, 0] - les_std, les_mean[:, 0] + les_std, alpha=0.2, color='b')
    ax.plot(alpha, mf_mean, 'k', label=f'Multi-Fidelity {model}')
    ax.fill_between(alpha[:, 0], mf_mean[:, 0] - mf_std, mf_mean[:, 0] + mf_std,
                    alpha=0.2, color='k')# , label="Model +/- 1 std")

    ax.set_ylabel(variable)
    ax.set_xlim(0, max(alpha))
    ax.legend(frameon=False, ncol=2, columnspacing=0.5)

    ax2.plot(rans_mean, mf_mean, 'k', label='Multi-Fidelity')
    ax2.set_ylabel(r'$\mathcal{Y}_H$')
    ax2.legend(frameon=False)


def main():
    variables = [r'$Cd_1$', r'$Cz_1$', r'$Cd_2$', r'$Cz_2$', 'Probe']
    variables = [r'$Cd_2$', 'Probe']

    fig1, axes1 = plt.subplots(len(variables), 2, figsize=(13, 3.2*len(variables)),
                               squeeze=False, constrained_layout=True, sharex='col', sharey='none',
                               gridspec_kw={'width_ratios': [2, 1]})

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpr', action='store_true', help='Use GPR (default is MLP)')
    args = parser.parse_args()
    if args.gpr:
        model = 'GPR'
    else:
        model = 'MLP'

    RANS_data = get_yaw('RANS')
    LES_data = get_yaw('LES')
    for it in range(len(variables)):
        quantity = variables[it]
        plot_yaw(axes1[it, 0], axes1[it, 1], RANS_data, LES_data, quantity, model)

    axes1[-1, 0].set_xlabel(r'$\alpha$')
    axes1[-1, 1].set_xlabel(r'$\mathcal{Y}_L$')
    plt.show()
    fig1.savefig(f'figures/yaw_{model}.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
