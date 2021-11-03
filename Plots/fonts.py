# Fonts

import matplotlib as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.serif": ["Computer Modern"]})

TINY_SIZE = 16
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIG_SIZE = 26
BIGGER_SIZE = 40
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)     # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
