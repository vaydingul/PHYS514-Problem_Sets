from GenAlgo import GenAlgo
from BrachCurve import BrachCurve
from Utils import cycloid
import numpy as np
import matplotlib.pyplot as plt


def mytests():

    # Create instance of GenAlgo
    gen_algo = GenAlgo(BrachCurve, n_population=10, mutation_rate=0.0001,  total_generation=1)
    # Hope to survive :p
    history = gen_algo.survive()

    ## Plotting routine ###
    plt.figure()
    history_sorted = sorted(history, key = lambda x: x[1])
    history_sorted = history_sorted[0:10]

    for dna in history_sorted:
        y = np.block([0.0, dna[0], 1.0])
        x = np.linspace(0.0 ,1.0, y.shape[0])
        plt.plot(x, y)

    x_sol, y_sol, T = cycloid(1, 1, 10)
    plt.plot(x_sol, y_sol, label = "Real solution", lw = 3, color = "red")
    plt.ylim([1.0, 0.0])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mytests()

