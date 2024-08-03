import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import circuit
import graph


def main():
    plot_prob_vs_theta(4)
    plt.show()


def plot_prob_vs_theta(n=3, file_name=False):
    """
    Create plot of prob dist
    """
    cycle_graph = graph.gen_graph(n, graph.GraphTypes.cycle)
    cir = circuit.Circuit(cycle_graph, 1)
    data = cir.prob_distribution()
    for theta in np.linspace(0, np.pi/2, 65):
        cir.theta = theta
        prob_distr = cir.prob_distribution()

        # update dataframe
        if data.empty: data = prob_distr
        else: data = pd.concat([data, prob_distr])

    sns.relplot(
        data=data,
        kind='scatter',
        x=circuit.ColumnHeaders.theta,
        y=circuit.ColumnHeaders.probability,
        hue=circuit.ColumnHeaders.bitstring,
        legend='brief',
    )

    if file_name:
        plt.savefig(file_name)


if __name__ == '__main__':
    main()
