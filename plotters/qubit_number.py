"""
Plot probability of outcomes against qubit number.
Start with all zeroes or all ones for simplicity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import circuit
import graph
from plotters import theta_textbox


GRAPH_TYPES = [
    graph.GraphTypes.path,
    graph.GraphTypes.cycle,
    graph.GraphTypes.star,
    graph.GraphTypes.complete,
]

THETA = np.random.rand()
N_ARRAY = np.arange(1, 11)

FILE_NAME = f"prob_all_zeroes_theta_{THETA:.2f}.png"
FILE_PATH = Path(__file__).parents[1] / 'figures' / FILE_NAME


def main():
    for graph_type in GRAPH_TYPES:
        probs = []
        for n in N_ARRAY:
            bitstring = "0" * n
            g = graph.gen_graph(n, graph_type)
            cir = circuit.Circuit(g, THETA)
            probs.append(cir.output_prob(bitstring))
        label = graph.LABEL[graph_type]
        plt.plot(N_ARRAY, probs, label=label, alpha=0.5)
    theta_textbox(THETA)
    plt.legend()
    plt.xlabel("Number of qubits")
    plt.ylabel("Probability")
    plt.title("Probability of all zeros")
    plt.savefig(FILE_PATH)
    plt.show()


def plot_prob_vs_theta(n, bitstring, file_name=False):
    """
    Create plot of prob dist against qubit number for a given output bitstring.
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
