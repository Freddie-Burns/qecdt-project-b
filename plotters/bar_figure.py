"""
Plot and save figures of probability distributions for IQP circuits from graphs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Imports from my modules in root dir
import circuit
import graph
from plotters import theta_textbox

sns.set()


def main():
    graph_types = [
        graph.GraphTypes.path,
        graph.GraphTypes.cycle,
        # graph.Enum.star,
        # graph.Enum.complete,
    ]
    theta = np.pi / 4
    plot_figure(5, graph_types, theta)
    plt.show()


def plot_figure(num_qubits: int, graph_types: list, theta: float):
    """
    Plot and save figure of outcome probability distributions.

    :param num_qubits:      int, number of nodes in network, qubits in circuit
    :param graph_types:     list, graph.Enum properties e.g. [graph.Enum.path]
    :param theta:           float, theta value for all XX operations in IQP

    :return:                None, saves plot to figures dir
    """
    labels = [graph.LABEL[graph_type] for graph_type in graph_types]
    labels.sort()
    label_str = "_".join(labels)
    file_name = f"{num_qubits}_qubit_theta_{theta}_{label_str}.png"
    file_path = Path(__file__).parents[1] / 'figures' / file_name

    # Calculate probability distributions
    data_frames = []
    for graph_type in graph_types:
        # Create network representing circuit
        g = graph.gen_graph(num_qubits, graph_type)
        iqp_circuit = circuit.Circuit(g, theta, graph.LABEL[graph_type])
        data_frames.append(iqp_circuit.prob_distribution())
    data = pd.concat(data_frames)

    # Plot barchart
    sns.barplot(
        data=data,
        x=circuit.ColumnHeaders.bitstring,
        y=circuit.ColumnHeaders.probability,
        hue=circuit.ColumnHeaders.graph_type,
        alpha=0.6,
    )

    label_str = " & ".join(labels)
    plt.title(f"{num_qubits} qubit {label_str}")

    # Set bitstring labels on axis to read vertically upwards
    plt.xticks(rotation=90)

    # Pad bottom so x-axis label isn't cut off
    plt.gcf().subplots_adjust(bottom=0.2)

    # Display theta value in textbox
    theta_textbox(theta)

    # Save figure to specified location
    plt.savefig(file_path)


if __name__ == "__main__":
    main()
