"""
Calculate probability difference vectors.
Plot heatmap of similarity matrix.
For a given pair of graphs plot similarity vs theta.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

import circuit
import graph


# sns.set()


DATA_PATH = Path(__file__).parent / "data"
FIG_PATH = Path(__file__).parent / "figures"


def main():
    with open(DATA_PATH / "atlas_4.pkl", "rb") as f:
        atlas = pickle.load(f)

    # Plot the pretty networks graphs.
    fig, axes = plt.subplots(nrows=2, ncols=3)
    axes = axes.flatten()
    for i, g in enumerate(atlas):
        plt.sca(axes[i])
        nx.draw(g)
        plt.title(g.name)
    plt.savefig(FIG_PATH / "4_vertex_graphs.png")

    circuits = [circuit.Circuit(g, 0, g.name) for g in atlas]
    thetas = np.linspace(0, np.pi, 101)

    # For each graph find its similarity to the others over a range of theta.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle("Similarity of probability distributions between 4 vertex graphs")
    for i in range(len(atlas)):
        c0 = circuits[i]
        arr = []
        for c in circuits:
            data = []
            for theta in thetas:
                data.append(calculate_similarity(c0, c, theta))
            arr.append(data)

        sim = np.array(arr)

        # Plot the similarities as a colourmap.
        ax = axes[i]
        plt.sca(ax)
        plt.imshow(sim, interpolation='none', aspect='auto')
        plt.colorbar()
        plt.xlabel(r"theta / $\pi$")
        plt.title(c0.graph_type)

        dim = len(thetas)
        ax.set_xticks([0, dim // 2, dim])
        ax.set_yticks(list(range(len(atlas))))
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_yticklabels([g.name for g in atlas])

    plt.savefig(FIG_PATH / "4_vertex_graph_similarities.png")
    plt.show()


def plot_similarity(g1, g2, data, thetas):
    """
    Line plot of similarity against theta for two graphs.
    """
    data = np.array(data)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    plt.sca(ax1)
    nx.draw(g1)
    plt.sca(ax2)
    nx.draw(g2)

    plt.figure()
    plt.plot(thetas / np.pi, data)
    plt.xlabel("theta / pi")
    plt.ylabel("similarity")
    plt.show()


def calculate_similarity(c1, c2, theta):
    """
    Calculate similarity between prob distribution of two circuits.
    """
    c1.theta = theta
    c2.theta = theta

    data1: pd.DataFrame = c1.prob_distribution()
    data2: pd.DataFrame = c2.prob_distribution()

    data1.sort_values(by=circuit.ColumnHeaders.outcome)
    data2.sort_values(by=circuit.ColumnHeaders.outcome)

    probs1 = data1[circuit.ColumnHeaders.probability]
    probs2 = data2[circuit.ColumnHeaders.probability]

    probs1 /= np.linalg.norm(probs1)
    probs2 /= np.linalg.norm(probs2)

    return np.inner(probs1, probs2)


class Similarity:
    """
    Calculate similarity between probability distributions for two circuits.
    """
    def __init__(self, circuits=None, graphs=None):
        if circuits is not None:
            self.circuits = circuits
        elif graphs is not None:
            self.circuits = [circuit.Circuit(g, 0, g.name) for g in graphs]
        else:
            raise ValueError("No circuits or graphs given.")

    def calculate(self, thetas=None):
        if thetas is None:
            thetas = np.linspace(0, np.pi, 101)


if __name__ == "__main__":
    main()
