import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="ticks")


def generate_animate_barplot(thetas, iqp_circuit, label_angle=90, title=None):
    """
    Return animate function for matplotlib animation.
    Animates a bar plot for a single circuit over varying theta.
    """
    def animate(i):
        plt.cla()
        ax = plt.gca()
        ax.set_ylim(0, 1)
        plt.xticks(rotation=label_angle)
        plt.title(title)

        theta = thetas[i]

        iqp_circuit.theta = theta
        data = iqp_circuit.prob_distribution()
        sns.barplot(data=data, x='bitstring', y='probability')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        pi_coeff = theta / np.pi
        label = f"$\\theta$ = {pi_coeff:.2f} $\\pi$"
        ax.text(
            0.95, 0.95,
            label,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=props,
        )

    return animate


def generate_animate_catplot(thetas, iqp_circuits, title=None):
    """
    Return animate function for matplotlib animation.
    Can animate multiple circuits prob distr over varying theta.
    """
    def animate(i):
        plt.clf()
        theta = thetas[i]

        data_frames = []
        for iqp_circuit in iqp_circuits:
            iqp_circuit.theta = theta
            data_frames.append(iqp_circuit.prob_distribution())
        data = pd.concat(data_frames)

        # Plot barchart
        sns.catplot(
            data=data,
            kind="bar",
            x="bitstring",
            y="probability",
            hue="network",
            palette="dark",
            alpha=0.6,
            height=6,
            legend_out=False,
        )

        # Set figure properties
        ax = plt.gca()
        ax.set_ylim(0, 1)
        plt.title(title)
        plt.xticks(rotation=90)

        # Pad bottom so x-axis label isn't cut off
        plt.gcf().subplots_adjust(bottom=0.15)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        pi_coeff = theta / np.pi
        label = f"$\\theta$ = {pi_coeff:.2f} $\\pi$"
        ax.text(
            0.95, 0.95,
            label,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=props,
        )

    return animate


def gen_network(n, edges):
    """
    Create networkx with n nodes and edges.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph
