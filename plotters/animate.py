from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, animation

# Imports from my modules in root dir
import circuit
import graph
from plotters import theta_textbox


def main():
    graph_types = [
        graph.GraphTypes.path,
        graph.GraphTypes.cycle,
        graph.GraphTypes.star,
        graph.GraphTypes.complete,
    ]
    animate_graph(4, graph_types)
    # plt.show()


def animate_graph(num_qubits, graph_types, thetas=None):
    """
    Animate bar chart prob distributions as theta evolves.

    :param num_qubits:      number of nodes in network, qubits in circuit
    :param graph_types:     list of graph.Enum properties e.g. [graph.Enum.path]
    :param thetas:          theta values for frames, default 65 values 0 to pi

    :return:                None, saves video to animations dir
    """
    labels = [graph.LABEL[graph_type] for graph_type in graph_types]
    labels.sort()
    label_str = "_".join(labels)
    file_name = f"{num_qubits}_qubit_{label_str}.mp4"
    file_path = Path(__file__).parents[1] / 'animations' / file_name

    iqp_circuits = []
    for graph_type in graph_types:
        # Create network representing circuit
        g = graph.gen_graph(num_qubits, graph_type=graph_type)
        iqp_circuits.append(circuit.Circuit(g, 0, graph.LABEL[graph_type]))

    # Default range of theta values to animate over
    if thetas is None:
        thetas = np.linspace(0, np.pi, 90)

    # Script to create figure, animate, and save
    fig = plt.figure()
    label_str = " & ".join(labels)
    title = f"{num_qubits} qubit {label_str}"
    animate = generate_animate_barplot(thetas, iqp_circuits, title)
    anim = animation.FuncAnimation(fig, animate, frames=len(thetas), repeat=True, blit=False, interval=100)
    anim.save(file_path, writer=animation.FFMpegWriter(fps=10))


def generate_animate_barplot(thetas, iqp_circuits, title=None):
    """
    Return animate function for matplotlib animation.
    Can animate multiple circuits prob distr over varying theta.
    """
    def animate(i):
        plt.cla()
        theta = thetas[i]

        data_frames = []
        for iqp_circuit in iqp_circuits:
            iqp_circuit.theta = theta
            data_frames.append(iqp_circuit.prob_distribution())
            print(f"theta: {theta:.2f}, circuit: {iqp_circuit.graph_type}")
        data = pd.concat(data_frames)

        # Plot barchart
        sns.barplot(
            data=data,
            x=circuit.ColumnHeaders.bitstring,
            y=circuit.ColumnHeaders.probability,
            hue=circuit.ColumnHeaders.graph_type,
            # palette="dark",
            alpha=0.6,
        )

        # Set figure properties
        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        plt.title(title)
        plt.xticks(rotation=90)

        # Pad bottom so x-axis label isn't cut off
        plt.gcf().subplots_adjust(bottom=0.2)

        # Display theta value in textbox
        theta_textbox(theta)

    return animate


if __name__ == "__main__":
    main()
