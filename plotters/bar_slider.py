"""
Figure of probability distributions with slider for theta.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Button, Slider

# Imports from my modules in root dir
import circuit
import graph
from plotters import theta_textbox

sns.set(font_scale=1.6)


def main():
    num_qubits = 4
    graph_types = [
        graph.GraphTypes.star,
        graph.GraphTypes.complete,
        graph.GraphTypes.path,
        graph.GraphTypes.cycle,
    ]

    # Must assign instance or slider is garbage collected!
    bar_slider = BarSlider(num_qubits, graph_types)
    plt.show()


class BarSlider:
    """
    Produces a bar chart of probability distributions.
    Can vary theta with a slider.
    """
    def __init__(self, num_qubits: int, graph_types: list, num_theta: int = 101):
        """
        Create the figure and slider.
        Plot the first bar chart for theta = 0.
        """
        self.num_qubits = num_qubits
        self.graph_types = graph_types

        self.fig, self.ax = plt.subplots()

        # Calculate all the prob distr for display
        self.thetas = np.linspace(0, np.pi, num_theta)
        self.circuits = self.gen_circuits()
        self.data = self.calculate()

        self.slider = self.gen_slider()
        self.slider.on_changed(self.update)
        self.update(0)

    def calculate(self):
        """
        Calculate probability distributions
        """
        data_frames = []
        for theta in self.thetas:
            for c in self.circuits:
                c.theta = theta
                data_frames.append(c.prob_distribution())
        return pd.concat(data_frames)

    def gen_circuits(self):
        """
        Create the IQP circuits from graphs.
        These will be used to calculate the prob distributions.
        """
        circuits = []
        for graph_type in self.graph_types:
            g = graph.gen_graph(self.num_qubits, graph_type)
            c = circuit.Circuit(g, 0, graph.LABEL[graph_type])
            circuits.append(c)
        return circuits

    def gen_slider(self):
        """
        Make a horizontal slider to control the frequency.
        """
        ax_theta = self.fig.add_axes([0.15, 0.1, 0.75, 0.03])

        theta_slider = Slider(
            ax=ax_theta,
            label=f"theta",
            valmin=0,
            valmax=np.pi,
            valinit=0,
            valstep=self.thetas,
        )

        return theta_slider

    def set_title(self):
        """
        Set title for figure based on types of network graphs.
        """
        labels = [graph.LABEL[graph_type] for graph_type in self.graph_types]
        labels.sort()
        label_str = " & ".join(labels)
        plt.title(f"{self.num_qubits} qubit {label_str}")

    def update(self, theta):
        """
        Method called when slider is moved.
        """
        mask = self.data[circuit.ColumnHeaders.theta] == theta
        data = self.data[mask]

        # Set current axis as chart axis, clear it, set y-axis limits
        plt.sca(self.ax)
        plt.cla()
        self.ax.set_ylim(0, 1)
        self.set_title()

        sns.barplot(
            data=data,
            x=circuit.ColumnHeaders.bitstring,
            y=circuit.ColumnHeaders.probability,
            hue=circuit.ColumnHeaders.graph_type,
            alpha=0.6,
        )

        plt.xticks(rotation=90)
        self.ax.legend(loc="upper left")
        plt.gcf().subplots_adjust(bottom=0.25)
        theta_textbox(theta, fontsize=20)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    main()
