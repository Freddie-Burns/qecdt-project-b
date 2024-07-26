from pprint import pprint
from sympy import fwht
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="ticks")


def main():
    n = 3
    edges = [[0, 1], [1, 2], [0, 2]]
    cir = Circuit(n=n, edges=edges)
    df = gen_prob_dataframe()

    for theta in np.linspace(0, np.pi/2, 65):
        cir.settheta(theta)                    # set new theta which generates new psi
        prob_distr = cir.prob_distribution()    # probability of each outcome

        # update dataframe
        if df.empty: df = prob_distr
        else: df = pd.concat([df, prob_distr])

    sns.relplot(
        data=df,
        kind='scatter',
        x='theta',
        y='probability',
        hue='outcome',
        legend='brief',
    )
    plt.savefig("prob_distr_fig.png")
    plt.show()


class Circuit:
    def __init__(self, graph, theta=np.pi):
        """
        Build IQP circuit from networkx object and calculate output prob distributions.
        """
        self._graph = graph         # networkx object defining circuit
        self._theta = theta         # IQP theta for all gates, "coupling strength"

        self.n = len(graph)         # number of vertices / qubits
        self.N = int(2 ** self.n)   # number of possible states / bitstrings
        self.psi = self._gen_psi()  # state before final hadamard and mmnt

    def _exponent_sum(self, bitstring):
        """
        Calculate the sum for the exponent required to evolve each qubit's state.
        """
        exponent = 0
        for j, k in self._graph.edges:
            exponent += (-1) ** (int(bitstring[j]) + int(bitstring[k]))
        return exponent

    def _gen_psi(self):
        """
        Psi is the circuit state before final Hadamard and measurement.
        """
        psi = np.array([-1] * self.N, dtype=np.complex128)
        for i in range(self.N):
            bitstring = gen_bitstring(i, self.n)
            psi[i] = np.e ** (1j * self.theta * self._exponent_sum(bitstring))
        return psi

    def _update_graph(self):
        """
        Ensure all edges are added to the graph.
        """
        for j, k in self._graph.edges:
            self._graph.add_edge(j, k)

    def draw_graph(self, circular=False):
        """
        Display graph using matplotlib.
        """
        if circular:
            nx.draw_circular(self._graph, with_labels=True, font_weight='bold')
        else:
            nx.draw(self._graph, with_labels=True, font_weight='bold')
        plt.show()

    def output_prob(self, output):
        """
        Calculate probability of a given output.
        Output can be given as int or str.
        """
        # Convert bitstring to int value
        if type(output) is str:
            ouput = int(output, 2)

        # Create vector repr bitstring
        mmnt_vector = np.zeros(self.N)
        mmnt_vector[output] = 1

        # Apply Hadamard transform to mmnt vector
        hadamard_output = fwht(mmnt_vector)
        hadamard_output_complex = np.array(hadamard_output, dtype=np.complex128)

        # Inner product of state and transformed output gives its probability
        result = np.inner(hadamard_output_complex, self.psi)
        return np.abs(result) ** 2

    def prob_distribution(self):
        """
        Return probability of each bitstring outcome.
        """
        outcomes = list(range(self.N))
        probabilities = []
        thetas = [self.theta] * self.N

        for i in outcomes:
            probabilities.append(self.output_prob(i))

        distribution = pd.DataFrame({
            "outcome": outcomes,
            "probability": probabilities,
            "theta": thetas,
        })
        return distribution

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        """
        Update state psi when theta is changed.
        """
        self.theta = theta
        self.psi = self._gen_psi()


def circuit_from_graph(graph):
    return Circuit(n=len(graph), edges=graph.edges)


def gen_prob_dataframe():
    """
    Create blank dataframes for probability distributions, n bit values.
    """
    return pd.DataFrame(columns=("outcome", "probability", "theta"))


def gen_bitstring(x, n):
    """
    Generate bitstring of length n for number x.
    """
    # Pad with leading zeroes to ensure length n
    return bin(x)[2:].zfill(n)


if __name__ == "__main__":
    main()
