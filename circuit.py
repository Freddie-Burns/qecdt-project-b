from pprint import pprint

from sympy import fwht
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="ticks")


class ColumnHeaders:
    """
    Reference for column headers of DataFrames.
    """
    outcome = "outcome"
    bitstring = "bitstring"
    probability = "probability"
    theta = "theta"
    graph_type = "graph type"


class Circuit:
    def __init__(self, network, theta=np.pi, graph_type=None):
        """
        Build IQP circuit from networkx Graph and calculate output prob distr.
        """
        self.network = network          # networkx.Graph object defining circuit
        self.graph_type = graph_type    # e.g. line, ring, complete...
        self._theta = theta             # IQP theta for all gates

        self.n = len(network)       # number of vertices / qubits
        self.N = int(2 ** self.n)   # number of possible states / bitstrings
        self.psi = self._gen_psi()  # state before final hadamard and mmnt

    def _exponent_sum(self, bitstring):
        """
        Calculate the sum for the exponent required to evolve each qubit state.
        """
        exponent = 0
        for j, k in self.network.edges:
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
        for j, k in self.network.edges:
            self.network.add_edge(j, k)

    def draw_graph(self, circular=False):
        """
        Display graph using matplotlib.
        """
        if circular:
            nx.draw_circular(self.network, with_labels=True, font_weight='bold')
        else:
            nx.draw(self.network, with_labels=True, font_weight='bold')
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
        bitstrings = [gen_bitstring(i, self.n) for i in outcomes]
        probabilities = []
        thetas = [self.theta] * self.N

        for i in outcomes:
            probabilities.append(self.output_prob(i))

        distribution = pd.DataFrame({
            ColumnHeaders.outcome: outcomes,
            ColumnHeaders.bitstring: bitstrings,
            ColumnHeaders.probability: probabilities,
            ColumnHeaders.theta: thetas,
            ColumnHeaders.graph_type: self.graph_type,
        })

        distribution["probability"] /= distribution["probability"].sum()
        return distribution

    @property
    def theta(self):
        """
        Theta is a property to ensure psi is updated if theta is changed.
        """
        return self._theta

    @theta.setter
    def theta(self, theta):
        """
        Update state psi when theta is changed.
        """
        self._theta = theta
        self.psi = self._gen_psi()


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
