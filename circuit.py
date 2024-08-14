from pprint import pprint
import itertools
from time import time

from hadamard_transform import hadamard_transform
from line_profiler_pycharm import profile
from sympy import fwht
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
# import seaborn as sns

# import graph

# sns.set(style="ticks")


@profile
def main():
    had_times = []
    fwht_times = []
    torch_times = []
    for n in range(1, 16):
        network = nx.complete_graph(n)
        cir = Circuit(network, theta=np.pi/4)
        h, f, t = cir.prob_distribution()
        had_times.append(h)
        fwht_times.append(f)
        torch_times.append(t)
    plt.plot(had_times, label="mat mul")
    plt.plot(fwht_times, label="fhwt")
    plt.plot(torch_times, label="torch")
    plt.legend()
    plt.show()


class ColumnHeaders:
    """
    Reference for column headers of DataFrames.
    """
    outcome = "Outcome"
    bitstring = "Bitstring"
    probability = "Probability"
    theta = "Theta"
    graph_type = "Graph type"
    num_qubits = "Number of qubits"


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
            output = int(output, 2)

        # Create vector repr bitstring
        mmnt_vector = np.zeros(self.N)
        mmnt_vector[output] = 1

        # Apply Hadamard transform to mmnt vector
        hadamard_output = fwht(mmnt_vector)
        hadamard_output_complex = np.array(hadamard_output, dtype=np.complex128)

        # Inner product of state and transformed output gives its probability
        result = np.inner(hadamard_output_complex, self.psi)
        return (np.abs(result) ** 2) / (self.N ** 2)

    @profile
    def prob_distribution(self):
        """
        Return probability of each bitstring outcome.
        """
        outcomes = list(range(self.N))
        bitstrings = [gen_bitstring(i, self.n) for i in outcomes]
        thetas = [self.theta] * self.N

        had_times = []
        fwht_times = []
        torch_times = []

        start = time()
        self.hadamard_mul_prob()
        had_times.append(time() - start)

        start = time()
        self.fwht_prob()
        fwht_times.append(time() - start)

        start = time()
        self.fwht_prob()
        torch_times.append(time() - start)

        # start = time()
        # self.list_comp_prob(outcomes)
        # list_times.append(time() - start)

        probs = [0] * self.N
        distribution = pd.DataFrame({
            ColumnHeaders.outcome: outcomes,
            ColumnHeaders.bitstring: bitstrings,
            ColumnHeaders.probability: probs,
            ColumnHeaders.theta: thetas,
            ColumnHeaders.graph_type: self.graph_type,
            ColumnHeaders.num_qubits: self.n,
        })

        # prob_header = ColumnHeaders.probability
        # distribution[prob_header] /= distribution[prob_header].sum()
        # return distribution

        return had_times, fwht_times, torch_times

    def hadamard_mul_prob(self):
        """
        First method to profile.
        Calculate Hadamard product of all bitstrings.
        """
        hadamard = HadamardSingleton.get(self.N)
        probs = np.matmul(hadamard, self.psi)
        return (np.abs(probs) ** 2) / (self.N ** 2)

    def fwht_prob(self):
        """
        Second method to profile.
        Calculate FWHT of all bitstrings.
        """
        hadamard_output = fwht(self.psi)
        result = np.array(hadamard_output, dtype=np.complex128)
        return (np.abs(result) ** 2) / (self.N ** 2)

    def list_comp_prob(self, outcomes):
        """
        Third method to profile.
        Calculate list of all bitstrings probabilities.
        """
        return [self.output_prob(output) for output in outcomes]

    def torch_prob(self):
        """
        Fourth method to profile.
        Calculate pytorch FWHT of all bitstrings.
        """
        result = hadamard_transform(self.psi)
        return (np.abs(result) ** 2) / (self.N ** 2)

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


class HadamardSingleton:
    """
    Prevent repeated calculation of large Hadamard matrices.
    """
    matrix = {}

    @classmethod
    def get(cls, N):
        """
        Return the hadamard of size N x N.
        """
        if N in cls.matrix.keys():
            return cls.matrix[N]
        else:
            cls.matrix[N] = hadamard(N)
            return cls.matrix[N]


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
