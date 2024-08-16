from pprint import pprint
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

import graph

# sns.set(style="ticks")


def main():
    graph_type = graph.GraphTypes.complete
    delays = []
    for n in range(1, 18):
        start = time()
        g = nx.complete_graph(n)
        cir = Circuit(g, np.pi/4, graph_type)
        cir.prob_distribution()
        delays.append(time() - start)
        print(n, delays[-1])
    plt.plot(delays)
    plt.xlabel('Qubits')
    plt.ylabel('Time / s')
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
    @profile
    def __init__(self, network, theta=np.pi, graph_type=None):
        """
        Build IQP circuit from networkx Graph and calculate output prob distr.
        """
        self.network = network          # networkx.Graph object defining circuit
        self.edges = tuple(network.edges)
        self.graph_type = graph_type    # e.g. line, ring, complete...
        self._theta = theta             # IQP theta for all gates

        self.n = len(network)       # number of vertices / qubits
        self.N = int(2 ** self.n)   # number of possible states / bitstrings

        self.exponents = self._gen_exponents()
        self.psi = self._gen_psi()  # state before final hadamard and mmnt

    def _exponent_sum_1(self, x):
        """
        Calculate the sum for the exponent required to evolve each qubit state.
        """
        x = gen_bitstring(x, self.n)
        return sum(map(lambda e: (-1) ** (int(x[e[0]]) + int(x[e[1]])), self.edges))

    def _exponent_sum_2(self, x):
        """
        Sum elements of bitstring corresponding to x according to edges.
        Uses bitwise logic for slight performance improvement.
        """
        # Each edge e has vertex i and j
        # Bitwise >> shifts elements i and j of bitstring to zeroth position
        # Bitwise XOR ^ between these elements
        # Double output with bitwise << 1
        # Give either 0 or 2 with bitwise & 2
        # Subtract 1 to output either +1 or -1 as required
        # Map over all edges and sum
        return sum(map((lambda e: ((((x >> e[0]) ^ (x >> e[1])) << 1) & 2) - 1), self.edges))

    def _exponent_sum_3(self, x):
        x = gen_bitstring(x, self.n)
        exponent = 0
        for i, j in self.edges:
            exponent += (-1) ** (int(x[i]) + int(x[j]))
        return exponent

    def _gen_exponents(self):
        """
        Generate the bitstring sums for the exponents once.
        """
        e = np.array([self._exponent_sum_3(x) for x in range(self.N)], dtype=complex)
        # e = np.array([self._exponent_sum_1(x) for x in range(self.N)], dtype=complex)
        # e = np.array([self._exponent_sum_2(x) for x in range(self.N)], dtype=complex)
        return e

    def _gen_psi(self):
        """
        Psi is the circuit state before final Hadamard and measurement.
        """
        return np.e ** (1j * self.theta * self.exponents)

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

    def prob_distribution(self):
        """
        Return probability of each bitstring outcome.
        """
        outcomes = list(range(self.N))
        bitstrings = [gen_bitstring(i, self.n) for i in outcomes]
        thetas = [self.theta] * self.N

        torch_psi = torch.asarray(self.psi)
        amps = hadamard_transform(torch_psi)
        probs = (np.abs(amps) ** 2) / (self.N ** 2)

        distribution = pd.DataFrame({
            ColumnHeaders.outcome: outcomes,
            ColumnHeaders.bitstring: bitstrings,
            ColumnHeaders.probability: probs,
            ColumnHeaders.theta: thetas,
            ColumnHeaders.graph_type: self.graph_type,
            ColumnHeaders.num_qubits: self.n,
        })

        prob_header = ColumnHeaders.probability
        distribution[prob_header] /= distribution[prob_header].sum()
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


class HadamardProfiling:
    """
    Determine which hadamard function is quickest.
    """
    def __init__(self):
        self.cir: Circuit = None

    def profile_function(self, lims=(3, 12)):
        """
        Time and plot different methods for hadamard operation.
        """
        had_times = []
        fwht_times = []
        torch_times = []
        for n in range(*lims):
            network = nx.complete_graph(n)
            self.cir = Circuit(network, theta=np.pi / 4)
            h, f, t = self.profile_hadamard_methods()
            had_times.append(h)
            fwht_times.append(f)
            torch_times.append(t)
        plt.plot(had_times, label="mat mul")
        plt.plot(fwht_times, label="fhwt")
        plt.plot(torch_times, label="torch")
        plt.legend()
        plt.show()

    def profile_torch(self, lims=(3, 16)):
        """
        Time and plot pytorch hadamard operation.
        """
        torch_times = []
        for n in range(*lims):
            network = nx.complete_graph(n)
            self.cir = Circuit(network, theta=np.pi / 4)
            start = time()
            _ = self.torch_prob()
            delay = time() - start
            torch_times.append(delay)
            print(n, delay)
        plt.plot(torch_times, label="torch")
        plt.legend()
        # plt.show()

    def profile_hadamard_methods(self):
        """
        Time different methods for hadamard operation.
        """
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
        _ = self.torch_prob()
        torch_times.append(time() - start)

        return had_times, fwht_times, torch_times

    def hadamard_mul_prob(self):
        """
        First method to profile.
        Calculate Hadamard product of all bitstrings.
        """
        hadamard = HadamardSingleton.get(self.cir.N)
        probs = np.matmul(hadamard, self.cir.psi)
        return (np.abs(probs) ** 2) / (self.cir.N ** 2)

    def fwht_prob(self):
        """
        Second method to profile.
        Calculate FWHT of all bitstrings.
        """
        hadamard_output = fwht(self.cir.psi)
        result = np.array(hadamard_output, dtype=np.complex128)
        return (np.abs(result) ** 2) / (self.cir.N ** 2)

    def torch_prob(self):
        """
        Fourth method to profile.
        Calculate pytorch FWHT of all bitstrings.
        """
        psi = torch.asarray(self.cir.psi)
        result = hadamard_transform(psi)
        return (np.abs(result) ** 2) / (self.cir.N ** 2)


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
