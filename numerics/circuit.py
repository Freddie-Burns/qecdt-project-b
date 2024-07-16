from sympy import fwht
import numpy as np
import networkx as nx


class Circuit:
    def __init__(self, n=1, edges=None, theta=0):
        self.edges = edges          # j, k pairs of vertices for each edge in the graph
        self.n = n                  # number of vertices
        self.theta = theta          # IQP theta for all gates, "coupling strength"
        self.graph = nx.Graph()     # graph of vertices & edges defining IQP circuit
        self._update_graph()         # add edges to graph

    def _exponent_sum(self, bitstring):
        """Calculate the sum for the exponent required to evolve each qubit's state."""
        exponent = 0
        for j, k in self.edges:
            exponent += (-1) ** (bitstring[j] + bitstring[k])
        return exponent

    def _update_graph(self):
        """Ensure all edges are added to the graph."""
        for j, k in self.edges:
            self.graph.add_edge(j, k)

    def draw_graph(self):
        """Display graph using matplotlib."""
        nx.draw_circular(self.graph, with_labels=True, font_weight='bold')

    def gen_psi(self):
        """Psi is the circuit state before final Hadamard and measurement."""
        psi = np.zeros(self.n)
        for i in range(self.n):
            psi[i] = np.e ** (1j * self.theta * self._exponent_sum(bin(i)[:2]))
        return psi
