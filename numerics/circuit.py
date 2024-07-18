from sympy import fwht
import numpy as np
import networkx as nx


def main():
    edges = [[0, 1]]
    cir = Circuit(n=2, edges=edges, theta=np.pi/4)
    print(np.round(cir.psi, 2))
    print(cir.output_prob("00"))
    print(cir.output_prob("01"))
    print(cir.output_prob("10"))
    print(cir.output_prob("11"))


class Circuit:
    def __init__(self, n=1, edges=(), theta=0):
        """Building IQP circuits from networks and calculating output prob distributions."""
        self.edges = edges          # j, k pairs of vertices for each edge in the graph
        self.n = n                  # number of vertices / qubits
        self.N = int(2 ** n)        # number of possible states / bitstrings
        self.theta = theta          # IQP theta for all gates, "coupling strength"
        self.psi = self._gen_psi()  # state before final hadamard and mmnt

        self.graph = nx.Graph()     # graph of vertices & edges defining IQP circuit
        self._update_graph()        # add edges to graph

    def _exponent_sum(self, bitstring):
        """Calculate the sum for the exponent required to evolve each qubit's state."""
        exponent = 0
        for j, k in self.edges:
            exponent += (-1) ** (int(bitstring[j]) + int(bitstring[k]))
        return exponent

    def _gen_psi(self):
        """Psi is the circuit state before final Hadamard and measurement."""
        psi = np.array([-1] * self.N, dtype=np.complex128)
        for i in range(self.N):
            bitstring = bin(i)[2:].zfill(self.n)  # ensure bitstring n long / pad with leading zeroes
            psi[i] = np.e ** (1j * self.theta * self._exponent_sum(bitstring))
        return psi

    def _update_graph(self):
        """Ensure all edges are added to the graph."""
        for j, k in self.edges:
            self.graph.add_edge(j, k)

    def draw_graph(self):
        """Display graph using matplotlib."""
        nx.draw_circular(self.graph, with_labels=True, font_weight='bold')

    def output_prob(self, bitstring):
        # Convert bitstring to int value
        bitstring = int(bitstring, 2)
        # Create vector repr bitstring
        mmnt_vector = np.zeros(self.N)
        mmnt_vector[bitstring] = 1
        # Apply Hadamard transform to mmnt vector
        hadamard_output = fwht(mmnt_vector)
        hadamard_output = np.array(hadamard_output, dtype=np.complex128)
        # Inner product of state and transformed output gives its probability
        result = np.inner(hadamard_output, self.psi) / self.N
        return np.real(result)


if __name__ == "__main__":
    main()
