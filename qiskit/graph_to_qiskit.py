"""
Turn networkx graph into corresponding IQP circuit.
Useful for verification of direct calculation approach.
"""

from pprint import pprint

from qiskit import QuantumRegister, QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import networkx as nx
from numpy import pi
from matplotlib import pyplot as plt

GRAPH = nx.path_graph(3)
THETA = pi
n = len(GRAPH)

# Correct theta for Qiskit definition
# https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZGate
THETA *= -2

qreg_q = QuantumRegister(n, 'q')
circuit = QuantumCircuit(qreg_q)

for i in range(n):
    circuit.h(qreg_q[i])

circuit.barrier(*qreg_q)

for i, j in GRAPH.edges():
    circuit.cx(qreg_q[i], qreg_q[j])
    circuit.rz(THETA, qreg_q[j])
    circuit.cx(qreg_q[i], qreg_q[j])
    circuit.barrier(*qreg_q)


# for i in range(n):
#     circuit.h(qreg_q[i])

backend = Aer.get_backend('statevector_simulator')
outputstate = backend.run(circuit, shots=1).result().get_statevector()
amps = [f"{a:.2f}" for a in outputstate]
pprint(amps)

probs = Statevector(outputstate).probabilities()
probstrings = [f"{p:.2f}" for p in probs]
bitstrings = [f"{i:b}".zfill(n) for i in range(2 ** n)]
pprint(list(zip(bitstrings, probstrings)))

total = sum(probs)
print(f"Total prob: {total:.2f}")

nx.draw(GRAPH, with_labels=True)

plt.figure()
plt.bar(bitstrings, probs)
plt.xticks(rotation=90)

circuit.draw(output='mpl')
plt.show()
