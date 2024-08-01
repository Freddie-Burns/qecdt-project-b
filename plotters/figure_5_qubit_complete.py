from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import circuit


sns.set(style='ticks')


THETA = 2
FILE_PATH = Path(__file__).parents[1] / 'figures' / f"5_qubit_complete_theta_{THETA}.png"
NUM_QUBITS = 5
NETWORK = nx.complete_graph(NUM_QUBITS)


iqp_circuit = circuit.Circuit(NETWORK, THETA)
data = iqp_circuit.prob_distribution()

fig = plt.figure(figsize=(8, 6))
ax = plt.gca()

sns.barplot(data=data, x='bitstring', y='probability')
plt.xticks(rotation=90)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
label = f"$\\theta$ = {THETA:.2f}"
ax.text(
    0.8, 0.95,
    label,
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=props,
)

fig.savefig(FILE_PATH)

plt.figure(figsize=(8, 6))
nx.draw_circular(NETWORK, with_labels=True)
plt.show()
