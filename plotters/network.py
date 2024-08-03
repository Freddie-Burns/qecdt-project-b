"""
Experimenting with displaying network graphs.
"""

import matplotlib.pyplot as plt
import networkx as nx

# Import my graph module from root dir
import graph


def main():
    path_graph = graph.gen_graph(3, graph_type=graph.GraphTypes.path)
    star_graph = graph.gen_graph(3, graph_type=graph.GraphTypes.star)

    plt.figure()
    plt.title("Path Graph")
    nx.draw_networkx(path_graph, with_labels=True)

    plt.figure()
    plt.title("Star Graph")
    nx.draw_networkx(star_graph, with_labels=True)
    plt.show()


if __name__ == "__main__":
    main()
