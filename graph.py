"""
Utils for working with networkx graphs.
"""

import networkx as nx
import seaborn as sns

sns.set(style="ticks")


class GraphTypes:
    """
    Enumerator for types of graph.
    """
    path = 0
    star = 1
    cycle = 2
    complete = 3


# Dict for creating graph labels for Dataframes and such
LABEL = {
    GraphTypes.path: "path",
    GraphTypes.star: "star",
    GraphTypes.cycle: "cycle",
    GraphTypes.complete: "complete",
}


# Dict for switch statements for nx.Graph creation functions
CREATE_GRAPH = {
    GraphTypes.path: nx.path_graph,
    GraphTypes.star: nx.star_graph,
    GraphTypes.cycle: nx.cycle_graph,
    GraphTypes.complete: nx.complete_graph,
}


def gen_graph(n, graph_type=None, edges=None, ):
    """
    Create networkx with n nodes.
    Define either by edges or graph type.
    """
    # Create standard graph of n nodes
    if graph_type is not None:
        # Star nx.star_graph creates n+1 nodes
        if graph_type is GraphTypes.star: n -= 1
        graph = CREATE_GRAPH[graph_type](n)

    # Create custom graph from specified edges
    elif edges is not None:
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)

    # Catch mistaken usages
    else:
        raise Exception("Either edges or graph type must be specified.")

    return graph
