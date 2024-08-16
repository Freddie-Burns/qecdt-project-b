"""
Utils for working with networkx graphs.
"""

from pathlib import Path
import pickle

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


def gen_graph(n, graph_type=None, edges=None):
    """
    Create networkx with n nodes.
    Define either by edges or graph type.
    """
    # Create standard graph of n nodes
    if graph_type is not None:
        # Star nx.star_graph creates n+1 nodes
        if graph_type == GraphTypes.star: n -= 1
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


def pickle_graphs():
    """
    Save all unlabeled connected graphs up to 7 nodes as pkl files.
    """
    data_path = Path(__file__).parents[1] / 'data'
    graph_matcher = nx.isomorphism.vf2userfunc.GraphMatcher

    # 0, 1, 2 => no edges so ignore
    # atlas = nx.graph_atlas_g()[3:209]   208 is last 6 node graph
    atlas = nx.graph_atlas_g()

    union = nx.Graph()  # graph for union of all graphs in atlas
    all_graphs = []  # list to contain all unlabeled connected graphs

    for g in atlas:
        # check if connected
        if nx.number_connected_components(g) == 1:
            # check if isomorphic to a previous graph
            if not graph_matcher(union, g).subgraph_is_isomorphic():
                union = nx.disjoint_union(union, g)
                all_graphs.append(g)

    with open(data_path / f"atlas.pkl", 'wb') as f:
        pickle.dump(all_graphs, f)

    for n in (4, 5, 6, 7):
        with open(data_path / f"atlas_{n}.pkl", 'wb') as f:
            atlas_n = [g for g in all_graphs if len(g) == n]
            pickle.dump(atlas_n, f)
