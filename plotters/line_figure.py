import matplotlib.pyplot as plt
import seaborn as sns

from circuit import Circuit, gen_prob_dataframe
import graph


def plot_prob_for_theta():
    """
    Create plot of prob dist
    """
    n = 5
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    ring_graph = graph.gen_network(n, edges)
    cir = Circuit(ring_graph, 1)
    df = gen_prob_dataframe()
    sns.barplot(x="bitstring", y="probability", data=df)
    plt.show()


def plot_prob_vs_theta(file_name=False):
    """
    Create plot of prob dist
    """
    n = 5
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    ring_graph = graph.gen_network(n, edges)
    cir = Circuit(ring_graph)
    df = gen_prob_dataframe()

    for theta in np.linspace(0, np.pi/2, 65):
        cir.theta = theta                       # set new theta which generates new psi
        prob_distr = cir.prob_distribution()    # probability of each outcome

        # update dataframe
        if df.empty: df = prob_distr
        else: df = pd.concat([df, prob_distr])

    sns.relplot(
        data=df,
        kind='scatter',
        x='theta',
        y='probability',
        hue='bitstring',
        legend='brief',
    )

    if file_name:
        plt.savefig(file_name)

    plt.show()
