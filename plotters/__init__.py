import matplotlib.pyplot as plt
import numpy as np


def theta_textbox(theta, ax=None, fontsize=14):
    """
    Display value of theta in textbox on plot in terms of pi.
    """
    pi_coeff = theta / np.pi
    label = f"$\\theta$ = {pi_coeff:.2f} $\\pi$"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    if ax is None:
        ax = plt.gca()

    ax.text(
        0.95, 0.95,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props,
    )
