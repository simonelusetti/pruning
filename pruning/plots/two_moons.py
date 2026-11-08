from pruning.plots.common import plot_supervised


def plot(curves, output, trigger=None):
    return plot_supervised(curves, output, "Two moons: pruning comparison")
