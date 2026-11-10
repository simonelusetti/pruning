from pruning.plots.common import plot_supervised


def plot(curves, output, trigger=None):
    if not curves:
        return False
    reference = next((curve for curve in curves if curve.run.path == trigger),
                     max(curves, key=lambda curve: curve.run.launched_on))
    matched = [curve for curve in curves if curve.config.benchmark == reference.config.benchmark]
    return plot_supervised(matched, output, "Neural network pruning comparison")
