import matplotlib.pyplot as plt
import numpy as np

def spaghetti_plot(x, y, n_samples=20, indices=None, ax=None, plot_kwargs=None):
    """
    Plots x against a few picked examples of y.

    Parameters
    ----------
    x: 1d array
        The values for the x axis
    y: 2d array.
        First axis (0) is the samples axis.
        Second axis (1) are the values for the y axis.
    n_samples: int (default 20)
        How many sampled values to plot. `n_samples` are
        selected uniformly at random.
    indices: 1d array
        Indices of the specific samples that will be selected
        for plotting.
    ax: matplotlib.Axes
        The axes where the figure will be plotted.
    plot_kwargs: dict
        Extra arguments passed to `plt.plot`

    Returns
    -------
    matplotlib.Axes

    """
    has_indices = indices is not None
    has_samples = bool(n_samples)
    if has_indices == has_samples:
        _msg = "Exactly one of `n_samples` or `indices` must be specified"
        raise ValueError(_msg)

    if has_samples:
        indices = np.random.choice(range(y.shape[1]), n_samples)

    ax = ax or plt.gca()
    for idx in indices:
        ax.plot(x, y[idx], **(plot_kwargs or {}))

    return ax


def ribbon_plot(x, y, n_ribbons=10, percentile_min=1, percentile_max=99, ribbon_color='r', plot_median=True, line_color='k', ax=None, fill_kwargs=None, line_kwargs=None):
    """
    Make a ribbon plot that shows the different quantiles of the
    distribution of y against x.

    Parameters
    ----------
    x: 1d array
        The values for the x axis
    y: 2d array
    n_ribbons: int (default 10)
        How many quantiles to show
    percentile_min: float, between 0 and 50
        The lowest percentile to be shown
    percentile_max: float between 50 and 100
        The highest percentile to show.
    ribbon_color: str (default 'r')
        Color for the ribbons. Must be a valid expression for
        matplotlib colors.
    plot_median: bool (default True)
        Whether or not to plot a line for the 50% percentile
    line_color: str (default 'k')
        Color to use for the median.
    ax: matplotlib.Axes
        Where to plot the figure.
    fill_kwargs: dict
        Extra arguments passed to `plt.fill_between`.
        Controls the aspect of the ribbons.
    line_kwargs: dict
        Extra arguments to be passed to `plt.plot`.
        Controls the aspect of the median line.

    Returns
    -------
    matplotlib.Axes
    """
    perc1 = np.percentile(y, np.linspace(
        percentile_min, 50, num=n_ribbons, endpoint=False), axis=0)
    perc2 = np.percentile(y, np.linspace(
        50, percentile_max, num=n_ribbons + 1)[1:], axis=0)
    fill_kwargs = fill_kwargs or {}
    line_kwargs = line_kwargs or {}
    alpha = fill_kwargs.pop('alpha', 1 / n_ribbons)
    ax = ax or plt.gca()
    plt.sca(ax)

    # fill ribbons
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha,
                         color=ribbon_color, **(fill_kwargs or {}))

    if plot_median:
        plot_func = plt.step if fill_kwargs.pop("step", None) else plt.plot
        plot_func(x, np.median(y, axis=0),
                  color=line_color, **(line_kwargs or {}))

    return plt.gca()