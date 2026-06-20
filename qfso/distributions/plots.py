from __future__ import annotations


def plot_distributions(distributions, labels=None, title: str = ""):
    import matplotlib.pyplot as plt
    import numpy as np

    if not isinstance(distributions, list):
        distributions = [distributions]

    if labels is None:
        labels = [f"dist {index}" for index in range(len(distributions))]

    fig, ax = plt.subplots()
    
    for index, distribution in enumerate(distributions):
        dist_array = np.asarray(distribution)
        x = np.arange(len(dist_array))
        ax.bar(x, dist_array, alpha=0.5, label=labels[index])

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.tight_layout()
    plt.show()


__all__ = ["plot_distributions"]