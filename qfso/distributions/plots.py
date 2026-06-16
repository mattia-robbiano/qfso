from __future__ import annotations


def plot_distributions(distributions, labels=None, title: str = ""):
    import matplotlib.pyplot as plt

    if not isinstance(distributions, list):
        distributions = [distributions]

    if labels is None:
        labels = [f"dist {index}" for index in range(len(distributions))]

    for index, distribution in enumerate(distributions):
        plt.bar(range(len(distribution)), distribution, alpha=0.5, label=labels[index])

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


__all__ = ["plot_distributions"]