# BEN 2026-11-01
# Change aim: Histogram of the needle targeting error from the C++ and Python trial CSV files.
# File: plot_results.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1)[:, 3]


def main():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 2.5, 51)
    for name, label in [("cpp", "C++"), ("py", "Python")]:
        ax.hist(load(f"results/trials_{name}.csv"), bins=bins, alpha=0.55, label=label)
    ax.set_xlabel("needle targeting error (mm), simulated")
    ax.set_ylabel("trials")
    ax.set_title("Targeting error over 5000 simulated trials")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/targeting_error_hist.png", dpi=130)


if __name__ == "__main__":
    main()
