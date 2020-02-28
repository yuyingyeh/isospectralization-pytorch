"""Plot a mesh."""
import argparse

import numpy as np
from matplotlib import pyplot as plt

# Configure Matplotlib
plt.rc("figure", titleweight="bold", dpi=100)
plt.rc("font", family="serif")
plt.rc("axes", labelweight="bold", linewidth=1.5, titleweight="bold")
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")


def parse_args():
    """Return the parsed command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file")
    parser.add_argument("output", help="Output file")
    args = parser.parse_args()
    return args.input, args.output


def main():
    """Main function."""
    in_file, out_file = parse_args()

    iterations = np.loadtxt(in_file, usecols=(0, 2, 3))

    plt.figure(figsize=(3, 2))
    plt.semilogy(iterations[:, 0], iterations[:, 1], label="Total loss")
    plt.semilogy(iterations[:, 0], iterations[:, 2], label="Eigenvalue alignment loss")
    plt.xlim(0, iterations[:, 0].max())
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(loc=1, prop={"size": 8})
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
