"""Plot the eigenvalue sequences stored in a txt file."""
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
    parser.add_argument("-k", type=int, default=30, help="Number of eigenvalues")
    parser.add_argument("--init", help="Input file for initial sequence")
    parser.add_argument("--target", help="Input file for target sequence")
    args = parser.parse_args()
    return args.input, args.output, args.k, args.init, args.target

def main():
    """Main function."""
    in_file, out_file, k, init_file, target_file = parse_args()

    in_eig = np.loadtxt(in_file)[:k]
    if init_file is not None:
        init_eig = np.loadtxt(init_file)[:k]
    if target_file is not None:
        target_eig = np.loadtxt(target_file)[:k]

    plt.figure(figsize=(3, 2))
    if init_file is not None:
        plt.plot(np.arange(k), init_eig, color='0.5', linewidth=2, label="Initial")
    plt.plot(np.arange(k), in_eig, color='b', linewidth=4, label="Final")
    if target_file is not None:
        plt.plot(np.arange(k), target_eig, color='r', linewidth=2, label="Target")
    plt.xlim(0, len(in_eig))
    y_max = in_eig.max()
    if init_file is not None and init_eig.max() > y_max:
        y_max = init_eig.max()
    if target_file is not None and target_eig.max() > y_max:
        y_max = target_eig.max()
    plt.ylim(0, y_max)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.legend(loc=0, prop={"size": 8})
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
