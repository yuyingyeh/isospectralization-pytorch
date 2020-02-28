"""Plot a mesh."""
import argparse
import os
import os.path

import numpy as np
from matplotlib import pyplot as plt
from plyfile import PlyData

# Configure Matplotlib
plt.rc("figure", titleweight="bold", dpi=100)
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


def load_ply(filename):
    """Load ply data."""
    plydata = PlyData.read(filename)
    v = np.array([(v[0], v[1], v[2]) for v in plydata.elements[0].data])
    t = np.array([t[0] for t in plydata.elements[1].data])
    return v, t


def main():
    """Main function."""
    in_file, out_file = parse_args()

    if os.path.isfile(in_file):
        v, t = load_ply(in_file)
    else:
        v = np.loadtxt(os.path.join(in_file, "mesh.vert"))
        t = np.loadtxt(os.path.join(in_file, "mesh.triv")) - 1

    plt.figure(figsize=(6, 6))
    plt.triplot(v[:, 0], v[:, 1], t)
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.axis("off")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
