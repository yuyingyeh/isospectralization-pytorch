"""Batch plot a folder of ply files into figures."""
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
    parser.add_argument("src", help="Source folder")
    parser.add_argument("-d", "--dst", help="Destination folder")
    parser.add_argument("-t", "--target", help="Target shape directory")
    args = parser.parse_args()
    return args.src, args.dst, args.target


def load_ply(filename):
    """Load ply data."""
    plydata = PlyData.read(filename)
    v = np.array([(v[0], v[1], v[2]) for v in plydata.elements[0].data])
    t = np.array([t[0] for t in plydata.elements[1].data])
    return v, t


def main():
    """Main function."""
    src, dst, target = parse_args()
    src = os.path.normpath(src)

    if dst is not None:
        dst = os.path.normpath(dst)
        os.makedirs(dst, exist_ok=True)
    else:
        os.makedirs(os.path.join(os.path.dirname(src), "png"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(src), "pdf"), exist_ok=True)

    if target is not None:
        target_v = np.loadtxt(os.path.join(target, "mesh.vert"))
        target_t = np.loadtxt(os.path.join(target, "mesh.triv"))  - 1

    for filename in os.listdir(src):
        if not os.path.isfile(os.path.join(src, filename)):
            continue

        v, t = load_ply(os.path.join(src, filename))
        plt.figure(figsize=(6, 6))
        if target is not None:
            plt.triplot(target_v[:, 0], target_v[:, 1], target_t)
        plt.triplot(v[:, 0], v[:, 1], t)
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.axis("off")
        if dst is not None:
            filepath = os.path.join(dst, os.path.splitext(filename)[0])
            plt.savefig(filepath + ".png", bbox_inches="tight")
            plt.savefig(filepath + ".pdf", bbox_inches="tight")
        else:
            for ext in ("png", "pdf"):
                filename_ = os.path.splitext(filename)[0] + "." + ext
                filepath = os.path.join(os.path.dirname(src), ext, filename_)
                plt.savefig(filepath, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
