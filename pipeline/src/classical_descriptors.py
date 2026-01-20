import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src.util.helpers as util
from src.util.logger import logger


def color_hist(piece, bins=(8, 4, 4), normalize=True):
    hsv = cv2.cvtColor(piece, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if normalize:
        hist = hist / np.sum(hist)

    return hist.flatten()


def precalculate_classical_descriptors(puzzle_path: str, hsv_bins=(8, 4, 4)):
    data = np.load(puzzle_path, allow_pickle=True)
    borders = {
        "top": data["borders_top"],
        "bottom": data["borders_bottom"],
        "left": data["borders_left"],
        "right": data["borders_right"],
    }

    P = int(data["P"])
    Q = int(data["Q"])
    D = hsv_bins[0] * hsv_bins[1] * hsv_bins[2]

    color_histograms = {
        "top": np.zeros((P, Q, D), dtype=np.float32),
        "bottom": np.zeros((P, Q, D), dtype=np.float32),
        "left": np.zeros((P, Q, D), dtype=np.float32),
        "right": np.zeros((P, Q, D), dtype=np.float32),
    }

    for r in range(P):
        for c in range(Q):
            for side in ["top", "bottom", "left", "right"]:
                border = borders[side][r, c]
                color_histograms[side][r, c] = color_hist(border, bins=hsv_bins, normalize=True)

    img_name = os.path.basename(os.path.dirname(os.path.dirname(puzzle_path)))
    os.makedirs(f"./data/processed/{img_name}/descriptors", exist_ok=True)

    np.savez_compressed(
        f"./data/processed/{img_name}/descriptors/classical_descriptors.npz",
        color_histograms_top=color_histograms["top"],
        color_histograms_bottom=color_histograms["bottom"],
        color_histograms_left=color_histograms["left"],
        color_histograms_right=color_histograms["right"],
    )
    logger.info(f"--Puzzle saved to ./data/processed/{img_name}/descriptors/classical_descriptors.npz--")

    util.plot_piece(borders["top"][0, 0], title="Puzzle Pieces")
    util.plot_color_hist(color_histograms["top"][0, 0], bins=hsv_bins)
    plt.show()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="calculate classical descriptors from a puzzle.")
    argParser.add_argument("--puzzle_path", type=str, required=True, help="Path to the input puzzle (.npz).")
    argParser.add_argument("--hsv_bins", type=tuple, default=(8, 4, 4), help="Number of bins for HSV color histogram.")

    args = argParser.parse_args()

    precalculate_classical_descriptors(
        puzzle_path=args.puzzle_path,
        hsv_bins=args.hsv_bins,
    )
