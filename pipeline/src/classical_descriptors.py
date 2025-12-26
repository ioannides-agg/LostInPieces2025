import argparse

import cv2
import numpy as np


def color_hist(piece, bins=(8, 4, 4), normalize=True):
    hsv = cv2.cvtColor(piece, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if normalize:
        hist = hist / np.sum(hist)

    return hist


def precalculate_classical_descriptors(puzzle_path: str):
    data = np.load(puzzle_path, allow_pickle=True)
    borders = data["borders"].item()
    # TODO: iterate over borders & tiles and calculate descriptors, save output to .npz


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="calculate classical descriptors from a puzzle.")
    argParser.add_argument("--puzzle_path", type=str, required=True, help="Path to the input puzzle (.npz).")

    args = argParser.parse_args()

    precalculate_classical_descriptors(
        puzzle_path=args.puzzle_path,
    )
