import argparse
import os

import cv2
import numpy as np
from numpy.typing import NDArray

import src.util.helpers as util
from src.util.logger import logger


def generate_puzzle_pieces(image: NDArray, target_P: int, target_Q: int) -> tuple[NDArray, int, int]:
    H, W, C = image.shape

    P = util.find_closest_divisor(H, target_P)
    Q = util.find_closest_divisor(W, target_Q)

    logger.info(f"target_P: {target_P}, target_Q: {target_Q}, chosen_P: {P}, chosen_Q: {Q}")

    piece_height = H // P
    piece_width = W // Q
    pieces = []

    for i in range(P):
        pieces_row = []
        for j in range(Q):
            y_start = i * piece_height
            y_end = (i + 1) * piece_height if i != P - 1 else H
            x_start = j * piece_width
            x_end = (j + 1) * piece_width if j != Q - 1 else W

            piece = image[y_start:y_end, x_start:x_end]
            pieces_row.append(piece)

        pieces.append(pieces_row)

    return np.array(pieces), P, Q


def shuffle_puzzle_pieces(pieces: NDArray, P: int, Q: int) -> tuple[NDArray, NDArray]:
    permutation = np.random.permutation(P * Q)
    shuffled_pieces = util.apply_permutation(pieces, permutation)
    return shuffled_pieces, permutation


def rotate_puzzle_piece(piece: NDArray, P: int, Q: int) -> tuple[NDArray, NDArray]:
    rotations = np.random.randint(0, 4, size=(P, Q))

    rotated_pieces = util.apply_rotation(piece, rotations)
    return rotated_pieces, rotations


def extract_borders(pieces, Wb):
    P, Q, h, w, C = pieces.shape
    borders = {
        "top": np.zeros((P, Q, Wb, w, C), dtype=pieces.dtype),
        "bottom": np.zeros((P, Q, Wb, w, C), dtype=pieces.dtype),
        "left": np.zeros((P, Q, h, Wb, C), dtype=pieces.dtype),
        "right": np.zeros((P, Q, h, Wb, C), dtype=pieces.dtype),
    }

    for r in range(P):
        for c in range(Q):
            piece = pieces[r, c]
            borders["top"][r, c] = piece[:Wb, :, :]
            borders["bottom"][r, c] = piece[-Wb:, :, :]
            borders["left"][r, c] = piece[:, :Wb, :]
            borders["right"][r, c] = piece[:, -Wb:, :]

    return borders


def generate_puzzle(img_path: str, P: int, Q: int, Wb: int):
    img = cv2.imread(img_path)
    pieces, aP, aQ = generate_puzzle_pieces(img, P, Q)
    logger.info(f"--Puzzle pieces generated: {pieces.shape}--")
    pieces, permutation = shuffle_puzzle_pieces(pieces, aP, aQ)
    logger.info(f"--Puzzle pieces shuffled with permutation: {permutation}--")
    pieces, rotations = rotate_puzzle_piece(pieces, aP, aQ)
    logger.info(f"--Puzzle pieces rotated with rotations: {[int(i * 90) for i in rotations.flatten()]}--")

    Wb = max(Wb, min(pieces.shape[2], pieces.shape[3]) // 8)
    borders = extract_borders(pieces, Wb=Wb)
    logger.info(
        f"--Borders extracted from puzzle pieces: Wb: {Wb} top: {borders['top'].shape}, bottom: {borders['bottom'].shape}, left: {borders['left'].shape}, right: {borders['right'].shape}--"
    )

    util.plot_pieces(pieces, P=aP, Q=aQ, title="Shuffled and Rotated Puzzle Pieces", borders=borders)

    img_name = img_path.split("/")[-1].split(".")[0]
    os.makedirs(f"./data/processed/{img_name}", exist_ok=True)

    np.savez_compressed(
        f"./data/processed/{img_name}/{img_name}_puzzle_P{aP}_Q{aQ}.npz",
        pieces=pieces,
        permutation=permutation,
        rotations=rotations,
        borders=borders,
    )
    logger.info(f"--Puzzle saved to data/processed/{img_name}_puzzle_P{aP}_Q{aQ}.npz--")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Generate a jigsaw puzzle from an image.")
    argParser.add_argument("--img_path", type=str, required=True, help="Path to the input image.")
    argParser.add_argument("--P", type=int, required=True, help="Number of pieces along the height.")
    argParser.add_argument("--Q", type=int, required=True, help="Number of pieces along the width.")

    args = argParser.parse_args()

    generate_puzzle(
        img_path=args.img_path,
        P=args.P,
        Q=args.Q,
        Wb=3,
    )
