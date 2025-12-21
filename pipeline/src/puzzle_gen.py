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


def generate_puzzle(img_path: str, P: int, Q: int, suppress_logs: bool = False):
    if suppress_logs:
        logger.disabled = True

    img = cv2.imread(img_path)
    pieces, aP, aQ = generate_puzzle_pieces(img, P, Q)
    logger.info("--Puzzle pieces generated--")
    pieces, permutation = shuffle_puzzle_pieces(pieces, aP, aQ)
    logger.info(f"--Puzzle pieces shuffled with permutation: {permutation}--")
    pieces, rotations = rotate_puzzle_piece(pieces, aP, aQ)
    logger.info(f"--Puzzle pieces rotated with rotations: {[int(i * 90) for i in rotations.flatten()]}--")

    util.plot_pieces(pieces, P=aP, Q=aQ, title="Shuffled and Rotated Puzzle Pieces")


if __name__ == "__main__":
    generate_puzzle(
        img_path="./data/raw/tree512x512.jpg",
        P=3,
        Q=3,
        suppress_logs=False,
    )
