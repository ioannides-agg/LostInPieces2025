import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.util.logger import logger


def find_closest_divisor(n: int, target: int) -> int:
    divisors = [i for i in range(2, n + 1) if n % i == 0]
    logger.info(f"Divisors of {n}: {divisors}")

    closest = min(divisors, key=lambda x: abs(x - target))

    return closest


def reverse_permutation(pieces: NDArray, permutation: NDArray) -> NDArray:
    P, Q, h, w, C = pieces.shape
    flat_pieces = pieces.reshape(P * Q, h, w, C)
    unshuffled_flat = np.empty_like(flat_pieces)
    unshuffled_flat[permutation] = flat_pieces
    unshuffled_grid = unshuffled_flat.reshape(P, Q, h, w, C)

    return unshuffled_grid


def apply_permutation(pieces: NDArray, permutation: NDArray) -> NDArray:
    P, Q, h, w, C = pieces.shape
    flat_pieces = pieces.reshape(P * Q, h, w, C)
    shuffled_flat = flat_pieces[permutation]
    shuffled_grid = shuffled_flat.reshape(P, Q, h, w, C)

    return shuffled_grid


def rotate_patch(patch, k):
    return np.rot90(patch, k=k, axes=(0, 1))


def apply_rotation(pieces: NDArray, rotations: NDArray) -> NDArray:
    P = len(pieces)
    Q = len(pieces[0])
    rotated_pieces = [[None for _ in range(Q)] for _ in range(P)]
    for r in range(P):
        for c in range(Q):
            rotated_pieces[r][c] = rotate_patch(pieces[r, c], rotations[r, c])

    return np.array(rotated_pieces)


def reverse_rotation(pieces: NDArray, rotations: NDArray) -> NDArray:
    P = len(pieces)
    Q = len(pieces[0])
    rotated_pieces = [[None for _ in range(Q)] for _ in range(P)]
    for r in range(P):
        for c in range(Q):
            rotated_pieces[r][c] = rotate_patch(pieces[r, c], -1 * rotations[r, c])

    return np.array(rotated_pieces)


def plot_pieces(pieces: NDArray, P: int, Q: int, title: str, borders: dict | None = None):
    plt.figure(figsize=(Q, P))
    plt.suptitle(title)

    red = np.array([0, 0, 255], dtype=np.uint8)

    for r in range(P):
        for c in range(Q):
            piece = pieces[r, c].copy()

            if borders is not None:
                red = np.array([0, 0, 255], dtype=np.uint8)
                mask = np.zeros_like(piece, dtype=np.uint8)

                mask[: borders["top"].shape[2], :, :] = red
                mask[-borders["bottom"].shape[2] :, :, :] = red
                mask[:, : borders["left"].shape[3], :] = red
                mask[:, -borders["right"].shape[3] :, :] = red

                alpha = 0.5

                piece = cv2.addWeighted(piece, alpha, mask, alpha, 0)

            plt.subplot(P, Q, r * Q + c + 1)
            plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
            plt.axis("off")

    plt.tight_layout()
    plt.show()
